# -*- coding: utf-8 -*-
# 作用:
#   定义 MutPred_v2 自监督图神经网络模型, 在口袋/全局图上执行 masked amino-acid 预测。
#   模型使用数值较稳定的 EGNN 在 3D 坐标上做消息传递, 并通过门控机制融合结构特征 (x_struct)
#   与 ESM-2 特征 (x_esm)。
#
# 输入:
#   - 初始化:
#       from models.mutpred_v2_model import MutPredV2Model
#
#       model = MutPredV2Model(
#           struct_dim=graph["x"].shape[1],          # 结构特征维度
#           esm_dim=graph["x_esm"].shape[1],         # ESM 特征维度
#           edge_dim=graph["edge_attr"].shape[1],    # 边特征维度
#           hidden_dim=256,
#           num_layers=4,
#           num_classes=20,
#       )
#
#   - 自监督前向 (masked AA):
#       logits = model(
#           x_struct=graph["x"],            # [N, F_struct]
#           x_esm=graph["x_esm"],           # [N, D_esm]
#           edge_index=graph["edge_index"], # [2, E]
#           edge_attr=graph["edge_attr"],   # [E, F_edge]
#           pos=graph["pos"],               # [N, 3]
#       )
#
#   - 获取节点表示 (用于湿实验监督头):
#       logits, h = model.forward_with_repr(
#           x_struct=graph["x"],
#           x_esm=graph["x_esm"],
#           edge_index=graph["edge_index"],
#           edge_attr=graph["edge_attr"],
#           pos=graph["pos"],
#       )
#
# 输出:
#   - forward: logits: FloatTensor [N, num_classes]
#   - forward_with_repr: (logits, h)
#       logits: [N, num_classes]
#       h:      [N, hidden_dim]  final_norm 之后、head 之前的节点 embedding

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EGNNLayer(nn.Module):
    """
    EGNN 层 (简化版 E(n) 等变图神经网络, 使用数值稳定的坐标更新形式):

    输入:
        h:          [N, H]       节点特征
        pos:        [N, 3]       节点 3D 坐标
        edge_index: [2, E]       边索引 (src, dst)
        edge_attr:  [E, De]      边特征 (不含几何信息, 例如 edge_type, Δresseq, Δchain)

    更新规则 (对每条边 i -> j):
        d_ij^2 = ||x_j - x_i||^2
        m_ij   = φ_e(h_i, h_j, d_ij^2, e_ij)          # 边消息
        s_ij   = φ_x(m_ij)                            # 坐标更新系数 (标量, 经过 tanh 和缩放)

        坐标更新 (使用单位方向向量, 防止步长随距离爆炸):
            u_ij = (x_j - x_i) / (||x_j - x_i|| + eps)
            Δx_j = Σ_i u_ij * s_ij * coord_scale
            x_j' = x_j + Δx_j

        特征更新:
            m_j  = Σ_i m_ij
            h_j' = LN( h_j + φ_h(h_j, m_j) )
    """

    def __init__(self, hidden_dim: int, edge_dim: int, coord_scale: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.coord_scale = coord_scale

        # 边消息 MLP: [h_i, h_j, d_ij^2, edge_attr] -> m_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 坐标更新系数: m_ij -> s_ij (标量)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 节点更新: [h_j, m_j] -> h_j'
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            h_out:  [N, H]  更新后的节点特征
            pos_out:[N, 3]  更新后的坐标
        """
        src, dst = edge_index  # [E], [E]

        pos_src = pos[src]                # [E, 3]
        pos_dst = pos[dst]                # [E, 3]
        diff = pos_dst - pos_src          # [E, 3]
        dist2 = (diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # 为数值稳定性加一个上界, 避免极端情况下 d^2 过大
        dist2 = torch.clamp(dist2, min=0.0, max=1000.0)

        # 单位方向向量 u_ij, 防止步长随距离线性放大
        direction = diff / (torch.sqrt(dist2 + 1e-8))

        # 计算边消息 m_ij
        h_src = h[src]                    # [E, H]
        h_dst = h[dst]                    # [E, H]
        edge_input = torch.cat([h_src, h_dst, dist2, edge_attr], dim=-1)  # [E, 2H+1+De]
        m_ij = self.edge_mlp(edge_input)  # [E, H]

        # 坐标更新系数: 先通过 MLP, 再 tanh 压缩到 [-1,1], 再乘 coord_scale
        coord_coef = self.coord_mlp(m_ij)          # [E, 1]
        coord_coef = torch.tanh(coord_coef)        # 保证有界
        coord_coef = coord_coef * self.coord_scale # 控制整体步长

        trans = direction * coord_coef             # [E, 3]

        # 累积到目标节点坐标
        delta_pos = torch.zeros_like(pos)          # [N, 3]
        delta_pos.index_add_(0, dst, trans)
        pos_out = pos + delta_pos

        # 节点特征更新
        agg_msg = torch.zeros_like(h)              # [N, H]
        agg_msg.index_add_(0, dst, m_ij)

        h_update = self.node_mlp(torch.cat([h, agg_msg], dim=-1))  # [N, H]
        h_out = self.node_norm(h + h_update)

        return h_out, pos_out


class MutPredV2Model(nn.Module):
    """
    MutPred_v2 主模型 (EGNN + 门控融合版本):

    流程:
        1) 结构特征 x_struct 和 ESM 特征 x_esm 分别映射到同一 hidden_dim 空间;
        2) 使用 per-node 门控网络计算 gate ∈ (0,1), 决定结构/ESM 的相对权重;
        3) 以融合后的 h0 和 pos 为输入, 通过多层 EGNNLayer 做消息传递与坐标更新;
        4) 最后通过 LayerNorm + 线性层输出 20 维氨基酸 logits。

    额外接口:
        - forward_with_repr: 在 forward 基础上同时返回节点 embedding h。
    """

    def __init__(
        self,
        struct_dim: int,
        esm_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 20,
        dropout: float = 0.1,
        esm_dropout: float = 0.2,
    ):
        super().__init__()
        self.struct_dim = struct_dim
        self.esm_dim = esm_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 结构 / ESM 各自投影到同一 hidden_dim 空间
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)

        # 门控网络: 输入 [h_struct || h_esm] -> gate ∈ (0,1)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 可学习缩放系数, 用于微调结构 / ESM 初始相对权重
        self.struct_scale = nn.Parameter(torch.tensor(1.0))
        self.esm_scale = nn.Parameter(torch.tensor(1.0))

        # 训练阶段对 ESM 特征做 dropout, 防止模型完全依赖 ESM
        self.esm_dropout = nn.Dropout(esm_dropout)

        # 多层 EGNN
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def _encode(
        self,
        x_struct: torch.Tensor,
        x_esm: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        内部编码函数:
            输入结构特征、ESM 特征和图结构, 返回:
                - h:       [N, hidden_dim]  (final_norm 之后)
                - cur_pos: [N, 3]           更新后的坐标
        """
        # 1) 结构特征投影
        h_struct = self.struct_proj(x_struct)  # [N, H]

        # 2) ESM 特征投影 + dropout (仅训练时生效)
        if self.training and self.esm_dim > 0:
            x_esm_in = self.esm_dropout(x_esm)
        else:
            x_esm_in = x_esm
        h_esm = self.esm_proj(x_esm_in)        # [N, H]

        # 3) 门控融合: gate 越大越偏向结构, 越小越偏向 ESM
        fusion_input = torch.cat([h_struct, h_esm], dim=-1)  # [N, 2H]
        gate = self.fusion_gate(fusion_input)                # [N, 1], 元素 ∈ (0,1)

        h = gate * (self.struct_scale * h_struct) + (1.0 - gate) * (self.esm_scale * h_esm)
        h = F.relu(h)

        # 4) 多层 EGNN 消息传递 (同时更新 h 和 pos)
        cur_pos = pos
        for layer in self.layers:
            h, cur_pos = layer(h, cur_pos, edge_index, edge_attr)
            h = self.dropout(h)

        # 5) 归一化
        h = self.final_norm(h)
        return h, cur_pos

    def forward(
        self,
        x_struct: torch.Tensor,     # [N, F_struct]
        x_esm: torch.Tensor,        # [N, D_esm]
        edge_index: torch.Tensor,   # [2, E]
        edge_attr: torch.Tensor,    # [E, F_edge]
        pos: torch.Tensor,          # [N, 3]
    ) -> torch.Tensor:
        """
        自监督 masked AA 任务使用的前向接口: 仅返回 logits。
        """
        h, _ = self._encode(x_struct, x_esm, edge_index, edge_attr, pos)
        logits = self.head(h)  # [N, num_classes]
        return logits

    def forward_with_repr(
        self,
        x_struct: torch.Tensor,     # [N, F_struct]
        x_esm: torch.Tensor,        # [N, D_esm]
        edge_index: torch.Tensor,   # [2, E]
        edge_attr: torch.Tensor,    # [E, F_edge]
        pos: torch.Tensor,          # [N, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        用于湿实验监督头的前向接口:
            返回:
                logits: [N, num_classes]
                h:      [N, hidden_dim]  (final_norm 之后、head 之前的节点 embedding)
        """
        h, _ = self._encode(x_struct, x_esm, edge_index, edge_attr, pos)
        logits = self.head(h)
        return logits, h
