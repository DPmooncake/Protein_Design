# -*- coding: utf-8 -*-
"""
mutpred_v2_model.py

作用:
    定义 MutPred_v2 的自监督图神经网络模型, 用于在口袋图上做 masked
    amino-acid 预测。模型同时融合:
        1) 结构/类型节点特征 x (来自 build_pockets_and_graphs.py);
        2) ESM-2 残基级别 embedding x_esm。

输入:
    forward(
        x_struct: FloatTensor [N, F_struct],   # 图中节点结构+类型特征
        x_esm:    FloatTensor [N, D_esm],      # 节点对应的 ESM embedding, 非蛋白节点为 0
        edge_index: LongTensor [2, E],         # 有向边索引 (src, dst)
        edge_attr:  FloatTensor [E, F_edge],   # 边特征
    )

输出:
    logits: FloatTensor [N, num_classes]       # 每个节点对应 20 种氨基酸的 logits

调用格式:
    from models.mutpred_v2_model import MutPredV2Model

    model = MutPredV2Model(
        struct_dim=graph["x"].shape[1],
        esm_dim=graph["x_esm"].shape[1],
        edge_dim=graph["edge_attr"].shape[1],
        hidden_dim=256,
        num_layers=4,
        num_classes=20,
    )

    logits = model(x_struct, x_esm, edge_index, edge_attr)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNNLayer(nn.Module):
    """
    简单的基于 edge_index 的消息传递层:
        m_ij = MLP_msg( h_i, e_ij )
        h_j' = MLP_upd( h_j, sum_i m_ij )
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # [N, H]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, De]
    ) -> torch.Tensor:
        src, dst = edge_index  # [E], [E]

        # 消息计算
        m_input = torch.cat([h[src], edge_attr], dim=-1)  # [E, H+De]
        m = self.msg_mlp(m_input)                         # [E, H]

        # 按 dst 聚合
        agg = torch.zeros_like(h)                         # [N, H]
        agg.index_add_(0, dst, m)

        # 更新
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))  # [N, H]
        h_new = self.norm(h_new + h)  # 残差 + LN
        return h_new


class MutPredV2Model(nn.Module):
    """
    MutPred_v2 主模型:
        - 先将 x_struct, x_esm 映射到同一 hidden_dim 空间并相加;
        - 堆叠若干 MPNN 层进行消息传递;
        - 最后通过线性层输出 20 类氨基酸 logits。
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
    ):
        super().__init__()
        self.struct_dim = struct_dim
        self.esm_dim = esm_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lin_struct = nn.Linear(struct_dim, hidden_dim)
        self.lin_esm = nn.Linear(esm_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x_struct: torch.Tensor,     # [N, F_struct]
        x_esm: torch.Tensor,        # [N, D_esm]
        edge_index: torch.Tensor,   # [2, E]
        edge_attr: torch.Tensor,    # [E, F_edge]
    ) -> torch.Tensor:
        # 融合结构特征和 ESM 特征
        h = self.lin_struct(x_struct) + self.lin_esm(x_esm)  # [N, H]
        h = F.relu(h)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            h = self.dropout(h)

        h = self.final_norm(h)
        logits = self.head(h)  # [N, num_classes]
        return logits
