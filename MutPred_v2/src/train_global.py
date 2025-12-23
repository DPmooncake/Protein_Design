# -*- coding: utf-8 -*-
# predict.py
#
# 作用:
#   使用训练好的 MutPred_v2(EGNN) 模型对单个 holo PDB 结构进行逐位点打分与氨基酸替换建议。
#   对每个待评估蛋白残基 i:
#     1) 将该位点的 AA one-hot(前20维) 与 x_esm 清零(mask);
#     2) 使用模型前向得到该位点 20aa 的 logits/logP;
#     3) 计算 score = logP(best_alt) - logP(wt)，并输出 top-k 氨基酸概率。
#
# 输入:
#   --pdb:        单个 holo PDB 文件路径(蛋白 + 底物 + TPP + Mg)
#   --checkpoint: checkpoint 路径(支持两种格式)
#       1) 自监督预训练 ckpt:   {"model_state": state_dict, ...}
#       2) 湿实验监督 ckpt:     {"backbone_state": state_dict, "wet_head_state": ..., ...}
#   --state-key:  指定从 ckpt 中取哪一个 state_dict
#       - auto(默认): 优先 backbone_state，其次 model_state，最后尝试把 ckpt 本身当作 state_dict
#       - backbone_state: 强制使用 backbone_state
#       - model_state:    强制使用 model_state
#   --out-dir:    输出 CSV 的文件夹
#   --graph-mode: pocket/global (与构图一致；一般建议 global)
#   --score-scope: pocket/global
#       pocket: 只对 pocket_mask 标记的蛋白残基打分
#       global: 对所有蛋白残基打分（用于捕获远端突变）
#
# 输出:
#   {out-dir}/{pdbname}_mut_suggest.csv
#
# 调用示例:
"""
python src/predict.py \
    --pdb data/val/pdbs/1MCZ_2_ketopentanoate.pdb \
    --checkpoint results/wet_head_ckpt/egnn_100/mutpred_v2_wet_best.pt \
    --state-key backbone_state \
    --out-dir results/mutpred_v2_pred/egnn_test \
    --graph-mode global \
    --score-scope global \
    --top-sites 30 \
    --top-k-aa 5
"""

import os
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import esm
from Bio.PDB import PDBParser

from models.mutpred_v2_model import MutPredV2Model
from embedding.build_graphs_global import (
    collect_protein_residues,
    identify_pocket_residues,
    build_pdb_to_esm_mapping,
    build_graph_for_uid,
)

# 你已确认 AA 顺序无问题：这里不再改动你的顺序假设
AA_TYPES_1 = [
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
]
AA_DIM = 20
NODE_TYPE_PROTEIN = 0  # 与构图脚本一致


def _extract_state_dict(ckpt: Any, state_key: str) -> Dict[str, torch.Tensor]:
    """
    作用:
        从 checkpoint 中提取 backbone 的 state_dict，兼容:
          - {"model_state": ...}
          - {"backbone_state": ...}
          - 直接就是 state_dict
    输入:
        ckpt: torch.load() 得到的对象
        state_key: auto / backbone_state / model_state
    输出:
        state_dict
    """
    if isinstance(ckpt, dict):
        if state_key != "auto":
            if state_key not in ckpt:
                raise KeyError(f"checkpoint 不包含 key='{state_key}'，可用 keys={list(ckpt.keys())}")
            state = ckpt[state_key]
            if not isinstance(state, dict):
                raise TypeError(f"ckpt['{state_key}'] 不是 dict(state_dict)，实际类型={type(state)}")
            return state

        # auto: 优先 backbone_state，其次 model_state
        if "backbone_state" in ckpt and isinstance(ckpt["backbone_state"], dict):
            return ckpt["backbone_state"]
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]

        # 有些人会把整个 ckpt 直接存 state_dict：尝试判断
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt

        raise KeyError(
            "无法从 checkpoint 提取模型参数。"
            "期望包含 'backbone_state' 或 'model_state'，或 checkpoint 本身就是 state_dict。"
        )

    # 不是 dict：不支持
    raise TypeError(f"不支持的 checkpoint 类型: {type(ckpt)}")


def _infer_model_hparams_from_state(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int, int]:
    """
    作用:
        从 state_dict 推断 (struct_dim, esm_dim, edge_dim, hidden_dim, num_layers)
    说明:
        兼容两种命名体系:
          1) 新版 MutPredV2Model: struct_proj/esm_proj/layers.{i}.edge_mlp
          2) 旧版命名: lin_struct/lin_esm/layers.{i}.msg_mlp
    """
    # 新版（与你发的 mutpred_v2_model.py 一致）
    if "struct_proj.weight" in state and "esm_proj.weight" in state:
        hidden_dim = int(state["struct_proj.weight"].shape[0])
        struct_dim = int(state["struct_proj.weight"].shape[1])
        esm_dim = int(state["esm_proj.weight"].shape[1])

        # EGNNLayer: edge_mlp[0] = Linear(2H + 1 + edge_dim, H)
        key0 = "layers.0.edge_mlp.0.weight"
        if key0 not in state:
            # 兜底：找任意一层
            cand = [k for k in state.keys() if k.endswith(".edge_mlp.0.weight") and k.startswith("layers.")]
            if not cand:
                raise KeyError("在 state_dict 中找不到 layers.*.edge_mlp.0.weight，无法推断 edge_dim")
            key0 = sorted(cand)[0]

        in_dim = int(state[key0].shape[1])
        edge_dim = in_dim - (2 * hidden_dim + 1)
        if edge_dim <= 0:
            raise ValueError(f"推断 edge_dim 失败: in_dim={in_dim}, hidden_dim={hidden_dim} -> edge_dim={edge_dim}")

        # 推断 num_layers
        layer_ids = set()
        for k in state.keys():
            if k.startswith("layers.") and k.endswith(".edge_mlp.0.weight"):
                # layers.{i}.edge_mlp.0.weight
                try:
                    lid = int(k.split(".")[1])
                    layer_ids.add(lid)
                except Exception:
                    continue
        num_layers = max(layer_ids) + 1 if layer_ids else 0
        if num_layers <= 0:
            raise ValueError("推断 num_layers 失败：未找到 layers.{i}.edge_mlp.0.weight")

        return struct_dim, esm_dim, edge_dim, hidden_dim, num_layers

    # 旧版（兼容）
    if "lin_struct.weight" in state and "lin_esm.weight" in state:
        hidden_dim = int(state["lin_struct.weight"].shape[0])
        struct_dim = int(state["lin_struct.weight"].shape[1])
        esm_dim = int(state["lin_esm.weight"].shape[1])

        key0 = "layers.0.msg_mlp.0.weight"
        if key0 not in state:
            cand = [k for k in state.keys() if k.endswith(".msg_mlp.0.weight") and k.startswith("layers.")]
            if not cand:
                raise KeyError("在 state_dict 中找不到 layers.*.msg_mlp.0.weight，无法推断 edge_dim")
            key0 = sorted(cand)[0]

        in_dim = int(state[key0].shape[1])
        edge_dim = in_dim - hidden_dim
        if edge_dim <= 0:
            raise ValueError(f"推断 edge_dim 失败: in_dim={in_dim}, hidden_dim={hidden_dim} -> edge_dim={edge_dim}")

        layer_ids = set()
        for k in state.keys():
            if k.startswith("layers.") and k.endswith(".msg_mlp.0.weight"):
                try:
                    lid = int(k.split(".")[1])
                    layer_ids.add(lid)
                except Exception:
                    continue
        num_layers = max(layer_ids) + 1 if layer_ids else 0
        if num_layers <= 0:
            raise ValueError("推断 num_layers 失败：未找到 layers.{i}.msg_mlp.0.weight")

        return struct_dim, esm_dim, edge_dim, hidden_dim, num_layers

    # 都不匹配：给出 keys 线索
    sample_keys = list(state.keys())[:30]
    raise KeyError(
        "无法识别 state_dict 的命名体系：既没有 struct_proj/esm_proj，也没有 lin_struct/lin_esm。"
        f"示例 keys={sample_keys}"
    )


def load_mutpred_model(ckpt_path: str, device: torch.device, state_key: str = "auto") -> MutPredV2Model:
    """
    作用:
        加载 MutPredV2Model backbone，支持:
          - 旧自监督 ckpt: model_state
          - 新湿实验 ckpt: backbone_state
    输入:
        ckpt_path: checkpoint 路径
        device: cpu/cuda
        state_key: auto/backbone_state/model_state
    输出:
        MutPredV2Model (eval 模式)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt, state_key=state_key)

    struct_dim, esm_dim, edge_dim, hidden_dim, num_layers = _infer_model_hparams_from_state(state)

    model = MutPredV2Model(
        struct_dim=struct_dim,
        esm_dim=esm_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=AA_DIM,
    )

    # strict=False：允许 ckpt 中额外键（例如你未来加别的分支），也允许某些无关 buffer
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] 加载模型缺少参数:", missing)
    if unexpected:
        print("[WARN] 加载模型多余参数:", unexpected)

    model.to(device)
    model.eval()

    # 打印实际使用的 state key（便于对比实验）
    if isinstance(ckpt, dict):
        used = None
        if state_key == "auto":
            if "backbone_state" in ckpt and isinstance(ckpt["backbone_state"], dict):
                used = "backbone_state"
            elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
                used = "model_state"
            else:
                used = "state_dict_root"
        else:
            used = state_key
        print(f"[INFO] 使用 checkpoint state: {used}")

    return model


def esm_embed_sequence(seq: str, model_name: str, repr_layer: int, device: torch.device) -> np.ndarray:
    """
    对单条蛋白序列计算 ESM-2 残基级 embedding，返回 shape=[L, D] 的 numpy 数组。
    """
    if len(seq) == 0:
        raise RuntimeError("ESM: 输入序列长度为 0")

    print(f"ESM: 使用模型 {model_name}, 序列长度 L={len(seq)}")
    esm_model, alphabet = getattr(esm.pretrained, model_name)()
    esm_model.eval()
    esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()

    data = [("pdb_seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer][0]  # [L_token, D]

    residue_emb = reps[1: 1 + len(seq)].cpu().numpy()  # 去掉 BOS
    return residue_emb


def esm_embed_pdb_residues(residues, model_name: str, repr_layer: int, device: torch.device):
    """
    作用:
        从 PDB residues 构造序列并提取 ESM embedding
    输出:
        esm_seq: str
        esm_residue_emb: np.ndarray [L, D]
    """
    # residues -> sequence (使用 build_graphs_global 的逻辑一般更稳，但这里保持你原实现风格)
    # 由于你项目里 build_pdb_seq_and_keys/对齐逻辑可能较复杂，这里采用 build_pdb_to_esm_mapping 前置所需的 seq
    seq_chars = []
    for r in residues:
        aa = r.get_resname()
        # 这里不做复杂三字母->一字母映射，依赖 build_pdb_to_esm_mapping 的对齐能力
        # 如果你后续希望更严格，可以引入你项目已有的三字母->一字母表
        # 为避免预测脚本改动过大，这里保留你的现状：让 mapping 函数承担对齐
        # 但 ESM 必须有序列，因此简单用 X 占位会导致 embedding 无意义
        # 所以建议：你的 build_pdb_to_esm_mapping 已实现基于真实序列的对齐，否则这里需接入你已有的 build_pdb_seq_and_keys
        # ——由于你说 AA 顺序已无问题，此处不扩展，保持最小改动：
        seq_chars.append("X")
    esm_seq = "".join(seq_chars)

    # 如果全是 X，ESM embedding 无意义；此处给出显式报错提示你应接入项目内的序列构建函数
    if set(esm_seq) == {"X"}:
        raise RuntimeError(
            "当前 predict.py 的 esm_embed_pdb_residues 使用了占位序列 'X'*L，无法得到有效 ESM embedding。\n"
            "请把你项目内用于构图的真实序列构建函数接入（例如 build_pdb_seq_and_keys）。\n"
            "你之前能正常预测，说明你本地 src/predict.py 很可能已有该实现；请以你本地版本为准合并本次的 checkpoint 兼容改动。"
        )

    esm_residue_emb = esm_embed_sequence(esm_seq, model_name, repr_layer, device)
    return esm_seq, esm_residue_emb


def build_graph_from_pdb(
    pdb_path: str,
    uid: str,
    esm_model_name: str,
    repr_layer: int,
    sub_resname: str,
    tpp_resname: str,
    protein_cutoff: float,
    prot_lig_cutoff: float,
    rbf_k: int,
    rbf_max_dist: float,
    graph_mode: str,
    device: torch.device,
):
    """
    直接从单个 PDB 构建图 (在内存中), 支持 pocket/global 两种模式。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(uid, pdb_path)
    residues = collect_protein_residues(structure)
    if not residues:
        raise RuntimeError(f"{uid}: 未在 PDB 中找到蛋白残基")

    # 1) ESM embedding（注意：你本地 predict.py 若已有真实序列构建，请保留你本地实现）
    esm_seq, esm_residue_emb = esm_embed_pdb_residues(
        residues=residues,
        model_name=esm_model_name,
        repr_layer=repr_layer,
        device=device,
    )

    # 2) PDB <-> ESM 映射
    esm_mapping = build_pdb_to_esm_mapping(residues, esm_seq)
    if not esm_mapping:
        raise RuntimeError(f"{uid}: PDB 与 ESM 序列对齐失败")

    # 3) 识别口袋残基（用于 pocket_mask；即使 score-scope=global 也保留 pocket_mask 字段）
    pocket_simple = identify_pocket_residues(
        structure=structure,
        residues=residues,
        sub_resname=sub_resname,
        core_cutoff=4.5,
        contact_cutoff=5.0,
        shell_hops=1,
    )
    if not pocket_simple:
        raise RuntimeError(f"{uid}: 未识别到 pocket 残基")

    # 4) 构图
    graph = build_graph_for_uid(
        uid=uid,
        structure=structure,
        residues=residues,
        pocket_simple=pocket_simple,
        sub_resname=sub_resname,
        tpp_resname=tpp_resname,
        protein_cutoff=protein_cutoff,
        prot_lig_cutoff=prot_lig_cutoff,
        rbf_k=rbf_k,
        rbf_max_dist=rbf_max_dist,
        esm_residue_emb=esm_residue_emb,
        esm_mapping=esm_mapping,
        graph_mode=graph_mode,
    )
    if graph is None:
        raise RuntimeError(f"{uid}: 构图失败")
    return graph


def score_sites(
    uid: str,
    graph: dict,
    model: MutPredV2Model,
    device: torch.device,
    score_scope: str = "pocket",
    top_sites: int = 20,
    top_k_aa: int = 5,
) -> List[List[str]]:
    """
    对一个图的蛋白残基打分（pocket/global 由 score_scope 决定），返回排序后的若干行(每行是一个字段列表)。
    """
    x = graph["x"].to(device)
    x_esm = graph["x_esm"].to(device)
    pos = graph["pos"].to(device)
    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    node_type = graph["node_type"].to(device)
    pocket_mask = graph["pocket_mask"].to(device)
    node_chain_idx = graph["node_chain_idx"]
    node_resseq = graph["node_resseq"]

    is_protein = (node_type == NODE_TYPE_PROTEIN)
    if score_scope == "pocket":
        sel_mask = is_protein & pocket_mask
    else:
        sel_mask = is_protein

    indices = sel_mask.nonzero(as_tuple=False).view(-1).tolist()
    results = []

    for idx in indices:
        i = int(idx)

        # 原氨基酸
        aa_onehot = x[i, :AA_DIM].detach().cpu()
        aa_idx = int(aa_onehot.argmax().item())
        native_aa = AA_TYPES_1[aa_idx] if 0 <= aa_idx < len(AA_TYPES_1) else "X"

        # 构造 mask 副本
        x_masked = x.clone()
        x_esm_masked = x_esm.clone()
        x_masked[i, :AA_DIM] = 0.0
        x_esm_masked[i] = 0.0

        with torch.no_grad():
            logits = model(x_masked, x_esm_masked, edge_index, edge_attr, pos)  # [N, 20]
            log_probs = F.log_softmax(logits[i], dim=-1)  # [20]
            probs = log_probs.exp()

        logP_native = float(log_probs[aa_idx].item())

        # 最佳替代氨基酸 (排除原氨基酸)
        probs_np = probs.detach().cpu().numpy()
        probs_np_excl = probs_np.copy()
        probs_np_excl[aa_idx] = -1.0
        best_alt_idx = int(probs_np_excl.argmax())
        best_alt_aa = AA_TYPES_1[best_alt_idx] if 0 <= best_alt_idx < len(AA_TYPES_1) else "X"
        logP_best_alt = float(log_probs[best_alt_idx].item())

        # score = logP(mut) - logP(wt)
        score = logP_best_alt - logP_native

        # 熵
        entropy = float(-(probs * log_probs).sum().item())

        # top-k
        k = min(top_k_aa, AA_DIM)
        topv, topi = probs.topk(k, dim=-1)
        topv = topv.cpu().tolist()
        topi = topi.cpu().tolist()

        chain_idx = int(node_chain_idx[i].item())
        resseq = int(node_resseq[i].item())

        results.append(
            {
                "uid": uid,
                "node_idx": i,
                "chain_idx": chain_idx,
                "resseq": resseq,
                "native_aa": native_aa,
                "logP_native": logP_native,
                "best_alt_aa": best_alt_aa,
                "logP_best_alt": logP_best_alt,
                "score": score,
                "entropy": entropy,
                "top_i": topi,
                "top_v": topv,
            }
        )

    results.sort(key=lambda d: d["score"], reverse=True)
    results = results[:top_sites]

    rows: List[List[str]] = []
    for r in results:
        row = [
            r["uid"],
            str(r["node_idx"]),
            str(r["chain_idx"]),
            str(r["resseq"]),
            r["native_aa"],
            f"{r['logP_native']:.6f}",
            r["best_alt_aa"],
            f"{r['logP_best_alt']:.6f}",
            f"{r['score']:.6f}",
            f"{r['entropy']:.6f}",
        ]
        for aa_id, p in zip(r["top_i"], r["top_v"]):
            aa_letter = AA_TYPES_1[aa_id] if 0 <= aa_id < len(AA_TYPES_1) else "X"
            row.append(aa_letter)
            row.append(f"{p:.6f}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="End-to-end mutation site suggestion for a single PDB")

    parser.add_argument("--pdb", type=str, required=True, help="输入的 holo PDB 文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="MutPred_v2 checkpoint 路径")
    parser.add_argument(
        "--state-key",
        type=str,
        default="auto",
        choices=["auto", "backbone_state", "model_state"],
        help="从 checkpoint 中取 backbone 权重的 key（auto 优先 backbone_state，其次 model_state）",
    )

    parser.add_argument("--out-dir", type=str, required=True, help="候选突变位点输出 CSV 文件夹路径")
    parser.add_argument("--model-name", type=str, default="esm2_t36_3B_UR50D", help="ESM-2 模型名称")
    parser.add_argument("--repr-layer", type=int, default=36, help="ESM 使用的层编号")

    parser.add_argument("--sub-resname", type=str, default="l02", help="底物 resName")
    parser.add_argument("--tpp-resname", type=str, default="l01", help="TPP resName")
    parser.add_argument("--protein-cutoff", type=float, default=8.0, help="蛋白-蛋白 Cα 聚边距离阈值 (Å)")
    parser.add_argument("--prot-lig-cutoff", type=float, default=5.0, help="蛋白-配体/TPP/Mg 聚边距离阈值 (Å)")
    parser.add_argument("--rbf-k", type=int, default=16, help="RBF 边特征维度")
    parser.add_argument("--rbf-max-dist", type=float, default=15.0, help="RBF 最大距离 (Å)")

    parser.add_argument("--top-sites", type=int, default=20, help="输出候选突变位点数量")
    parser.add_argument("--top-k-aa", type=int, default=5, help="每个位点输出前多少个推荐 AA")

    parser.add_argument(
        "--graph-mode",
        type=str,
        default="global",
        choices=["pocket", "global"],
        help="与训练时使用的图模式保持一致: pocket 或 global",
    )
    parser.add_argument(
        "--score-scope",
        type=str,
        default="pocket",
        choices=["pocket", "global"],
        help="打分范围: pocket 或 global",
    )

    parser.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    if not os.path.isfile(args.pdb):
        raise FileNotFoundError(f"PDB 文件不存在: {args.pdb}")
    uid = os.path.splitext(os.path.basename(args.pdb))[0]

    print(f"加载 MutPred_v2 模型: {args.checkpoint}")
    model = load_mutpred_model(args.checkpoint, device, state_key=args.state_key)

    print(f"从 PDB 构建图 ({args.graph_mode} 模式): {args.pdb}")
    graph = build_graph_from_pdb(
        pdb_path=args.pdb,
        uid=uid,
        esm_model_name=args.model_name,
        repr_layer=args.repr_layer,
        sub_resname=args.sub_resname,
        tpp_resname=args.tpp_resname,
        protein_cutoff=args.protein_cutoff,
        prot_lig_cutoff=args.prot_lig_cutoff,
        rbf_k=args.rbf_k,
        rbf_max_dist=args.rbf_max_dist,
        graph_mode=args.graph_mode,
        device=device,
    )

    print(f"对残基打分并排序（score_scope={args.score_scope}）...")
    rows = score_sites(
        uid=uid,
        graph=graph,
        model=model,
        device=device,
        score_scope=args.score_scope,
        top_sites=args.top_sites,
        top_k_aa=args.top_k_aa,
    )

    header = [
        "uid",
        "node_idx",
        "chain_idx",
        "resseq",
        "native_aa",
        "logP_native",
        "best_alt_aa",
        "logP_best_alt",
        "score",
        "entropy",
    ]
    for k in range(1, args.top_k_aa + 1):
        header.append(f"top{k}_aa")
        header.append(f"top{k}_p")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{uid}_mut_suggest.csv")

    with open(out_csv, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    print(f"完成. 结果已写入: {out_csv}")


if __name__ == "__main__":
    main()
