# -*- coding: utf-8 -*-
# predict.py
#
# 作用:
#   使用训练好的 MutPred_v2(EGNN) 模型对单个 holo PDB 结构进行逐“位点(resseq)”打分并给出突变建议。
#   与 train_wet_head.py 对齐的核心点：
#     1) 在线构图参数与 build_graphs_global.py 离线构图命令保持一致；
#     2) score 定义对齐湿实验监督：
#           score(aa) = mean_chains( log10 P(aa | masked_context) - log10 P(wt | masked_context) )
#        - masked_context 的 mask 是对该 resseq 在所有亚基上的对应节点同时 mask（模拟多聚体“同位点全突变”）
#     3) 输出按 resseq 聚合（不再输出 0/1/2/3 亚基的重复位点）
#     4) 不输出 n_chains/topK 等列（按用户要求精简）
#
# 输入:
#   --pdb:        单个 holo PDB 文件路径(蛋白 + 底物 + TPP + Mg)
#   --checkpoint: 训练好的 MutPred_v2 checkpoint 路径（必须含 key="model_state"）
#   --out-dir:    输出 CSV 的文件夹
#   在线构图参数（必须与 build_graphs_global.py 构图命令保持一致）:
#     --sub-resname l02
#     --tpp-resname l01
#     --core-cutoff 4.5
#     --contact-cutoff 5.0
#     --shell-hops 1
#     --graph-mode global
#     --protein-cutoff 16
#     --prot-lig-cutoff 5.0
#
# 输出:
#   {out-dir}/{pdbname}_mut_suggest.csv
#
# 调用示例:
"""
 python src/predict.py \
   --pdb data/pdb/ARO10_AF3.pdb \
   --checkpoint results/wet_head_ckpt/egnn_100_noz/mutpred_v2_wet_best.pt \
   --out-dir results/mutpred_v2_pred/wet_head_test \
   --graph-mode global \
   --score-scope global \
   --top-sites 30 \
   --top-k-aa 5 \
   --sub-resname l02 \
   --tpp-resname l01 \
   --core-cutoff 4.5 \
   --contact-cutoff 5.0 \
   --shell-hops 1 \
   --protein-cutoff 16 \
   --prot-lig-cutoff 5.0
"""

import os
import math
import argparse
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import esm
from Bio.PDB import PDBParser

from models.mutpred_v2_model import MutPredV2Model
from embedding.build_graphs_global import (
    collect_protein_residues,
    identify_pocket_residues,
    build_pdb_seq_and_keys,
    build_pdb_to_esm_mapping,
    build_graph_for_uid,
)

# =========================
# 常量：唯一 AA 顺序来源
# =========================
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TYPES_1 = list(AA_ORDER)
AA_DIM = 20
NODE_TYPE_PROTEIN = 0


def _infer_model_hparams_from_state(state: dict) -> Tuple[int, int, int, int, int]:
    """
    从 state_dict 推断:
        struct_dim, esm_dim, edge_dim, hidden_dim, num_layers
    约定:
        使用 MutPredV2Model 当前版本命名:
          - struct_proj / esm_proj
          - layers.{i}.edge_mlp
    """
    if "struct_proj.weight" not in state or "esm_proj.weight" not in state:
        raise KeyError(
            "checkpoint 的 model_state 不符合当前 MutPredV2Model 命名体系："
            "缺少 struct_proj.weight 或 esm_proj.weight。"
        )

    hidden_dim = int(state["struct_proj.weight"].shape[0])
    struct_dim = int(state["struct_proj.weight"].shape[1])
    esm_dim = int(state["esm_proj.weight"].shape[1])

    key0 = "layers.0.edge_mlp.0.weight"
    if key0 not in state:
        cand = [k for k in state.keys() if k.startswith("layers.") and k.endswith(".edge_mlp.0.weight")]
        if not cand:
            raise KeyError("无法在 state_dict 中找到 layers.*.edge_mlp.0.weight，无法推断 edge_dim。")
        key0 = sorted(cand)[0]

    in_dim = int(state[key0].shape[1])
    edge_dim = in_dim - (2 * hidden_dim + 1)
    if edge_dim < 0:
        raise ValueError(f"推断 edge_dim 失败：in_dim={in_dim}, hidden_dim={hidden_dim} -> edge_dim={edge_dim}")

    layer_ids = set()
    for k in state.keys():
        if k.startswith("layers.") and k.endswith(".edge_mlp.0.weight"):
            try:
                lid = int(k.split(".")[1])
                layer_ids.add(lid)
            except Exception:
                pass
    if not layer_ids:
        raise ValueError("推断 num_layers 失败：未找到 layers.{i}.edge_mlp.0.weight")
    num_layers = max(layer_ids) + 1

    return struct_dim, esm_dim, edge_dim, hidden_dim, num_layers


def load_mutpred_model(ckpt_path: str, device: torch.device) -> MutPredV2Model:
    """
    统一规范：
      - ckpt 必须包含 key="model_state"
      - model_state 必须与当前 MutPredV2Model 命名体系匹配（struct_proj/esm_proj/edge_mlp）
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise KeyError(f"checkpoint 缺少 'model_state'：{ckpt_path}")

    state = ckpt["model_state"]
    if not isinstance(state, dict):
        raise TypeError(f"ckpt['model_state'] 不是 state_dict(dict)，实际类型={type(state)}")

    struct_dim, esm_dim, edge_dim, hidden_dim, num_layers = _infer_model_hparams_from_state(state)

    model = MutPredV2Model(
        struct_dim=struct_dim,
        esm_dim=esm_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=AA_DIM,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
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


def build_graph_from_pdb(
    pdb_path: str,
    uid: str,
    esm_model_name: str,
    repr_layer: int,
    sub_resname: str,
    tpp_resname: str,
    core_cutoff: float,
    contact_cutoff: float,
    shell_hops: int,
    protein_cutoff: float,
    prot_lig_cutoff: float,
    graph_mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    直接从单个 PDB 构建图(内存中)，支持 pocket/global 两种模式。
    已彻底移除 RBF 参数；graph 必须包含 pos 供 EGNN 前向使用。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(uid, pdb_path)
    residues = collect_protein_residues(structure)
    if not residues:
        raise RuntimeError(f"{uid}: 未在 PDB 中找到蛋白残基")

    pdb_seq, pdb_keys = build_pdb_seq_and_keys(residues)
    if len(pdb_seq) == 0:
        raise RuntimeError(f"{uid}: 从 PDB 中未提取到任何蛋白残基序列")

    esm_seq = pdb_seq
    esm_residue_emb = esm_embed_sequence(
        seq=esm_seq,
        model_name=esm_model_name,
        repr_layer=repr_layer,
        device=device,
    )

    esm_mapping = build_pdb_to_esm_mapping(pdb_seq, pdb_keys, esm_seq)
    if not esm_mapping:
        raise RuntimeError(f"{uid}: PDB 与 ESM 序列对齐失败")

    pocket_simple = identify_pocket_residues(
        structure=structure,
        residues=residues,
        sub_resname=sub_resname,
        core_cutoff=core_cutoff,
        contact_cutoff=contact_cutoff,
        shell_hops=shell_hops,
    )
    if not pocket_simple:
        raise RuntimeError(f"{uid}: 未识别到 pocket 残基")

    graph = build_graph_for_uid(
        uid=uid,
        structure=structure,
        residues=residues,
        pocket_simple=pocket_simple,
        sub_resname=sub_resname,
        tpp_resname=tpp_resname,
        protein_cutoff=protein_cutoff,
        prot_lig_cutoff=prot_lig_cutoff,
        esm_residue_emb=esm_residue_emb,
        esm_mapping=esm_mapping,
        graph_mode=graph_mode,
    )
    if graph is None:
        raise RuntimeError(f"{uid}: 构图失败")
    if "pos" not in graph:
        raise RuntimeError(f"{uid}: graph 中缺少 pos，无法对齐 EGNN 前向")
    return graph


def _get_native_aa_idx(x: torch.Tensor, node_idx: torch.Tensor) -> int:
    """
    对同一 resseq 的多个链节点，native AA 理论应一致；
    这里取多数表决（更鲁棒）。
    """
    aa_idx = x[node_idx, :AA_DIM].argmax(dim=-1).detach().cpu().numpy().tolist()
    if len(aa_idx) == 0:
        return -1
    from collections import Counter
    c = Counter(aa_idx)
    return int(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def score_sites_site_level(
    uid: str,
    graph: Dict[str, Any],
    model: MutPredV2Model,
    device: torch.device,
    score_scope: str = "global",
    top_sites: int = 20,
    top_k_aa: int = 5,
) -> List[List[str]]:
    """
    按“位点(resseq)”聚合打分，score 定义与 train_wet_head.py 对齐：

      对某一位点 s（在所有亚基链上对应节点集合 node_idx）：
        1) 同时 mask 该位点的所有 node_idx（AA onehot 与 x_esm 清零）
        2) forward 得到 logits
        3) 对每个候选 aa:
             score_s(aa) = mean_{node in node_idx} ( log10 P(aa) - log10 P(wt) )
        4) best_alt = argmax_{aa != wt} score_s(aa)
        5) 以 best_alt 的 score 作为该位点的排序分数

    输出行(精简版，不含 n_chains/topk 列)：
      uid, resseq, native_aa, best_alt_aa, score, log10P_native, log10P_best_alt, entropy
    """
    x = graph["x"].to(device)
    x_esm = graph["x_esm"].to(device)
    pos = graph["pos"].to(device)
    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    node_type = graph["node_type"].to(device)
    node_resseq = graph["node_resseq"].to(device)

    is_protein = (node_type == NODE_TYPE_PROTEIN)

    if score_scope == "pocket":
        if "pocket_mask" not in graph:
            raise RuntimeError(f"{uid}: score_scope=pocket 但 graph 缺少 pocket_mask")
        pocket_mask = graph["pocket_mask"].to(device)
        cand_node_mask = is_protein & pocket_mask
    else:
        cand_node_mask = is_protein

    cand_resseq = torch.unique(node_resseq[cand_node_mask]).detach().cpu().tolist()
    cand_resseq = [int(r) for r in cand_resseq]
    cand_resseq.sort()

    ln10 = math.log(10.0)

    site_records: List[Dict[str, Any]] = []
    for resseq in cand_resseq:
        node_idx = torch.nonzero(cand_node_mask & (node_resseq == int(resseq)), as_tuple=False).view(-1)
        if node_idx.numel() == 0:
            continue

        native_idx = _get_native_aa_idx(x, node_idx)
        if not (0 <= native_idx < AA_DIM):
            continue
        native_aa = AA_TYPES_1[native_idx]

        x_masked = x.clone()
        x_esm_masked = x_esm.clone()
        x_masked[node_idx, :AA_DIM] = 0.0
        x_esm_masked[node_idx] = 0.0

        with torch.no_grad():
            logits = model(x_masked, x_esm_masked, edge_index, edge_attr, pos)  # [N, 20]
            logp_ln = F.log_softmax(logits, dim=-1)  # [N, 20]
            probs = logp_ln.exp()

        lp_nodes = logp_ln[node_idx]         # [K,20]
        probs_nodes = probs[node_idx]        # [K,20]
        lp_mean = lp_nodes.mean(dim=0)       # [20]
        probs_mean = probs_nodes.mean(dim=0) # [20]

        lp10_mean = lp_mean / ln10  # [20] log10P

        log10P_native = float(lp10_mean[native_idx].item())
        score_vec = lp10_mean - lp10_mean[native_idx]  # [20]
        score_vec[native_idx] = -1e9  # 排除 WT

        best_alt_idx = int(score_vec.argmax().item())
        best_alt_aa = AA_TYPES_1[best_alt_idx]
        score_best = float(score_vec[best_alt_idx].item())
        log10P_best = float(lp10_mean[best_alt_idx].item())

        entropy = float(-(probs_mean * (probs_mean.clamp_min(1e-12)).log()).sum().item())

        # 保留 top-k 计算能力（不输出到 CSV），便于未来内部分析
        if top_k_aa is not None and int(top_k_aa) > 0:
            k = min(int(top_k_aa), AA_DIM)
            _ = probs_mean.topk(k, dim=-1)  # 仅触发计算，结果不写出

        site_records.append(
            {
                "uid": uid,
                "resseq": int(resseq),
                "native_aa": native_aa,
                "log10P_native": log10P_native,
                "best_alt_aa": best_alt_aa,
                "log10P_best_alt": log10P_best,
                "score": score_best,
                "entropy": entropy,
            }
        )

    site_records.sort(key=lambda d: d["score"], reverse=True)
    site_records = site_records[: int(top_sites)]

    rows: List[List[str]] = []
    for r in site_records:
        row = [
            r["uid"],
            str(r["resseq"]),
            r["native_aa"],
            r["best_alt_aa"],
            f"{r['score']:.6f}",
            f"{r['log10P_native']:.6f}",
            f"{r['log10P_best_alt']:.6f}",
            f"{r['entropy']:.6f}",
        ]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Mutation site suggestion for a single holo PDB (site-level scoring)")

    parser.add_argument("--pdb", type=str, required=True, help="输入的 holo PDB 文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="MutPred_v2 checkpoint 路径（必须含 model_state）")
    parser.add_argument("--out-dir", type=str, required=True, help="候选突变位点输出 CSV 文件夹路径")

    parser.add_argument("--model-name", type=str, default="esm2_t36_3B_UR50D", help="ESM-2 模型名称")
    parser.add_argument("--repr-layer", type=int, default=36, help="ESM 使用的层编号")

    # 在线构图参数：与 build_graphs_global.py 构图命令保持一致
    parser.add_argument("--sub-resname", type=str, default="l02", help="底物 resName")
    parser.add_argument("--tpp-resname", type=str, default="l01", help="TPP resName")
    parser.add_argument("--core-cutoff", type=float, default=4.5, help="pocket core cutoff (Å)")
    parser.add_argument("--contact-cutoff", type=float, default=5.0, help="pocket contact cutoff (Å)")
    parser.add_argument("--shell-hops", type=int, default=1, help="pocket shell hops")

    parser.add_argument("--protein-cutoff", type=float, default=16.0, help="蛋白-蛋白构边参数（需与构图一致）")
    parser.add_argument("--prot-lig-cutoff", type=float, default=5.0, help="蛋白-配体/TPP/Mg 构边参数（需与构图一致）")

    parser.add_argument("--top-sites", type=int, default=20, help="输出候选突变位点数量(按 score 排序截断)")
    parser.add_argument("--top-k-aa", type=int, default=5, help="内部保留：用于触发 top-k 概率计算（不写 CSV）")

    parser.add_argument(
        "--graph-mode",
        type=str,
        default="global",
        choices=["pocket", "global"],
        help="构图模式: pocket 或 global（需与构图/训练保持一致）",
    )
    parser.add_argument(
        "--score-scope",
        type=str,
        default="global",
        choices=["pocket", "global"],
        help="打分范围: pocket(仅口袋) 或 global(全蛋白)",
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
    model = load_mutpred_model(args.checkpoint, device)

    print(f"从 PDB 构建图 ({args.graph_mode} 模式): {args.pdb}")
    graph = build_graph_from_pdb(
        pdb_path=args.pdb,
        uid=uid,
        esm_model_name=args.model_name,
        repr_layer=args.repr_layer,
        sub_resname=args.sub_resname,
        tpp_resname=args.tpp_resname,
        core_cutoff=args.core_cutoff,
        contact_cutoff=args.contact_cutoff,
        shell_hops=args.shell_hops,
        protein_cutoff=args.protein_cutoff,
        prot_lig_cutoff=args.prot_lig_cutoff,
        graph_mode=args.graph_mode,
        device=device,
    )

    print(f"对残基位点打分并排序 (score_scope={args.score_scope})...")
    rows = score_sites_site_level(
        uid=uid,
        graph=graph,
        model=model,
        device=device,
        score_scope=args.score_scope,
        top_sites=args.top_sites,
        top_k_aa=args.top_k_aa,
    )

    # 精简输出列：不含 n_chains/topk
    header = [
        "uid",
        "resseq",
        "native_aa",
        "best_alt_aa",
        "score",
        "log10P_native",
        "log10P_best_alt",
        "entropy",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{uid}_mut_suggest.csv")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    print(f"完成. 结果已写入: {out_csv}")


if __name__ == "__main__":
    main()
