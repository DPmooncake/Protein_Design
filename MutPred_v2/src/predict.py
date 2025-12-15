# -*- coding: utf-8 -*-
# predict.py
#
# 作用:
#     使用训练好的 MutPred_v2 模型对单个 holo PDB 结构进行打分,
#     输出每个 pocket 残基在当前结构/序列环境下最优替代氨基酸以及
#     自监督分数 score = logP(mut) - logP(wt), 可作为突变筛选的参考。
#
# 支持两种图构建模式:
#     1) graph_mode = "pocket": 仅对口袋周围残基建图 (原始口袋图行为);
#     2) graph_mode = "global": 对全蛋白所有残基建图, 但仍用 pocket_mask
#        标记 core/shell, 预测时只对 pocket 残基打分, 其它节点作为上下文。
#
# 输入:
#     --pdb:        单个 PDB 文件路径 (包含蛋白 + 底物 + TPP + Mg 的 holo 结构);
#     --checkpoint: 训练好的 MutPred_v2 checkpoint 路径 (mutpred_v2_best.pt);
#     --out-dir:    输出 CSV 文件夹路径, 程序自动保存为:
#                   {pdbname}_mut_suggest.csv, 其中 pdbname 为 PDB 文件名(无扩展名);
#     --graph-mode: "pocket" 或 "global", 必须与训练该 checkpoint 使用的图模式一致。
#
# 输出:
#     在 out-dir 下生成一个 CSV 文件, 每行对应一个 pocket 蛋白残基, 包含:
#         uid, node_idx, chain_idx, resseq, native_aa,
#         logP_native, best_alt_aa, logP_best_alt,
#         score(=logP_best_alt - logP_native), entropy,
#         top1_aa, top1_p, ..., topK_aa, topK_p
"""
# 调用示例:
python src/predict.py \
    --pdb data/val/2JCL.pdb \
    --checkpoint results/mutpred_v2_ckpt/global_graph_global_mask_100/mutpred_v2_best.pt \
    --out-dir results/mutpred_v2_pred \
    --graph-mode global \
    --top-sites 30 \
    --top-k-aa 5
"""


import os
import argparse
from typing import List, Tuple

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
)  # :contentReference[oaicite:0]{index=0}

AA_TYPES_1 = [
    "A", "R", "N", "D", "C",
    "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
]
AA_DIM = 20
NODE_TYPE_PROTEIN = 0  # 与构图脚本中的 NODE_TYPE_MAP["protein"] 一致


def load_mutpred_model(ckpt_path: str, device: torch.device) -> MutPredV2Model:
    """
    根据 checkpoint 自动推断结构特征维度/ESM 维度/边特征维度, 构建 MutPredV2Model 并加载参数。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"]

    hidden_dim = state["lin_struct.weight"].shape[0]
    struct_dim = state["lin_struct.weight"].shape[1]
    esm_dim = state["lin_esm.weight"].shape[1]
    first_msg_weight = state["layers.0.msg_mlp.0.weight"]
    edge_dim = first_msg_weight.shape[1] - hidden_dim

    num_layers = 0
    for k in state.keys():
        if k.startswith("layers.") and k.endswith(".msg_mlp.0.weight"):
            num_layers += 1

    model = MutPredV2Model(
        struct_dim=struct_dim,
        esm_dim=esm_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=AA_DIM,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def esm_embed_pdb_residues(
    residues: List[dict],
    model_name: str,
    repr_layer: int,
    device: torch.device,
) -> Tuple[str, np.ndarray]:
    """
    给定 PDB 残基列表, 构建拼接序列, 使用 ESM-2 计算残基级别 embedding。
    """
    seq, _ = build_pdb_seq_and_keys(residues)
    if len(seq) == 0:
        raise RuntimeError("从 PDB 中未提取到任何蛋白残基序列")

    print(f"ESM: 使用模型 {model_name}, 序列长度 L={len(seq)}")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()

    data = [("pdb_seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer][0]  # [L_token, D]

    L = len(seq)
    residue_emb = reps[1: 1 + L].cpu().numpy()  # 去掉 BOS
    return seq, residue_emb


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

    # 1) ESM embedding
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

    # 3) 识别口袋残基 (用于 pocket_mask, 即使 graph_mode = global 也需要)
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

    # 4) 构图 (关键: 传入 graph_mode)
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


def score_pocket_sites(
    uid: str,
    graph: dict,
    model: MutPredV2Model,
    device: torch.device,
    top_sites: int = 20,
    top_k_aa: int = 5,
) -> List[List[str]]:
    """
    对一个图的 pocket 蛋白残基打分, 返回排序后的若干行(每行是一个字段列表)。

    打分逻辑(每个 pocket 蛋白节点 i):
        1) mask 该节点残基 (one-hot 和 x_esm 清零);
        2) forward 得到该位点对 20 AA 的 log_probs;
        3) 记 WT 氨基酸为 w, 最佳替代氨基酸为 m (排除 w 后概率最大的氨基酸);
        4) score = logP(m) - logP(w);
        5) 同时记录熵与前 top_k 个氨基酸的概率。
    """
    x = graph["x"].to(device)
    x_esm = graph["x_esm"].to(device)
    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    node_type = graph["node_type"].to(device)
    pocket_mask = graph["pocket_mask"].to(device)
    node_chain_idx = graph["node_chain_idx"]
    node_resseq = graph["node_resseq"]

    is_protein = (node_type == NODE_TYPE_PROTEIN)
    pocket_protein_mask = (is_protein & pocket_mask)
    pocket_indices = pocket_protein_mask.nonzero(as_tuple=False).view(-1).tolist()

    results = []

    for idx in pocket_indices:
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
            logits = model(x_masked, x_esm_masked, edge_index, edge_attr)  # [N, 20]
            log_probs = F.log_softmax(logits[i], dim=-1)  # [20]
            probs = log_probs.exp()

        logP_native = float(log_probs[aa_idx].item())

        # 最佳替代氨基酸 (排除原氨基酸)
        probs_np = probs.cpu().numpy()
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

    # 按 score 从大到小排序, 并截断到 top_sites
    results.sort(key=lambda d: d["score"], reverse=True)
    results = results[:top_sites]

    # 转成写 CSV 用的行
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
    parser = argparse.ArgumentParser(
        description="End-to-end mutation site suggestion for a single PDB"
    )
    parser.add_argument("--pdb", type=str, required=True, help="输入的 holo PDB 文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="MutPred_v2 训练好的 checkpoint 路径")
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
    parser.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    pdb_path = args.pdb
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB 文件不存在: {pdb_path}")
    uid = os.path.splitext(os.path.basename(pdb_path))[0]

    print(f"加载 MutPred_v2 模型: {args.checkpoint}")
    model = load_mutpred_model(args.checkpoint, device)

    print(f"从 PDB 构建图 ({args.graph_mode} 模式): {pdb_path}")
    graph = build_graph_from_pdb(
        pdb_path=pdb_path,
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

    print("对 pocket 残基打分并排序...")
    rows = score_pocket_sites(
        uid=uid,
        graph=graph,
        model=model,
        device=device,
        top_sites=args.top_sites,
        top_k_aa=args.top_k_aa,
    )

    # 写 CSV
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
