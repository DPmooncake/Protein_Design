#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pockets_and_graphs.py

作用:
    1) 从 WT holo PDB (蛋白 + 底物 + TPP + Mg, 可为寡聚体) 中识别催化口袋残基;
    2) 基于 "口袋蛋白残基 + 底物 super-node + TPP super-node + Mg" 构建异质图;
    3) 将 ESM-2 残基级别 embedding 对齐到 PDB 残基, 作为额外节点特征写入图文件。

输入:
    - --pdb_dir:      存放 WT holo PDB 的目录, 每个文件名形如 {uid}.pdb;
    - --esm_dir:      存放 ESM-2 提取的序列特征目录, 内含 {uid}.pt, 由前一步
                      extract_esm2_embedding.py 生成, 结构为:
                          {
                              "seq_id": uid,
                              "seq": "ACDE...",
                              "residue_emb": FloatTensor [L, D],  # 残基级别 embedding
                              "seq_emb": FloatTensor [D],         # 序列级别 embedding
                          }
    - --out_graph_dir: 输出图 PT 文件目录, 每个图保存为 {uid}.pt;
    - --sub_resname:  底物在 PDB 中的 resName (如 l02);
    - --tpp_resname:  TPP 在 PDB 中的 resName (如 l01);
    - 若干口袋/图构建的超参数, 见 argparse 部分。

输出:
    对每个成功处理的 uid, 在 out_graph_dir 下生成 {uid}.pt, 其内容为字典:
        {
            "uid": uid: str,
            "pos": FloatTensor [N, 3],              # 节点坐标
            "x": FloatTensor [N, F_struct],         # 结构+类型节点特征
            "x_esm": FloatTensor [N, D_esm],        # ESM 残基 embedding (非蛋白节点为 0)
            "node_type": LongTensor [N],            # 节点类型: 0=蛋白,1=sub,2=tpp,3=mg
            "edge_index": LongTensor [2, E],        # 有向边
            "edge_attr": FloatTensor [E, F_edge],   # 边特征 (RBF+direction+edge_type+Δresseq+Δchain)
            "prot_node_idx": List[int],             # 蛋白节点在全图中的下标
            "sub_node_idx": List[int],
            "tpp_node_idx": List[int],
            "mg_node_idx": List[int],
            "node_chain_idx": LongTensor [N],       # 节点所属链的整数编号 (蛋白), 0=非蛋白或其它
            "node_resseq": LongTensor [N],          # 节点所属残基号 (蛋白), 0=非蛋白
            "pocket_mask": BoolTensor [N],          # True 表示蛋白 pocket 残基节点
            "pocket_layer": LongTensor [N],         # -1=非 pocket, 0=core, 1=shell
        }

调用格式示例:
    python src/embedding/build_graphs.py \
        --pdb_dir data/pdb/tpp_family \
        --esm_dir results/sequence_features \
        --out_graph_dir data/graphs \
        --sub_resname l02 \
        --tpp_resname l01 \
        --core_cutoff 4.5 \
        --contact_cutoff 5.0 \
        --shell_hops 1
"""

import os
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio import pairwise2

# ======================
# 一些基础常量和工具
# ======================

AA_TYPES_3 = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLU": 5, "GLN": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
AA_DIM = 20

# 三字母到一字母, 用于序列比对
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

NODE_TYPE_MAP = {
    "protein": 0,
    "sub": 1,
    "tpp": 2,
    "mg": 3,
}

EDGE_TYPE_MAP = {
    "prot-prot": 0,
    "prot-sub": 1,
    "prot-tpp": 2,
    "prot-mg": 3,
}

# 保留最大链数的上限用于整数链编号（用于 rel_chain）
MAX_CHAIN_DIM = 4


def one_hot(index: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    if 0 <= index < dim:
        v[index] = 1.0
    return v


def rbf_encode(dist: np.ndarray, K: int = 16, dmax: float = 15.0) -> np.ndarray:
    """
    标量距离 -> RBF 特征
    dist: (...,) numpy 数组
    返回: (..., K)
    """
    centers = np.linspace(0.0, dmax, K, dtype=np.float32)
    gamma = 1.0 / ((centers[1] - centers[0]) ** 2 + 1e-8)
    diff = dist[..., None] - centers[None, ...]
    return np.exp(-gamma * diff ** 2).astype(np.float32)


def atoms_center(atoms) -> Optional[np.ndarray]:
    if not atoms:
        return None
    coords = np.array([a.get_coord() for a in atoms], dtype=np.float32)
    return coords.mean(axis=0)


# ==========================
# 口袋识别相关函数
# ==========================

def collect_substrate_atoms(structure, sub_resname: str):
    """
    收集底物原子（resName == sub_resname）;
    默认将所有同名底物残基的原子合并。
    """
    model = structure[0]
    sub_atoms = []
    for chain in model:
        for residue in chain:
            hetflag, resseq, icode = residue.id
            if hetflag.strip() == "":
                continue
            resname = residue.get_resname().strip()
            if resname == sub_resname:
                for atom in residue:
                    sub_atoms.append(atom)
    return sub_atoms


def collect_protein_residues(structure):
    """
    收集所有蛋白残基（hetflag == ' '），并记录:
      - chain_id
      - resseq
      - icode
      - residue 对象
      - Cα 坐标
    返回列表, 每个元素为字典。
    """
    model = structure[0]
    residues = []
    for chain in model:
        cid = chain.id
        for residue in chain:
            hetflag, resseq, icode = residue.id
            if hetflag.strip() != "":
                continue
            if not residue.has_id("CA"):
                continue
            ca = residue["CA"].get_coord()
            residues.append({
                "chain_id": cid,
                "resseq": int(resseq),
                "icode": (icode or "").strip(),
                "residue": residue,
                "ca_coord": np.asarray(ca, dtype=np.float32),
            })
    return residues


def compute_min_dist_residue_to_sub(residue_entry, sub_coords: np.ndarray) -> float:
    """
    计算一个蛋白残基到底物的最小原子-原子距离。
    """
    res = residue_entry["residue"]
    res_coords = np.array([atom.get_coord() for atom in res], dtype=np.float32)
    diff = res_coords[:, None, :] - sub_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return float(dists.min())


def build_contact_graph(residues, contact_cutoff: float):
    """
    基于 Cα 构建残基接触图，返回邻接表 adjacency[i] = [j1, j2,...]
    """
    n = len(residues)
    coords = np.stack([r["ca_coord"] for r in residues], axis=0)
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= contact_cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def bfs_shell(core_indices, adjacency, hops: int):
    """
    从 core 出发，在接触图上扩展 hops 层邻居作为 shell。
    返回 shell 节点索引列表（不含 core 本身）。
    """
    core_set = set(core_indices)
    visited = set(core_indices)
    frontier = set(core_indices)

    for _ in range(hops):
        new_frontier = set()
        for u in frontier:
            for v in adjacency[u]:
                if v not in visited:
                    visited.add(v)
                    new_frontier.add(v)
        frontier = new_frontier

    shell = [i for i in visited if i not in core_set]
    return shell


def identify_pocket_residues(structure,
                             residues,
                             sub_resname: str,
                             core_cutoff: float,
                             contact_cutoff: float,
                             shell_hops: int):
    """
    以底物为锚点识别 pocket 残基, 返回列表:
      [
        { "chain_id": "A", "resseq": 135, "icode": "", "layer": "core" },
        ...
      ]
    """
    sub_atoms = collect_substrate_atoms(structure, sub_resname=sub_resname)
    if not sub_atoms:
        return []

    sub_coords = np.array([a.get_coord() for a in sub_atoms], dtype=np.float32)
    if not residues:
        return []

    min_dists = []
    for res_entry in residues:
        d = compute_min_dist_residue_to_sub(res_entry, sub_coords)
        min_dists.append(d)
    min_dists = np.array(min_dists, dtype=np.float32)

    core_indices = np.where(min_dists <= core_cutoff)[0].tolist()
    if not core_indices:
        return []

    adjacency = build_contact_graph(residues, contact_cutoff=contact_cutoff)
    shell_indices = bfs_shell(core_indices, adjacency, hops=shell_hops)

    pocket_list = []
    for idx in core_indices:
        r = residues[idx]
        pocket_list.append({
            "chain_id": r["chain_id"],
            "resseq": r["resseq"],
            "icode": r["icode"],
            "layer": "core",
        })
    for idx in shell_indices:
        r = residues[idx]
        pocket_list.append({
            "chain_id": r["chain_id"],
            "resseq": r["resseq"],
            "icode": r["icode"],
            "layer": "shell",
        })

    seen = set()
    unique_pocket = []
    for item in pocket_list:
        key = (item["chain_id"], item["resseq"], item["icode"])
        if key in seen:
            continue
        seen.add(key)
        unique_pocket.append(item)

    return unique_pocket


# ==========================
# Backbone torsion 相关
# ==========================

def _dihedral(p0: np.ndarray, p1: np.ndarray,
              p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算四个点定义的二面角, 返回弧度制。
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-6:
        return 0.0
    b1 /= b1_norm

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    angle = np.arctan2(y, x)
    return float(angle)


def compute_backbone_torsions(
    residues: List[Dict]
) -> Dict[Tuple[str, int, str], np.ndarray]:
    """
    基于 residues 列表, 计算每个残基的 backbone torsion:
      φ, ψ, ω 的 sin/cos, 一共 6 维。
    返回字典:
      key = (chain_id, resseq, icode)
      value = np.array([sin φ, cos φ, sin ψ, cos ψ, sin ω, cos ω], float32)
    端点 / 原子缺失时用 0 向量填充。
    """
    torsion_dict: Dict[Tuple[str, int, str], np.ndarray] = {}

    # 按链分组并排序
    chain_to_res = {}
    for entry in residues:
        cid = entry["chain_id"]
        chain_to_res.setdefault(cid, []).append(entry)

    for cid, res_list in chain_to_res.items():
        res_list_sorted = sorted(
            res_list, key=lambda e: (e["resseq"], e["icode"])
        )
        n = len(res_list_sorted)
        for idx, entry in enumerate(res_list_sorted):
            res = entry["residue"]
            key = (entry["chain_id"], entry["resseq"], entry["icode"])
            torsion = np.zeros(6, dtype=np.float32)

            try:
                if not (res.has_id("N") and res.has_id("CA") and res.has_id("C")):
                    torsion_dict[key] = torsion
                    continue

                N_i = res["N"].get_coord()
                CA_i = res["CA"].get_coord()
                C_i = res["C"].get_coord()

                prev_res = res_list_sorted[idx - 1]["residue"] if idx > 0 else None
                next_res = res_list_sorted[idx + 1]["residue"] if idx < n - 1 else None

                phi = 0.0
                psi = 0.0
                omega = 0.0

                if prev_res is not None and prev_res.has_id("C"):
                    C_prev = prev_res["C"].get_coord()
                    phi = _dihedral(C_prev, N_i, CA_i, C_i)
                    omega = _dihedral(C_prev, N_i, CA_i, C_i)

                if next_res is not None and next_res.has_id("N"):
                    N_next = next_res["N"].get_coord()
                    psi = _dihedral(N_i, CA_i, C_i, N_next)

                torsion = np.array(
                    [
                        np.sin(phi), np.cos(phi),
                        np.sin(psi), np.cos(psi),
                        np.sin(omega), np.cos(omega),
                    ],
                    dtype=np.float32,
                )
            except Exception:
                torsion = np.zeros(6, dtype=np.float32)

            torsion_dict[key] = torsion

    return torsion_dict


# ==========================
# ESM 对齐相关
# ==========================

def build_pdb_seq_and_keys(residues: List[Dict]) -> Tuple[str, List[Tuple[str, int, str]]]:
    """
    将 PDB 残基列表转换为一条序列字符串和对应的 key 列表。
    为了对齐, 按 (chain_id, resseq, icode) 排序。
    """
    res_list_sorted = sorted(
        residues, key=lambda e: (e["chain_id"], e["resseq"], e["icode"])
    )
    seq_chars = []
    keys = []
    for entry in res_list_sorted:
        res = entry["residue"]
        resname3 = res.get_resname().strip()
        aa1 = AA3_TO_1.get(resname3, "X")
        seq_chars.append(aa1)
        keys.append((entry["chain_id"], entry["resseq"], entry["icode"]))
    return "".join(seq_chars), keys


def build_pdb_to_esm_mapping(
    residues: List[Dict],
    esm_seq: str,
) -> Dict[Tuple[str, int, str], int]:
    """
    将 PDB 残基序列与 ESM 的序列做全局比对, 得到:
      key = (chain_id, resseq, icode) -> esm_index (0-based)
    若某个残基在比对中对齐到 gap, 则不会出现在 mapping 中。
    """
    pdb_seq, pdb_keys = build_pdb_seq_and_keys(residues)

    if len(pdb_seq) == 0 or len(esm_seq) == 0:
        return {}

    # 全局比对, 打分略偏向比对长度
    alignments = pairwise2.align.globalms(
        pdb_seq, esm_seq,
        2.0,  # match
        -1.0, # mismatch
        -2.0, # gap open
        -0.5, # gap extend
        one_alignment_only=True,
    )
    if not alignments:
        return {}

    aligned_pdb, aligned_esm, _, _, _ = alignments[0]

    mapping: Dict[Tuple[str, int, str], int] = {}
    i_pdb = 0
    i_esm = 0
    for c_pdb, c_esm in zip(aligned_pdb, aligned_esm):
        if c_pdb != "-" and c_esm != "-":
            key = pdb_keys[i_pdb]
            mapping[key] = i_esm
        if c_pdb != "-":
            i_pdb += 1
        if c_esm != "-":
            i_esm += 1

    return mapping


# ==========================
# 图构建相关函数
# ==========================

def residue_repr_coord(residue_entry) -> Optional[np.ndarray]:
    return residue_entry["ca_coord"]


def collect_ligand_atoms_for_graph(structure, sub_resname: str, tpp_resname: str):
    """
    收集底物、TPP 及 Mg 原子。
    """
    model = structure[0]
    sub_atoms = []
    tpp_residues = []
    mg_atoms = []

    for chain in model:
        for residue in chain:
            hetflag, resseq, icode = residue.id
            resname = residue.get_resname().strip()
            if resname.upper() == "MG":
                for atom in residue:
                    mg_atoms.append(atom)
                continue

            if hetflag.strip() == "":
                continue

            if resname == sub_resname:
                for atom in residue:
                    sub_atoms.append(atom)
            elif resname == tpp_resname:
                tpp_residues.append(residue)

    return sub_atoms, tpp_residues, mg_atoms


def find_nearest_tpp_and_mg(center_sub: np.ndarray,
                            tpp_residues,
                            mg_atoms):
    """
    在多个 TPP / Mg 中，选出与底物中心最近的一套。
    """
    tpp_atoms_selected = []
    mg_atom_selected = []

    min_dist_tpp = np.inf
    for res in tpp_residues:
        atoms = [a for a in res]
        c = atoms_center(atoms)
        if c is None:
            continue
        d = np.linalg.norm(c - center_sub)
        if d < min_dist_tpp:
            min_dist_tpp = d
            tpp_atoms_selected = atoms

    min_dist_mg = np.inf
    for atom in mg_atoms:
        c = np.asarray(atom.get_coord(), dtype=np.float32)
        d = np.linalg.norm(c - center_sub)
        if d < min_dist_mg:
            min_dist_mg = d
            mg_atom_selected = [atom]

    return tpp_atoms_selected, mg_atom_selected


def build_graph_for_uid(uid: str,
                        structure,
                        residues: List[Dict],
                        pocket_simple: List[Dict],
                        sub_resname: str,
                        tpp_resname: str,
                        protein_cutoff: float,
                        prot_lig_cutoff: float,
                        rbf_k: int,
                        rbf_max_dist: float,
                        esm_residue_emb: np.ndarray,
                        esm_mapping: Dict[Tuple[str, int, str], int]):
    """
    从结构和 pocket 残基列表构建联合图, 并接入 ESM 残基 embedding。
    """
    if not pocket_simple:
        print(f"[WARN] {uid}: empty pocket_simple, skip graph.")
        return None

    # 1) backbone torsion
    torsion_dict = compute_backbone_torsions(residues)

    key_to_res_entry = {
        (r["chain_id"], r["resseq"], r["icode"]): r for r in residues
    }

    # 2) 根据 pocket_simple 收集 pocket 残基, 加上 layer 信息
    pocket_residues = []
    for item in pocket_simple:
        key = (item["chain_id"], item["resseq"], item["icode"])
        if key not in key_to_res_entry:
            print(f"[WARN] pocket residue not found in residues: {key}")
            continue
        entry = key_to_res_entry[key]
        pocket_residues.append({
            "chain_id": item["chain_id"],
            "resseq": item["resseq"],
            "icode": item["icode"],
            "layer": item["layer"],
            "residue": entry["residue"],
            "ca_coord": entry["ca_coord"],
        })

    if not pocket_residues:
        print(f"[WARN] {uid}: no valid pocket residues found in residues list.")
        return None

    # 3) 底物 / TPP / Mg
    sub_atoms, tpp_residues, mg_atoms = collect_ligand_atoms_for_graph(
        structure, sub_resname=sub_resname, tpp_resname=tpp_resname
    )
    if not sub_atoms:
        print(f"[WARN] {uid}: no substrate {sub_resname} atoms, skip graph.")
        return None

    center_sub = atoms_center(sub_atoms)
    tpp_atoms_sel, mg_atom_sel = find_nearest_tpp_and_mg(center_sub, tpp_residues, mg_atoms)

    node_coords: List[np.ndarray] = []
    node_feats: List[np.ndarray] = []
    node_types: List[int] = []
    node_chain_idx: List[int] = []
    node_resseq: List[int] = []
    node_esm_feats: List[np.ndarray] = []

    prot_node_idx: List[int] = []
    sub_node_idx: List[int] = []
    tpp_node_idx: List[int] = []
    mg_node_idx: List[int] = []

    # pocket 掩码 / layer
    pocket_mask: List[bool] = []
    pocket_layer: List[int] = []  # -1=非 pocket, 0=core, 1=shell

    # 为蛋白链分配整数 chain index (1..MAX_CHAIN_DIM-1)
    chains_in_pocket = sorted({r["chain_id"] for r in pocket_residues})
    chain_id_to_idx = {}
    for i, cid in enumerate(chains_in_pocket[:MAX_CHAIN_DIM - 1], start=1):
        chain_id_to_idx[cid] = i

    esm_dim = int(esm_residue_emb.shape[1])

    def make_protein_node_feature(res_entry) -> Tuple[np.ndarray, int, int]:
        """
        返回:
          feat: 结构+类型特征
          chain_idx: 整数链编号
          resseq: 残基号
        """
        res = res_entry["residue"]
        resname3 = res.get_resname().strip()
        aa_idx = AA_TYPES_3.get(resname3, -1)
        aa_oh = one_hot(aa_idx, AA_DIM)

        node_type_oh = one_hot(NODE_TYPE_MAP["protein"], len(NODE_TYPE_MAP))

        key = (res_entry["chain_id"], res_entry["resseq"], res_entry["icode"])
        torsion = torsion_dict.get(key, np.zeros(6, dtype=np.float32))

        # 3 维距离特征占位, 后面统一填 (d_sub, d_tpp, d_mg)
        dist_feat = np.zeros(3, dtype=np.float32)

        feat = np.concatenate([aa_oh, node_type_oh, torsion, dist_feat], axis=0)

        cid = res_entry["chain_id"]
        cid_idx = chain_id_to_idx.get(cid, 0)
        resseq = int(res_entry["resseq"])

        return feat, cid_idx, resseq

    # 3.1 蛋白 pocket 节点
    for res_entry in pocket_residues:
        coord = residue_repr_coord(res_entry)
        if coord is None:
            print(f"[WARN] {uid}: missing CA coord for {res_entry['chain_id']} {res_entry['resseq']}")
            continue
        feat, cid_idx, resseq = make_protein_node_feature(res_entry)

        node_coords.append(coord)
        node_feats.append(feat)
        node_types.append(NODE_TYPE_MAP["protein"])
        node_chain_idx.append(cid_idx)
        node_resseq.append(resseq)
        prot_node_idx.append(len(node_coords) - 1)

        # pocket mask / layer
        pocket_mask.append(True)
        if res_entry["layer"] == "core":
            pocket_layer.append(0)
        else:
            pocket_layer.append(1)

        # ESM embedding
        key = (res_entry["chain_id"], res_entry["resseq"], res_entry["icode"])
        esm_idx = esm_mapping.get(key, None)
        if esm_idx is not None and 0 <= esm_idx < esm_residue_emb.shape[0]:
            esm_vec = esm_residue_emb[esm_idx]
        else:
            esm_vec = np.zeros(esm_dim, dtype=np.float32)
        node_esm_feats.append(esm_vec)

    # 3.2 底物 super-node
    sub_center = center_sub
    zero_aa = np.zeros(AA_DIM, dtype=np.float32)
    node_type_oh_sub = one_hot(NODE_TYPE_MAP["sub"], len(NODE_TYPE_MAP))
    zero_torsion = np.zeros(6, dtype=np.float32)
    dist_feat_zero = np.zeros(3, dtype=np.float32)
    sub_feat = np.concatenate([zero_aa, node_type_oh_sub, zero_torsion, dist_feat_zero], axis=0)

    node_coords.append(sub_center)
    node_feats.append(sub_feat)
    node_types.append(NODE_TYPE_MAP["sub"])
    node_chain_idx.append(0)
    node_resseq.append(0)
    sub_node_idx.append(len(node_coords) - 1)
    pocket_mask.append(False)
    pocket_layer.append(-1)
    node_esm_feats.append(np.zeros(esm_dim, dtype=np.float32))

    # 3.3 TPP super-node
    if tpp_atoms_sel:
        tpp_center = atoms_center(tpp_atoms_sel)
        node_type_oh_tpp = one_hot(NODE_TYPE_MAP["tpp"], len(NODE_TYPE_MAP))
        tpp_feat = np.concatenate([zero_aa, node_type_oh_tpp, zero_torsion, dist_feat_zero], axis=0)

        node_coords.append(tpp_center)
        node_feats.append(tpp_feat)
        node_types.append(NODE_TYPE_MAP["tpp"])
        node_chain_idx.append(0)
        node_resseq.append(0)
        tpp_node_idx.append(len(node_coords) - 1)
        pocket_mask.append(False)
        pocket_layer.append(-1)
        node_esm_feats.append(np.zeros(esm_dim, dtype=np.float32))

    # 3.4 Mg 节点
    if mg_atom_sel:
        mg_center = atoms_center(mg_atom_sel)
        node_type_oh_mg = one_hot(NODE_TYPE_MAP["mg"], len(NODE_TYPE_MAP))
        mg_feat = np.concatenate([zero_aa, node_type_oh_mg, zero_torsion, dist_feat_zero], axis=0)

        node_coords.append(mg_center)
        node_feats.append(mg_feat)
        node_types.append(NODE_TYPE_MAP["mg"])
        node_chain_idx.append(0)
        node_resseq.append(0)
        mg_node_idx.append(len(node_coords) - 1)
        pocket_mask.append(False)
        pocket_layer.append(-1)
        node_esm_feats.append(np.zeros(esm_dim, dtype=np.float32))

    node_coords = np.stack(node_coords, axis=0)
    node_feats = np.stack(node_feats, axis=0)
    node_esm_feats = np.stack(node_esm_feats, axis=0)
    N = node_coords.shape[0]

    # 4) 填距离特征 (d_sub, d_tpp, d_mg), 这 3 维始终放在结构特征最后
    feat_dim = node_feats.shape[1]
    dist_offset = feat_dim - 3

    sub_center = node_coords[sub_node_idx[0]]
    tpp_center = node_coords[tpp_node_idx[0]] if tpp_node_idx else None
    mg_center = node_coords[mg_node_idx[0]] if mg_node_idx else None

    for i in range(N):
        coord = node_coords[i]
        d_sub = np.linalg.norm(coord - sub_center)
        d_tpp = np.linalg.norm(coord - tpp_center) if tpp_center is not None else 0.0
        d_mg = np.linalg.norm(coord - mg_center) if mg_center is not None else 0.0
        node_feats[i, dist_offset + 0] = d_sub
        node_feats[i, dist_offset + 1] = d_tpp
        node_feats[i, dist_offset + 2] = d_mg

    # 5) 构边
    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_feats: List[np.ndarray] = []

    def add_edge(i: int, j: int, edge_type: str):
        """
        构建一条 i -> j 的有向边:
          - RBF(CA-CA)
          - direction
          - edge_type one-hot
          - rel_resseq (Δresseq)
          - rel_chain (Δchain)
        """
        vec = node_coords[j] - node_coords[i]
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            direction = np.zeros(3, dtype=np.float32)
        else:
            direction = (vec / dist).astype(np.float32)
        rbf = rbf_encode(np.array([dist], dtype=np.float32), K=rbf_k, dmax=rbf_max_dist)[0]
        edge_type_oh = np.zeros(len(EDGE_TYPE_MAP), dtype=np.float32)
        if edge_type in EDGE_TYPE_MAP:
            edge_type_oh[EDGE_TYPE_MAP[edge_type]] = 1.0

        if edge_type == "prot-prot":
            rel_resseq = node_resseq[j] - node_resseq[i]
            rel_resseq = float(max(-32, min(32, rel_resseq)))
            rel_chain = float(node_chain_idx[j] - node_chain_idx[i])
        else:
            rel_resseq = 0.0
            rel_chain = 0.0

        feat = np.concatenate(
            [rbf, direction, edge_type_oh, np.array([rel_resseq, rel_chain], dtype=np.float32)],
            axis=0,
        )
        edges_src.append(i)
        edges_dst.append(j)
        edge_feats.append(feat)

    # 5.1 蛋白-蛋白边
    for idx_i, i in enumerate(prot_node_idx):
        for j in prot_node_idx[idx_i + 1:]:
            d = np.linalg.norm(node_coords[i] - node_coords[j])
            if d <= protein_cutoff:
                add_edge(i, j, "prot-prot")
                add_edge(j, i, "prot-prot")

    # 5.2 蛋白-配体/TPP/Mg
    sub_idx = sub_node_idx[0]
    for i in prot_node_idx:
        d_sub = np.linalg.norm(node_coords[i] - node_coords[sub_idx])
        if d_sub <= prot_lig_cutoff:
            add_edge(i, sub_idx, "prot-sub")
            add_edge(sub_idx, i, "prot-sub")

        if tpp_node_idx:
            tpp_idx = tpp_node_idx[0]
            d_tpp = np.linalg.norm(node_coords[i] - node_coords[tpp_idx])
            if d_tpp <= prot_lig_cutoff:
                add_edge(i, tpp_idx, "prot-tpp")
                add_edge(tpp_idx, i, "prot-tpp")

        if mg_node_idx:
            mg_idx = mg_node_idx[0]
            d_mg = np.linalg.norm(node_coords[i] - node_coords[mg_idx])
            if d_mg <= prot_lig_cutoff:
                add_edge(i, mg_idx, "prot-mg")
                add_edge(mg_idx, i, "prot-mg")

    # 5.3 底物-TPP/Mg 功能性边
    if tpp_node_idx:
        tpp_idx = tpp_node_idx[0]
        add_edge(sub_idx, tpp_idx, "prot-sub")
        add_edge(tpp_idx, sub_idx, "prot-sub")

    if mg_node_idx:
        mg_idx = mg_node_idx[0]
        if tpp_node_idx:
            tpp_idx = tpp_node_idx[0]
            add_edge(tpp_idx, mg_idx, "prot-mg")
            add_edge(mg_idx, tpp_idx, "prot-mg")
        add_edge(sub_idx, mg_idx, "prot-mg")
        add_edge(mg_idx, sub_idx, "prot-mg")

    pos_t = torch.tensor(node_coords, dtype=torch.float32)
    x_t = torch.tensor(node_feats, dtype=torch.float32)
    x_esm_t = torch.tensor(node_esm_feats, dtype=torch.float32)
    node_type_t = torch.tensor(node_types, dtype=torch.long)
    edge_index_t = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr_t = torch.tensor(np.stack(edge_feats, axis=0), dtype=torch.float32)
    node_chain_idx_t = torch.tensor(node_chain_idx, dtype=torch.long)
    node_resseq_t = torch.tensor(node_resseq, dtype=torch.long)
    pocket_mask_t = torch.tensor(pocket_mask, dtype=torch.bool)
    pocket_layer_t = torch.tensor(pocket_layer, dtype=torch.long)

    graph = {
        "uid": uid,
        "pos": pos_t,
        "x": x_t,
        "x_esm": x_esm_t,
        "node_type": node_type_t,
        "edge_index": edge_index_t,
        "edge_attr": edge_attr_t,
        "prot_node_idx": prot_node_idx,
        "sub_node_idx": sub_node_idx,
        "tpp_node_idx": tpp_node_idx,
        "mg_node_idx": mg_node_idx,
        "node_chain_idx": node_chain_idx_t,
        "node_resseq": node_resseq_t,
        "pocket_mask": pocket_mask_t,
        "pocket_layer": pocket_layer_t,
    }
    return graph


# ==========================
# 主流程
# ==========================

def main():
    ap = argparse.ArgumentParser(
        description="从 WT holo PDB 生成口袋异质图 (含 ESM 节点特征)"
    )
    ap.add_argument("--pdb_dir", type=str, required=True,
                    help="WT holo PDB 目录, 文件名形如 {uid}.pdb")
    ap.add_argument("--esm_dir", type=str, required=True,
                    help="ESM 残基 embedding 目录, 内含 {uid}.pt")
    ap.add_argument("--out_graph_dir", type=str, required=True,
                    help="输出图 PT 目录")

    ap.add_argument("--sub_resname", type=str, default="l02",
                    help="底物 resName (例如 l02)")
    ap.add_argument("--tpp_resname", type=str, default="l01",
                    help="TPP resName (例如 l01)")

    # 口袋参数: 第一版建议
    ap.add_argument("--core_cutoff", type=float, default=4.5,
                    help="core 层定义的残基-底物最小距离阈值 (Å)")
    ap.add_argument("--contact_cutoff", type=float, default=5.0,
                    help="CA-CA 接触图距离阈值 (Å)")
    ap.add_argument("--shell_hops", type=int, default=1,
                    help="从 core 往外扩展的 shell 层数")

    # 图构建参数
    ap.add_argument("--protein_cutoff", type=float, default=8.0,
                    help="蛋白-蛋白 Cα 聚边距离阈值 (Å)")
    ap.add_argument("--prot_lig_cutoff", type=float, default=5.0,
                    help="蛋白-配体/TPP/Mg 聚边距离阈值 (Å)")
    ap.add_argument("--rbf_k", type=int, default=16,
                    help="RBF 维数")
    ap.add_argument("--rbf_max_dist", type=float, default=15.0,
                    help="RBF 最大距离 (Å)")
    args = ap.parse_args()

    os.makedirs(args.out_graph_dir, exist_ok=True)

    pdb_files = [f for f in os.listdir(args.pdb_dir)
                 if f.lower().endswith(".pdb")]
    pdb_files.sort()

    if not pdb_files:
        print(f"[WARN] No PDB files found in {args.pdb_dir}")
        return

    parser = PDBParser(QUIET=True)

    for fn in tqdm(pdb_files, desc="Processing PDBs"):
        uid = os.path.splitext(fn)[0]
        pdb_path = os.path.join(args.pdb_dir, fn)

        # 1) 读 PDB, 收集蛋白残基
        structure = parser.get_structure(uid, pdb_path)
        residues = collect_protein_residues(structure)
        if not residues:
            print(f"[WARN] {uid}: no protein residues found, skip.")
            continue

        # 2) 读 ESM 特征
        esm_path = os.path.join(args.esm_dir, f"{uid}.pt")
        if not os.path.isfile(esm_path):
            print(f"[WARN] {uid}: ESM file not found: {esm_path}, skip.")
            continue
        esm_data = torch.load(esm_path, map_location="cpu")
        esm_seq = esm_data["seq"]
        esm_residue_emb = esm_data["residue_emb"].numpy()  # [L, D]

        # 3) PDB 残基 <-> ESM 序列位置映射
        esm_mapping = build_pdb_to_esm_mapping(residues, esm_seq)
        if not esm_mapping:
            print(f"[WARN] {uid}: empty PDB-ESM mapping, skip.")
            continue

        # 4) 识别口袋残基
        pocket_simple = identify_pocket_residues(
            structure=structure,
            residues=residues,
            sub_resname=args.sub_resname,
            core_cutoff=args.core_cutoff,
            contact_cutoff=args.contact_cutoff,
            shell_hops=args.shell_hops,
        )

        if not pocket_simple:
            print(f"[WARN] {uid}: no pocket residues detected, skip.")
            continue

        # 5) 构图
        graph = build_graph_for_uid(
            uid=uid,
            structure=structure,
            residues=residues,
            pocket_simple=pocket_simple,
            sub_resname=args.sub_resname,
            tpp_resname=args.tpp_resname,
            protein_cutoff=args.protein_cutoff,
            prot_lig_cutoff=args.prot_lig_cutoff,
            rbf_k=args.rbf_k,
            rbf_max_dist=args.rbf_max_dist,
            esm_residue_emb=esm_residue_emb,
            esm_mapping=esm_mapping,
        )
        if graph is None:
            continue

        graph_path = os.path.join(args.out_graph_dir, f"{uid}.pt")
        torch.save(graph, graph_path)
        print(f"[OK] {uid}: graph -> {graph_path}")


if __name__ == "__main__":
    main()
