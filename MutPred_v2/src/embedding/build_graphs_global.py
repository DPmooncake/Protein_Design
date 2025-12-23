# 作用: 从 holo PDB + ESM-2 embedding 构建用于 EGNN 的全局/口袋异质图(pt)
# 输入: --pdb_dir, --esm_dir, --out_graph_dir, --sub_resname, --tpp_resname, --graph_mode 等(详见 argparse)
# 输出: out_graph_dir/{uid}.pt (包含 pos/x/x_esm/edge_index/edge_attr/pocket_mask 等字段)
# 调用: python src/embedding/build_graphs_global.py --pdb_dir ... --esm_dir ... --out_graph_dir ... --graph_mode global

"""
build_graphs_global.py

作用:
    1) 从 WT holo PDB (蛋白 + 底物 + TPP + Mg, 可为寡聚体) 中识别催化口袋残基;
    2) 基于 "蛋白残基 + 底物 super-node + TPP super-node + Mg" 构建异质图;
    3) 将 ESM-2 残基级别 embedding 对齐到 PDB 残基, 作为额外节点特征写入图文件;
    4) 使用全局蛋白 Cα 坐标构建蛋白–蛋白 k 近邻图, 为后续 EGNN 提供拓扑结构。

图模式:
    - graph_mode = "pocket":
        仅对口袋残基建蛋白节点 (core + shell), 非口袋残基不进入图。
    - graph_mode = "global":
        对所有蛋白残基建节点 (全局图), 但仍用 pocket_mask / pocket_layer 标记哪些是 core/shell,
        便于后续只在口袋区域做监督或统计。

蛋白–蛋白连边策略 (全局结构):
    - 不再使用固定距离 cutoff + RBF 编码;
    - 对所有蛋白节点基于 Cα 坐标构建 kNN 图:
        * 对每个蛋白节点 i, 在其它蛋白节点中寻找 k 个最近邻 j;
        * k 由命令行参数 --protein_cutoff 控制 (会取上界, 见下方);
        * 对每条蛋白–蛋白边 (i, j) 建双向边 i->j, j->i;
        * 几何信息 (距离、方向) 不再写入 edge_attr, 由 EGNN 直接用 pos 计算。

蛋白–配体连边策略:
    - 对蛋白–底物、蛋白–TPP、蛋白–Mg 使用欧氏距离 cutoff;
    - 距离阈值由 --prot_lig_cutoff 控制 (单位 Å);
    - 若任一蛋白原子与配体原子距离 <= cutoff, 即在蛋白节点与对应 super-node 之间加双向边。

输入:
    - --pdb_dir:         存放 WT holo PDB 的目录, 每个文件名形如 {uid}.pdb;
    - --esm_dir:         存放 ESM-2 提取的序列特征目录, 内含 {uid}.pt, 结构为:
                             {
                                 "seq_id": uid,
                                 "seq": "ACDE...",
                                 "residue_emb": FloatTensor [L, D],  # 残基级别 embedding
                                 "seq_emb": FloatTensor [D],         # 序列级别 embedding
                             }
    - --out_graph_dir:   输出图 PT 文件目录, 每个图保存为 {uid}.pt;
    - --pocket_json:     可选, 预先识别好的 pocket 信息 (simple 格式 JSON), 不用可不传;
    - --sub_resname:     底物在 PDB 中的 resName (如 l02 / SUB 等);
    - --tpp_resname:     TPP 在 PDB 中的 resName (如 l01 / TPP);

    - --core_cutoff:     定义 pocket core 残基的蛋白–底物最小距离阈值 (Å);
    - --contact_cutoff:  构建残基接触图时的 Cα–Cα 距离阈值 (Å), 用于从 core 扩展 shell;
    - --shell_hops:      以 core 残基为中心, 在残基接触图上向外扩展 shell 的 hop 数;

    - --graph_mode:      "pocket" 或 "global";
    - --protein_cutoff:  蛋白–蛋白 kNN 中的 k, 取值规则:
                             k = ceil(protein_cutoff),
                             同时强制 1 <= k <= (P - 1), 其中 P 为该结构中的蛋白节点数;
                         例如:
                             --protein_cutoff 16  -> 每个残基连 16 个最近邻;
                             --protein_cutoff 128 -> 实际 k 会被截断为 min(128, P-1);
    - --prot_lig_cutoff: 蛋白–底物/TPP/Mg 连边的距离阈值 (Å), 例如 5.0;
    - --batch_size:      一次处理多少个 PDB 文件 (只影响进度条刷新频率)。

输出 (graph_mode 两种模式下结构一致, 只是蛋白节点数不同):
    对每个成功处理的 uid, 在 out_graph_dir 下生成 {uid}.pt, 内容为字典:
        {
            "uid": uid: str,

            # 节点信息
            "pos": FloatTensor [N, 3],            # 节点坐标 (蛋白 Cα 或 super-node 中心), 供 EGNN 使用
            "x": FloatTensor [N, F_struct],       # 结构+类型节点特征:
                                                 #   - 20 维 AA one-hot (蛋白节点)
                                                 #   - node_type one-hot 等
                                                 #   - backbone torsion, d_sub/d_tpp/d_mg 等
            "x_esm": FloatTensor [N, D_esm],      # ESM 残基 embedding (非蛋白节点为 0 向量)
            "node_type": LongTensor [N],          # 节点类型: 0=蛋白, 1=sub, 2=tpp, 3=mg

            # 边信息
            "edge_index": LongTensor [2, E],      # 有向边 (src, dst)
            "edge_attr": FloatTensor [E, 6],      # 边特征(不含几何信息), 统一为:
                                                 #   - 4 维 edge_type one-hot
                                                 #   - 1 维 Δresseq (仅 prot-prot 有意义, 其他置 0)
                                                 #   - 1 维 Δchain (仅 prot-prot 有意义, 其他置 0)

            # 节点索引分组 (方便后续统计/可视化)
            "prot_node_idx": List[int],           # 蛋白节点在全图中的下标
            "sub_node_idx": List[int],            # 底物 super-node 下标 (通常 1 个)
            "tpp_node_idx": List[int],            # TPP super-node 下标
            "mg_node_idx": List[int],             # Mg super-node 下标

            # 残基/链索引
            "node_chain_idx": LongTensor [N],     # 节点所属链的整数编号 (蛋白), 非蛋白为 0 或 -1
            "node_resseq": LongTensor [N],        # 节点所属残基号 (蛋白), 非蛋白为 0

            # 口袋标记
            "pocket_mask": BoolTensor [N],        # True 表示蛋白 pocket 残基节点 (core+shell)
            "pocket_layer": LongTensor [N],       # -1=非 pocket, 0=core, 1=shell
        }

调用格式示例:

    # 1) 仅口袋节点 (原始口袋图, 只在 core+shell 上建蛋白节点)
    python src/embedding/build_graphs_global.py \
        --pdb_dir data/pdb/tpp_family \
        --esm_dir results/sequence_features \
        --out_graph_dir data/graphs_pocket \
        --sub_resname l02 \
        --tpp_resname l01 \
        --core_cutoff 4.5 \
        --contact_cutoff 5.0 \
        --shell_hops 1 \
        --graph_mode pocket \
        --protein_cutoff 16 \
        --prot_lig_cutoff 5.0 \
        --batch_size 16

    # 2) 全局蛋白图 (所有残基建节点) + 保留 pocket_mask/pocket_layer 方便后续口袋监督
    python src/embedding/build_graphs_global.py \
        --pdb_dir data/pdb/tpp_family \
        --esm_dir results/sequence_features \
        --out_graph_dir data/graphs_global_k16 \
        --sub_resname l02 \
        --tpp_resname l01 \
        --core_cutoff 4.5 \
        --contact_cutoff 5.0 \
        --shell_hops 1 \
        --graph_mode global \
        --protein_cutoff 16 \
        --prot_lig_cutoff 5.0 \
        --batch_size 16

    python src/embedding/build_graphs_global.py \
        --pdb_dir data/test/pdbs \
        --esm_dir data/test/sequence_features \
        --out_graph_dir data/test/graphs \
        --sub_resname l02 \
        --tpp_resname l01 \
        --core_cutoff 4.5 \
        --contact_cutoff 5.0 \
        --shell_hops 1 \
        --graph_mode global \
        --protein_cutoff 16 \
        --prot_lig_cutoff 5.0 \
        --batch_size 16

"""

import os
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import math
import torch
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio import pairwise2

# ======================
# 一些基础常量和工具
# ======================

AA_TYPES_3 = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19
}
AA_DIM = len(AA_TYPES_3)

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

# 保留最大链数的上限用于整数链编号（用于 Δchain）
MAX_CHAIN_DIM = 4


def one_hot(index: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    if 0 <= index < dim:
        v[index] = 1.0
    return v


def atoms_center(atoms) -> Optional[np.ndarray]:
    if not atoms:
        return None
    coords = np.array([a.get_coord() for a in atoms], dtype=np.float32)
    return coords.mean(axis=0)


# ==========================
# 口袋识别相关
# ==========================

def collect_substrate_atoms(structure, sub_resname: str):
    """
    收集所有底物残基的原子 (假定底物以特定 resname 标记).
    返回:
      - sub_atoms: List[Atom]
    """
    sub_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == sub_resname:
                    for atom in residue:
                        sub_atoms.append(atom)
    return sub_atoms


def collect_protein_residues(structure):
    """
    收集蛋白残基 (标准 20 AA), 返回列表, 每个元素包含:
      {
        "chain_id": str,
        "resseq": int,
        "icode": str,
        "residue": Residue,
        "ca_coord": np.ndarray [3]
      }
    """
    residues = []
    for model in structure:
        for chain in model:
            cid = chain.id
            for residue in chain:
                resname = residue.get_resname()
                if resname not in AA_TYPES_3:
                    continue
                if not residue.has_id("CA"):
                    continue
                ca_coord = residue["CA"].get_coord().astype(np.float32)
                res_id = residue.get_id()
                resseq = int(res_id[1])
                icode = res_id[2] if len(res_id) > 2 else " "
                residues.append({
                    "chain_id": cid,
                    "resseq": resseq,
                    "icode": icode,
                    "residue": residue,
                    "ca_coord": ca_coord,
                })
    return residues


def compute_min_dist_residue_to_sub(residue_entry, sub_coords: np.ndarray) -> float:
    """
    计算某个残基 (所有原子) 到底物坐标集合的最小距离。
    """
    residue = residue_entry["residue"]
    coords = np.array([atom.get_coord() for atom in residue], dtype=np.float32)
    diff = coords[:, None, :] - sub_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return float(dists.min())


def build_contact_graph(residues, contact_cutoff: float):
    """
    基于 Cα 构建残基接触图，返回邻接表 adjacency[i] = [j1, j2,...]
    用于从 core pocket 残基扩展 shell。
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

    shell = list(visited - core_set)
    return shell


def identify_pocket_residues(structure,
                             residues: List[Dict],
                             sub_resname: str,
                             core_cutoff: float = 5.0,
                             shell_hops: int = 1,
                             contact_cutoff: float = 8.0):
    """
    识别 pocket 残基:
      1) core: 所有到底物最小距离 <= core_cutoff 的残基;
      2) shell: 在残基接触图上, 从 core 扩展 shell_hops 层邻居.
    返回 pocket_residues 列表, 每个元素结构为:
      {
        "chain_id": str,
        "resseq": int,
        "icode": str,
        "residue": Residue,
        "ca_coord": np.ndarray [3],
        "layer": int,  # 0 for core, 1..shell
      }
    """
    sub_atoms = collect_substrate_atoms(structure, sub_resname)
    if not sub_atoms:
        return []

    sub_coords = np.array([a.get_coord() for a in sub_atoms], dtype=np.float32)

    # core 残基
    core_indices = []
    for i, r in enumerate(residues):
        dmin = compute_min_dist_residue_to_sub(r, sub_coords)
        if dmin <= core_cutoff:
            core_indices.append(i)

    if not core_indices:
        return []

    adjacency = build_contact_graph(residues, contact_cutoff)
    shell_indices = bfs_shell(core_indices, adjacency, shell_hops)

    pocket_entries: List[Dict] = []
    core_set = set(core_indices)
    for idx in core_indices:
        entry = residues[idx].copy()
        entry["layer"] = 0
        pocket_entries.append(entry)
    for idx in shell_indices:
        entry = residues[idx].copy()
        entry["layer"] = 1
        pocket_entries.append(entry)

    return pocket_entries


# ==========================
# Backbone torsion 计算
# ==========================

def _dihedral(p0: np.ndarray, p1: np.ndarray,
              p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算四个点定义的二面角，返回弧度制。
    """
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = b1 / (np.linalg.norm(b1) + 1e-8)

    v = b0 - np.dot(b0, b1_norm) * b1_norm
    w = b2 - np.dot(b2, b1_norm) * b1_norm

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_norm, v), w)
    angle = np.arctan2(y, x)
    return float(angle)


def compute_backbone_torsions(
    residues: List[Dict]
) -> Dict[Tuple[str, int, str], np.ndarray]:
    """
    基于 residues 列表, 计算每个残基的 backbone torsion:
      φ, ψ, ω 的 sin/cos, 一共 6 维。

    定义:
      φ(i) = C(i-1) - N(i) - CA(i) - C(i)
      ψ(i) = N(i)   - CA(i) - C(i)  - N(i+1)
      ω(i) = CA(i-1)- C(i-1)- N(i)  - CA(i)

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
        if cid not in chain_to_res:
            chain_to_res[cid] = []
        chain_to_res[cid].append(entry)

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

                # φ(i) = C(i-1) - N(i) - CA(i) - C(i)
                if prev_res is not None and prev_res.has_id("C"):
                    C_prev = prev_res["C"].get_coord()
                    phi = _dihedral(C_prev, N_i, CA_i, C_i)

                # ψ(i) = N(i) - CA(i) - C(i) - N(i+1)
                if next_res is not None and next_res.has_id("N"):
                    N_next = next_res["N"].get_coord()
                    psi = _dihedral(N_i, CA_i, C_i, N_next)

                # ω(i) = CA(i-1) - C(i-1) - N(i) - CA(i)
                if prev_res is not None and prev_res.has_id("CA") and prev_res.has_id("C"):
                    CA_prev = prev_res["CA"].get_coord()
                    C_prev = prev_res["C"].get_coord()
                    omega = _dihedral(CA_prev, C_prev, N_i, CA_i)

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
    为了与 ESM residue_emb 对齐, 我们需要有一条 "线性序列"。
    """
    seq_chars = []
    keys = []
    for r in residues:
        cid = r["chain_id"]
        resseq = r["resseq"]
        icode = r["icode"]
        resname = r["residue"].get_resname()
        if resname not in AA_TYPES_3:
            continue
        idx = AA_TYPES_3[resname]
        aa_char = "ARNDCQEGHILKMFPSTWYV"[idx]
        seq_chars.append(aa_char)
        keys.append((cid, resseq, icode))
    seq_str = "".join(seq_chars)
    return seq_str, keys


def build_pdb_to_esm_mapping(
    pdb_seq: str,
    pdb_keys: List[Tuple[str, int, str]],
    esm_seq: str,
) -> Dict[Tuple[str, int, str], int]:
    """
    将 PDB 残基序列与 ESM 提取的序列比对, 生成映射:
      PDB key -> ESM residue_emb 的索引。
    """
    alignments = pairwise2.align.globalxx(pdb_seq, esm_seq, one_alignment_only=True)
    if not alignments:
        return {}

    aln = alignments[0]
    seqA, seqB, score, start, end = aln

    pdb_idx = 0
    esm_idx = 0
    mapping: Dict[Tuple[str, int, str], int] = {}

    for a, b in zip(seqA, seqB):
        if a != "-":
            pdb_key = pdb_keys[pdb_idx]
            pdb_idx += 1
        else:
            pdb_key = None
        if b != "-":
            esm_pos = esm_idx
            esm_idx += 1
        else:
            esm_pos = None

        if pdb_key is not None and esm_pos is not None:
            mapping[pdb_key] = esm_pos

    return mapping


def residue_repr_coord(residue_entry) -> Optional[np.ndarray]:
    """
    为残基选择一个代表坐标 (当前使用 Cα).
    """
    return residue_entry["ca_coord"]


def collect_ligand_atoms_for_graph(structure, sub_resname: str, tpp_resname: str):
    """
    收集用于图构建的底物/TPP/Mg 原子:
      - 底物: 所有匹配 sub_resname 的原子
      - TPP:  所有匹配 tpp_resname 的残基 (后续取质心)
      - Mg:   所有 Mg 原子
    """
    sub_atoms = []
    tpp_residues = []
    mg_atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                if resname == sub_resname:
                    for atom in residue:
                        sub_atoms.append(atom)
                elif resname == tpp_resname:
                    tpp_residues.append(residue)

    # Mg 原子
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "MG":
                    for atom in residue:
                        mg_atoms.append(atom)

    return sub_atoms, tpp_residues, mg_atoms


def find_nearest_tpp_and_mg(center_sub: np.ndarray,
                            tpp_residues,
                            mg_atoms):
    """
    在多个 TPP / Mg 中，选出与底物中心最近的一套。
    """
    tpp_atoms_selected = []
    mg_atom_selected = []

    if tpp_residues:
        min_dist = float("inf")
        best_tpp = None
        for res in tpp_residues:
            coords = np.array([a.get_coord() for a in res], dtype=np.float32)
            center = coords.mean(axis=0)
            d = np.linalg.norm(center - center_sub)
            if d < min_dist:
                min_dist = d
                best_tpp = res
        if best_tpp is not None:
            for atom in best_tpp:
                tpp_atoms_selected.append(atom)

    if mg_atoms:
        min_dist = float("inf")
        best_mg = None
        for atom in mg_atoms:
            d = np.linalg.norm(atom.get_coord() - center_sub)
            if d < min_dist:
                min_dist = d
                best_mg = atom
        if best_mg is not None:
            mg_atom_selected.append(best_mg)

    return tpp_atoms_selected, mg_atom_selected


def build_graph_for_uid(uid: str,
                        structure,
                        residues: List[Dict],
                        pocket_simple: List[Dict],
                        sub_resname: str,
                        tpp_resname: str,
                        protein_cutoff: float,
                        prot_lig_cutoff: float,
                        esm_residue_emb: np.ndarray,
                        esm_mapping: Dict[Tuple[str, int, str], int],
                        core_cutoff: float = 5.0,
                        shell_hops: int = 1,
                        contact_cutoff: float = 8.0,
                        graph_mode: str = "pocket"):
    """
    从结构和 pocket 残基列表构建异质图, 并接入 ESM 残基 embedding。

    参数说明:
      - protein_cutoff: 用作蛋白-蛋白 kNN 的近邻个数 k (向上取整), 不再作为距离阈值;
      - prot_lig_cutoff: 蛋白-底物/TPP/Mg 的聚边距离阈值;
      - core_cutoff / shell_hops / contact_cutoff: 控制 pocket core/shell 识别;
    """
    if not residues:
        print(f"[WARN] {uid}: no protein residues, skip graph.")
        return None

    # 1) 计算 backbone torsion
    torsion_dict = compute_backbone_torsions(residues)

    # 2) 决定 pocket 残基集合 (使用 simple pocket 或自动识别)
    pocket_residues: List[Dict] = []
    if pocket_simple:
        for r in pocket_simple:
            pocket_residues.append({
                "chain_id": r["chain_id"],
                "resseq": r["resseq"],
                "icode": r["icode"],
                "residue": r["residue"],
                "ca_coord": r["ca_coord"],
                "layer": r["layer"],
            })
    else:
        pocket_residues = identify_pocket_residues(
            structure,
            residues,
            sub_resname=sub_resname,
            core_cutoff=core_cutoff,
            shell_hops=shell_hops,
            contact_cutoff=contact_cutoff,
        )

    if not pocket_residues:
        print(f"[WARN] {uid}: no valid pocket residues found in residues list.")
        return None

    pocket_key_to_layer = {
        (r["chain_id"], r["resseq"], r["icode"]): r["layer"]
        for r in pocket_residues
    }

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

    pocket_mask: List[bool] = []
    pocket_layer: List[int] = []

    prot_node_idx: List[int] = []
    sub_node_idx: List[int] = []
    tpp_node_idx: List[int] = []
    mg_node_idx: List[int] = []

    # 链 id -> 整数编号
    all_chain_ids = sorted({r["chain_id"] for r in residues})
    chain_id_to_idx = {cid: i for i, cid in enumerate(all_chain_ids)}
    for cid, idx in chain_id_to_idx.items():
        if idx >= MAX_CHAIN_DIM:
            chain_id_to_idx[cid] = MAX_CHAIN_DIM - 1

    esm_dim = esm_residue_emb.shape[1] if esm_residue_emb is not None else 0

    def build_protein_node_feat(res_entry: Dict) -> Tuple[np.ndarray, int, int]:
        """
        为一个蛋白残基构建节点特征:
          x = [AA one-hot (20), node_type one-hot (4), torsion(6), d_sub/d_tpp/d_mg(3)]
        其中 d_sub/d_tpp/d_mg 先用 0 占位, 后续统一填充。
        """
        resname = res_entry["residue"].get_resname()
        aa_oh = np.zeros(AA_DIM, dtype=np.float32)
        if resname in AA_TYPES_3:
            aa_oh[AA_TYPES_3[resname]] = 1.0
        node_type_oh = one_hot(NODE_TYPE_MAP["protein"], len(NODE_TYPE_MAP))

        key = (res_entry["chain_id"], res_entry["resseq"], res_entry["icode"])
        torsion = torsion_dict.get(key, np.zeros(6, dtype=np.float32))

        dist_feat = np.zeros(3, dtype=np.float32)

        feat = np.concatenate([aa_oh, node_type_oh, torsion, dist_feat], axis=0)

        cid = res_entry["chain_id"]
        cid_idx = chain_id_to_idx.get(cid, 0)
        resseq = int(res_entry["resseq"])

        return feat, cid_idx, resseq

    # 3.1 蛋白节点: 根据 graph_mode 决定仅 pocket 还是全局
    if graph_mode == "pocket":
        protein_nodes = pocket_residues
    elif graph_mode == "global":
        protein_nodes = []
        for r in residues:
            key = (r["chain_id"], r["resseq"], r["icode"])
            layer = pocket_key_to_layer.get(key, None)
            protein_nodes.append({
                "chain_id": r["chain_id"],
                "resseq": r["resseq"],
                "icode": r["icode"],
                "layer": layer,
                "residue": r["residue"],
                "ca_coord": r["ca_coord"],
            })
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")

    # 3.1.1 逐个添加蛋白节点
    for res_entry in protein_nodes:
        coord = residue_repr_coord(res_entry)
        if coord is None:
            continue
        node_coords.append(coord)
        feat, cid_idx, resseq = build_protein_node_feat(res_entry)
        node_feats.append(feat)
        node_types.append(NODE_TYPE_MAP["protein"])
        node_chain_idx.append(cid_idx)
        node_resseq.append(resseq)
        prot_node_idx.append(len(node_coords) - 1)
        # pocket 标记
        key = (res_entry["chain_id"], res_entry["resseq"], res_entry["icode"])
        layer = pocket_key_to_layer.get(key, -1)
        is_pocket = layer is not None and layer >= 0
        pocket_mask.append(is_pocket)
        pocket_layer.append(layer if is_pocket else -1)

        # ESM 节点特征
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

    # 3.4 Mg super-node
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

    # 转为数组
    node_coords = np.stack(node_coords, axis=0).astype(np.float32)
    node_feats = np.stack(node_feats, axis=0).astype(np.float32)
    node_types_arr = np.array(node_types, dtype=np.int64)
    node_chain_idx_arr = np.array(node_chain_idx, dtype=np.int64)
    node_resseq_arr = np.array(node_resseq, dtype=np.int64)
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
          - edge_type one-hot
          - rel_resseq (Δresseq)
          - rel_chain (Δchain)
        几何相关信息 (距离 / 方向) 不再编码进 edge_attr,
        后续由 EGNN 在模型内部基于 pos 和 edge_index 自行计算。
        """
        edge_type_oh = np.zeros(len(EDGE_TYPE_MAP), dtype=np.float32)
        if edge_type in EDGE_TYPE_MAP:
            edge_type_oh[EDGE_TYPE_MAP[edge_type]] = 1.0

        if edge_type == "prot-prot":
            rel_resseq = node_resseq_arr[j] - node_resseq_arr[i]
            rel_resseq = float(max(-32, min(32, rel_resseq)))
            rel_chain = float(node_chain_idx_arr[j] - node_chain_idx_arr[i])
        else:
            rel_resseq = 0.0
            rel_chain = 0.0

        feat = np.concatenate(
            [edge_type_oh, np.array([rel_resseq, rel_chain], dtype=np.float32)],
            axis=0,
        )
        edges_src.append(i)
        edges_dst.append(j)
        edge_feats.append(feat)

    # 5.1 蛋白-蛋白边: 使用 kNN 图 (每个蛋白残基连向 k 个最近的蛋白残基)
    if len(prot_node_idx) > 1:
        prot_indices = np.array(prot_node_idx, dtype=np.int64)  # 蛋白节点在全局节点列表中的索引
        prot_coords = node_coords[prot_indices]  # [P, 3]
        P = prot_coords.shape[0]
        # k 至少为 1, 至多为 P-1
        k = int(max(1, min(P - 1, math.ceil(protein_cutoff))))
        # 计算成对距离平方矩阵 [P, P]
        diff = prot_coords[:, None, :] - prot_coords[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        # 避免选到自身
        np.fill_diagonal(dist2, np.float32(1e8))
        # 对每个 i 取距离最近的 k 个邻居
        nn_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]  # [P, k]

        for i_local in range(P):
            i_global = prot_indices[i_local]
            for j_local in nn_idx[i_local]:
                j_local = int(j_local)
                j_global = prot_indices[j_local]
                if i_global == j_global:
                    continue
                # 为避免重复, 只在 i_local < j_local 时添加 pair, 并显式加入双向边
                if i_local < j_local:
                    add_edge(i_global, j_global, "prot-prot")
                    add_edge(j_global, i_global, "prot-prot")

    # 5.2 蛋白-配体/TPP/Mg (仍然使用 prot_lig_cutoff 作为距离阈值)
    if sub_node_idx:
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

    if not edges_src:
        print(f"[WARN] {uid}: no edges constructed, skip graph.")
        return None

    edge_index = np.stack(
        [np.array(edges_src, dtype=np.int64),
         np.array(edges_dst, dtype=np.int64)],
        axis=0,
    )
    edge_attr = np.stack(edge_feats, axis=0).astype(np.float32)

    # 转成 torch.Tensor
    pos_t = torch.from_numpy(node_coords)              # [N, 3]
    x_t = torch.from_numpy(node_feats)                 # [N, F_struct]
    x_esm_t = torch.from_numpy(node_esm_feats)         # [N, D_esm]
    node_type_t = torch.from_numpy(node_types_arr)     # [N]
    edge_index_t = torch.from_numpy(edge_index)        # [2, E]
    edge_attr_t = torch.from_numpy(edge_attr)          # [E, 6]
    node_chain_idx_t = torch.from_numpy(node_chain_idx_arr)  # [N]
    node_resseq_t = torch.from_numpy(node_resseq_arr)        # [N]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", type=str, required=True,
                    help="WT holo PDB 目录, 文件名形如 {uid}.pdb")
    ap.add_argument("--esm_dir", type=str, required=True,
                    help="ESM-2 提取的序列特征目录, 内含 {uid}.pt")
    ap.add_argument("--pocket_json", type=str, default=None,
                    help="可选, 预先识别的 pocket 信息 JSON (simple 格式)")
    ap.add_argument("--out_graph_dir", type=str, required=True,
                    help="输出图文件目录")

    ap.add_argument("--sub_resname", type=str, default="SUB",
                    help="底物残基名称 (PDB resname)")
    ap.add_argument("--tpp_resname", type=str, default="TPP",
                    help="TPP 残基名称 (PDB resname)")

    # pocket 识别参数
    ap.add_argument("--core_cutoff", type=float, default=5.0,
                    help="定义 core pocket 残基的距离阈值 (Å, 到底物最小原子-原子距离)")
    ap.add_argument("--contact_cutoff", type=float, default=8.0,
                    help="构建残基接触图的 Cα–Cα 距离阈值 (Å), 用于从 core 扩展 shell")
    ap.add_argument("--shell_hops", type=int, default=1,
                    help="从 core 往外扩展的 shell 层数")

    # 图构建参数
    ap.add_argument("--protein_cutoff", type=float, default=16.0,
                    help="蛋白-蛋白 kNN 的近邻个数 k (向上取整, 不再作为距离阈值)")
    ap.add_argument("--prot_lig_cutoff", type=float, default=5.0,
                    help="蛋白-配体/TPP/Mg 聚边距离阈值 (Å)")

    ap.add_argument("--graph_mode", type=str, default="pocket",
                    choices=["pocket", "global"],
                    help="图模式: pocket 仅对口袋残基建蛋白节点; global 对所有残基建节点")

    ap.add_argument("--batch_size", type=int, default=16,
                    help="一次处理多少个 PDB (仅影响进度显示, 不影响图本身)")

    args = ap.parse_args()

    os.makedirs(args.out_graph_dir, exist_ok=True)

    pdb_parser = PDBParser(QUIET=True)

    pdb_files = [
        f for f in os.listdir(args.pdb_dir)
        if f.lower().endswith(".pdb")
    ]
    pdb_files.sort()

    # 如果将来要用 pocket_json, 可以在这里读入并按 uid 建 map
    pocket_info = {}

    with tqdm(total=len(pdb_files), desc="Building graphs") as pbar:
        for fname in pdb_files:
            uid = os.path.splitext(fname)[0]
            pdb_path = os.path.join(args.pdb_dir, fname)
            esm_path = os.path.join(args.esm_dir, f"{uid}.pt")

            if not os.path.exists(esm_path):
                print(f"[WARN] {uid}: no ESM file: {esm_path}, skip.")
                pbar.update(1)
                continue

            try:
                structure = pdb_parser.get_structure(uid, pdb_path)
            except Exception as e:
                print(f"[ERROR] parsing PDB {pdb_path}: {e}")
                pbar.update(1)
                continue

            residues = collect_protein_residues(structure)
            if not residues:
                print(f"[WARN] {uid}: no protein residues, skip.")
                pbar.update(1)
                continue

            # ESM embedding
            try:
                esm_data = torch.load(esm_path, map_location="cpu")
            except Exception as e:
                print(f"[ERROR] loading ESM file {esm_path}: {e}")
                pbar.update(1)
                continue

            esm_seq = esm_data["seq"]
            esm_residue_emb = esm_data["residue_emb"].numpy()

            # PDB 序列与 ESM 序列对齐
            pdb_seq, pdb_keys = build_pdb_seq_and_keys(residues)
            esm_mapping = build_pdb_to_esm_mapping(
                pdb_seq, pdb_keys, esm_seq
            )

            pocket_simple = pocket_info.get(uid, None)

            graph = build_graph_for_uid(
                uid=uid,
                structure=structure,
                residues=residues,
                pocket_simple=pocket_simple,
                sub_resname=args.sub_resname,
                tpp_resname=args.tpp_resname,
                protein_cutoff=args.protein_cutoff,
                prot_lig_cutoff=args.prot_lig_cutoff,
                esm_residue_emb=esm_residue_emb,
                esm_mapping=esm_mapping,
                core_cutoff=args.core_cutoff,
                shell_hops=args.shell_hops,
                contact_cutoff=args.contact_cutoff,
                graph_mode=args.graph_mode,
            )
            if graph is not None:
                graph_path = os.path.join(args.out_graph_dir, f"{uid}.pt")
                torch.save(graph, graph_path)

            pbar.update(1)


if __name__ == "__main__":
    main()
