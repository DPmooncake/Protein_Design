# -*- coding: utf-8 -*-
"""
脚本作用:
    根据代表序列 rep.fasta 与整理好的 TPP_family_clean.xlsx 构建 Protenix 所需的 json 文件。
    对于 rep.fasta 中的每一个 UniProt entry，按照 TPP_family_clean.xlsx 中的
    Subunit_clean / ThDP_per_subunit / Metal_types / Metal_per_subunit /
    Substrate_smiles 等信息，为每条序列单独生成一个 json 文件，
    文件内容格式为:

    [
      {
        "name": "A0A1J5RDM9",
        "sequences": [
          {
            "proteinChain": {
              "sequence": "...氨基酸序列...",
              "count": 4
            }
          },
          {
            "ligand": {
              "ligand": THDP_SMILES,
              "count": 4
            }
          },
          {
            "ion": {
              "ion": "MG",
              "count": 4
            }
          },
          {
            "ligand": {
              "ligand": Substrate_SMILES,
              "count": 1
            }
          }
        ]
      }
    ]

说明:
    1. name 字段使用 UniProt Entry (ID)，从 fasta 的 header 中解析:
         >sp|A0A0C1USY5|... 或 >A0A0C1USY5 ...
    2. 亚基数由 Subunit_clean 转换:
         - homodimer      -> 2
         - homotetramer   -> 4
         - 其它/unknown   -> 1
    3. ThDP 在 json 中作为 ligand，SMILES 用固定常量 THDP_SMILES。
    4. 金属离子:
         - 从 Metal_types / Metal_per_subunit 解析
         - 可有多个金属, 对每种金属生成一个 "ion" 条目
         - "ion" 字段仅写元素符号并大写, 例如 Mg2+ -> "MG"
    5. 当前版本为避免 RDKit SMILES 解析错误，暂时不在 json 中加入 FAD，
       但 FAD_per_subunit 仍保留在 TPP_family_clean.xlsx 中。

输入:
    --fasta: 代表序列 fasta 文件路径 (例如 rep.fasta)
    --xlsx:  TPP_family_clean.xlsx 路径 (由 enrich_tpp_table.py 生成)
    --outdir: 输出 json 文件夹路径，每条序列一个文件，文件名为 "<Entry>.json"

调用示例:
    python src/data_clean/build_json.py \
        --fasta data/fasta/TPP_family_mmseqs/rep.fasta \
        --xlsx data/fasta/TPP_family_clean.xlsx \
        --outdir data/json/tpp_family
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd


# ThDP 的 SMILES (与你之前示例保持一致)
THDP_SMILES = "n1c(C)nc(N)c(c1)CN1CSC(=C1C)CCO[P@](=O)([O-])OP(=O)([O-])[O-]"


def parse_fasta(path: str) -> Dict[str, str]:
    """读取 fasta 文件，返回:
    entry_id -> 序列

    entry_id 解析规则:
        >sp|A0A0C1USY5|xxx -> A0A0C1USY5
        >A0A0C1USY5 xxx   -> A0A0C1USY5
    """
    seqs: Dict[str, List[str]] = {}
    current_id = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:].split()[0]
                if "|" in header:
                    parts = header.split("|")
                    entry = parts[1] if len(parts) >= 2 else parts[0]
                else:
                    entry = header
                current_id = entry
                if current_id not in seqs:
                    seqs[current_id] = []
            else:
                if current_id is None:
                    continue
                seqs[current_id].append(line)

    return {k: "".join(v) for k, v in seqs.items()}


def get_subunit_count(subunit_clean: str) -> int:
    """根据 Subunit_clean 计算亚基数"""
    if subunit_clean == "homodimer":
        return 2
    if subunit_clean == "homotetramer":
        return 4
    return 1


def parse_metals(metal_types: str, metal_per_subunit: str) -> List[Tuple[str, float]]:
    """把 Metal_types / Metal_per_subunit 解析为列表:
    [("Mg2+", 1.0), ("Mn2+", 0.5), ...]
    """
    metals: List[Tuple[str, float]] = []
    if not isinstance(metal_types, str) or not metal_types.strip():
        return metals

    types = [t.strip() for t in metal_types.split(";") if t.strip()]
    if isinstance(metal_per_subunit, str) and metal_per_subunit.strip():
        amounts_str = [a.strip() for a in metal_per_subunit.split(";")]
    else:
        amounts_str = ["1"] * len(types)

    for i, mtype in enumerate(types):
        if i < len(amounts_str):
            try:
                amt = float(amounts_str[i])
            except ValueError:
                amt = 1.0
        else:
            amt = 1.0
        metals.append((mtype, amt))
    return metals


def metal_type_to_ion_symbol(metal_type: str) -> str:
    """例如 Mg2+ -> MG, Mn2+ -> MN, Ca2+ -> CA"""
    letters = "".join(ch for ch in metal_type if ch.isalpha())
    return letters.upper()


def build_entry_json(entry_id: str,
                     sequence: str,
                     row: pd.Series) -> Dict:
    """由一条 TPP_family_clean 的记录 + 序列构建单个 entry 的 json dict"""

    # 1. 亚基与基本信息
    subunit_clean = str(row.get("Subunit_clean", "unknown"))
    subunit_count = get_subunit_count(subunit_clean)

    # 2. ThDP 总数
    thdp_per_subunit = float(row.get("ThDP_per_subunit", 1.0))
    thdp_total = int(round(thdp_per_subunit * subunit_count))

    # 3. 金属离子
    metal_types = row.get("Metal_types", "")
    metal_per_subunit = row.get("Metal_per_subunit", "")
    metals = parse_metals(metal_types, metal_per_subunit)

    # 4. 底物
    substrate_smiles = str(row.get("Substrate_smiles", "")).strip()

    sequences_block: List[Dict] = []

    # 蛋白链
    sequences_block.append({
        "proteinChain": {
            "sequence": sequence,
            "count": subunit_count
        }
    })

    # ThDP 作为 ligand
    if thdp_total > 0:
        sequences_block.append({
            "ligand": {
                "ligand": THDP_SMILES,
                "count": thdp_total
            }
        })

    # 金属离子
    for mtype, per_subunit in metals:
        total = int(round(per_subunit * subunit_count))
        if total <= 0:
            continue
        ion_symbol = metal_type_to_ion_symbol(mtype)
        sequences_block.append({
            "ion": {
                "ion": ion_symbol,
                "count": total
            }
        })

    # 底物 ligand（只用 SMILES）
    if substrate_smiles:
        sequences_block.append({
            "ligand": {
                "ligand": substrate_smiles,
                "count": 1
            }
        })

    entry_json = {
        "name": entry_id,
        "sequences": sequences_block
    }
    return entry_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="代表序列 fasta 文件路径，例如 rep.fasta")
    parser.add_argument("--xlsx", required=True, help="TPP_family_clean.xlsx 路径")
    parser.add_argument("--outdir", required=True, help="输出 json 文件夹路径，每条序列一个文件")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读取 fasta
    fasta_seqs = parse_fasta(args.fasta)

    # 读取表格
    df = pd.read_excel(args.xlsx)
    if "Entry" not in df.columns:
        raise ValueError("TPP_family_clean.xlsx 中未找到 'Entry' 列")
    df = df.set_index("Entry", drop=False)

    missing_in_table: List[str] = []
    written = 0

    for entry_id, seq in fasta_seqs.items():
        if entry_id not in df.index:
            missing_in_table.append(entry_id)
            continue

        row = df.loc[entry_id]
        entry_json = build_entry_json(entry_id, seq, row)

        out_path = os.path.join(args.outdir, f"{entry_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            # 与之前约定保持一致: 每个文件最外层是一个列表
            json.dump([entry_json], f, indent=2, ensure_ascii=False)

        written += 1

    print(f"已为 {written} 条序列生成 json 文件，输出目录: {args.outdir}")
    if missing_in_table:
        print("警告: 下列 fasta 中的 Entry 在 TPP_family_clean.xlsx 中未找到, 已跳过:")
        for eid in sorted(missing_in_table):
            print("  ", eid)


if __name__ == "__main__":
    main()
