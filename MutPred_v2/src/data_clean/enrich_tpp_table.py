# -*- coding: utf-8 -*-
"""
脚本作用:
    1) 读取 TPP_family.xlsx 和 Substrate.xlsx
    2) 根据 "Protein names" 在 Substrate.xlsx 中查找底物信息, 为每条酶补充:
         - Substrate
         - Substrate_smiles
    3) 规范化 "Subunit structure" 与 "Cofactor" 列, 生成:
         - Subunit_clean: homodimer / homotetramer / unknown
         - ThDP_per_subunit
         - Metal_types
         - Metal_per_subunit
         - FAD_per_subunit
    4) 对 Oxalyl-CoA decarboxylase (EC 4.1.1.8) 在缺少金属注释时,
       强制补充 Mg2+ 1 个/亚基

输入:
    --tpp_xlsx:       TPP_family.xlsx
    --substrate_xlsx: Substrate.xlsx
    --out_xlsx:       输出的整理表路径

调用示例:
    python src/data_clean/enrich_tpp_table.py \
        --tpp_xlsx data/fasta/TPP_family.xlsx \
        --substrate_xlsx data/fasta/Substrate.xlsx \
        --out_xlsx data/fasta/TPP_family_clean.xlsx
"""

import argparse
import re
from typing import Dict, List, Tuple

import pandas as pd


def load_substrate_map(path: str) -> Dict[str, Tuple[str, str]]:
    """读取 Substrate.xlsx, 返回:
    key = Protein names 小写去空格 -> (Substrate, Substrate_smiles)
    """
    df = pd.read_excel(path)

    # Protein names 列
    if "Protein names" in df.columns:
        name_col = "Protein names"
    elif "Protein_names" in df.columns:
        name_col = "Protein_names"
    else:
        raise ValueError("Substrate.xlsx 中未找到 'Protein names' 或 'Protein_names' 列")

    # Substrate 列
    if "Substrate" in df.columns:
        sub_col = "Substrate"
    else:
        raise ValueError("Substrate.xlsx 中未找到 'Substrate' 列")

    # Smile / Smiles / SMILES 列 (不区分大小写)
    smile_col_candidates = [c for c in df.columns if c.lower() in {"smile", "smiles"}]
    if not smile_col_candidates:
        raise ValueError("Substrate.xlsx 中未找到 'Smile' / 'Smiles' / 'SMILES' 列")
    smile_col = smile_col_candidates[0]

    mapping: Dict[str, Tuple[str, str]] = {}
    for _, row in df.iterrows():
        pname_raw = str(row[name_col])
        pname_key = pname_raw.strip().lower()
        if not pname_key:
            continue
        substrate = str(row[sub_col]).strip()
        smiles = str(row[smile_col]).strip()
        mapping[pname_key] = (substrate, smiles)
    return mapping


def normalize_subunit(s: str) -> str:
    """把 Subunit structure 文本归一为 homodimer / homotetramer / unknown"""
    if not isinstance(s, str):
        return "unknown"
    s_low = s.lower()
    if "homodimer" in s_low:
        return "homodimer"
    if "homotetramer" in s_low or "tetramer" in s_low:
        return "homotetramer"
    return "unknown"


def get_subunit_count(subunit_clean: str) -> int:
    """根据标准化寡聚体类型推断亚基数"""
    if subunit_clean == "homodimer":
        return 2
    if subunit_clean == "homotetramer":
        return 4
    # 未知情况先按 1 处理, 后续可以人工筛掉
    return 1


def parse_cofactor_text(raw: str, subunit_clean: str) -> Tuple[float, Dict[str, float], float]:
    """解析 Cofactor 文本:
    返回:
        thdp_per_subunit: 每亚基 ThDP 数量
        metal_dict: {金属类型: 每亚基数量}
        fad_per_subunit: 每亚基 FAD 数量 (0 或 1)
    """
    thdp_per_subunit = 0.0
    metal_dict: Dict[str, float] = {}
    fad_per_subunit = 0.0

    if not isinstance(raw, str) or not raw.strip():
        return thdp_per_subunit, metal_dict, fad_per_subunit

    parts = raw.split("COFACTOR:")
    _ = get_subunit_count(subunit_clean)  # 目前 unit 换算不直接用到 count, 保留以备扩展

    for part in parts[1:]:
        text = part.strip()
        if not text:
            continue

        # Name=...;
        m_name = re.search(r"Name=([^;]+);", text)
        if not m_name:
            continue
        orig_name = m_name.group(1).strip()
        name_low = orig_name.lower()

        if "thiamine" in name_low:
            ctype = "ThDP"
        elif "mg(2+)" in name_low:
            ctype = "Mg2+"
        elif "mn(2+)" in name_low:
            ctype = "Mn2+"
        elif "ca(2+)" in name_low:
            ctype = "Ca2+"
        elif "fad" in name_low:
            ctype = "FAD"
        elif "a metal cation" in name_low:
            ctype = "metal_unspecified"
        else:
            # 其它辅因子暂时忽略
            continue

        # Note=Binds X ... per Y
        m_note = re.search(r"Binds\s+(\d+)\s+[^\s]+?\s+per\s+([a-zA-Z]+)", text)
        if m_note:
            count = float(m_note.group(1))
            unit = m_note.group(2).lower()
        else:
            # 未写 Note: 默认每亚基 1 个
            count = 1.0
            unit = "subunit"

        # 换算为每亚基数量
        if unit.startswith("subunit") or unit.startswith("monomer") or unit.startswith("chain"):
            per_subunit = count
        elif unit.startswith("dimer"):
            per_subunit = count / 2.0
        elif unit.startswith("tetramer"):
            per_subunit = count / 4.0
        else:
            per_subunit = count

        if ctype == "ThDP":
            thdp_per_subunit += per_subunit
        elif ctype == "FAD":
            # 你已经确认 FAD 基本是每亚基 1 个, 这里统一记 1.0
            fad_per_subunit = 1.0
        else:
            metal_dict[ctype] = metal_dict.get(ctype, 0.0) + per_subunit

    # 处理 a metal cation
    if "metal_unspecified" in metal_dict:
        unspecified = metal_dict.pop("metal_unspecified")
        if not metal_dict:
            # 没有任何具体金属, 默认 Mg2+
            metal_dict["Mg2+"] = metal_dict.get("Mg2+", 0.0) + unspecified
        # 若已有 Mg/Mn/Ca, 则忽略 metal_unspecified

    # 没解析到 ThDP 时默认 1 个/亚基
    if thdp_per_subunit <= 0.0:
        thdp_per_subunit = 1.0

    return thdp_per_subunit, metal_dict, fad_per_subunit


def metal_dict_to_strings(metal_dict: Dict[str, float]) -> Tuple[str, str]:
    """把 {金属: 数量} 转换为 Metal_types / Metal_per_subunit 两个字符串"""
    if not metal_dict:
        return "", ""
    metals_sorted = sorted(metal_dict.items(), key=lambda x: x[0])
    types = ";".join(m for m, _ in metals_sorted)
    amounts = ";".join(str(v) for _, v in metals_sorted)
    return types, amounts


def enrich_tpp_table(tpp_xlsx: str, substrate_xlsx: str, out_xlsx: str) -> None:
    """主流程: 合并底物信息 + 规范化 Subunit/Cofactor + Oxalyl-CoA 特殊处理"""

    substrate_map = load_substrate_map(substrate_xlsx)
    df = pd.read_excel(tpp_xlsx)

    # 列名确定
    if "Protein names" in df.columns:
        pname_col = "Protein names"
    elif "Protein_names" in df.columns:
        pname_col = "Protein_names"
    else:
        raise ValueError("TPP_family.xlsx 中未找到 'Protein names' 或 'Protein_names' 列")

    if "Subunit structure" in df.columns:
        subunit_col = "Subunit structure"
    elif "Subunit_structure" in df.columns:
        subunit_col = "Subunit_structure"
    else:
        raise ValueError("TPP_family.xlsx 中未找到 'Subunit structure' 列")

    if "Cofactor" in df.columns:
        cofactor_col = "Cofactor"
    else:
        raise ValueError("TPP_family.xlsx 中未找到 'Cofactor' 列")

    # 新列初始化
    df["Substrate"] = ""
    df["Substrate_smiles"] = ""
    df["Subunit_clean"] = ""
    df["ThDP_per_subunit"] = 0.0
    df["Metal_types"] = ""
    df["Metal_per_subunit"] = ""
    df["FAD_per_subunit"] = 0.0

    unmatched_pnames: List[str] = []
    no_cofactor_thdp: List[str] = []

    for idx, row in df.iterrows():
        pname_raw = str(row[pname_col])
        pname_key = pname_raw.strip().lower()

        # 1) 补充底物信息 (大小写不敏感匹配)
        if pname_key in substrate_map:
            sub_name, smiles = substrate_map[pname_key]
            df.at[idx, "Substrate"] = sub_name
            df.at[idx, "Substrate_smiles"] = smiles
        else:
            unmatched_pnames.append(pname_raw.strip())

        # 2) 规范化 Subunit
        subunit_raw = row[subunit_col]
        subunit_clean = normalize_subunit(subunit_raw)
        df.at[idx, "Subunit_clean"] = subunit_clean

        # 3) 解析 Cofactor
        cofactor_raw = row[cofactor_col]
        thdp_per_subunit, metal_dict, fad_per_subunit = parse_cofactor_text(
            cofactor_raw, subunit_clean
        )

        # 3.1 Oxalyl-CoA decarboxylase 特殊处理:
        # 若 Protein names 含 "oxalyl-coa decarboxylase" 或 "ec 4.1.1.8"
        # 且目前没有任何金属信息, 则强制设为 Mg2+ 1 个/亚基
        pname_low = pname_raw.lower()
        is_oxc = ("oxalyl-coa decarboxylase" in pname_low) or ("ec 4.1.1.8" in pname_low)
        if is_oxc and not metal_dict:
            metal_dict["Mg2+"] = 1.0

        df.at[idx, "ThDP_per_subunit"] = thdp_per_subunit
        metal_types, metal_per_subunit = metal_dict_to_strings(metal_dict)
        df.at[idx, "Metal_types"] = metal_types
        df.at[idx, "Metal_per_subunit"] = metal_per_subunit
        df.at[idx, "FAD_per_subunit"] = fad_per_subunit

        if thdp_per_subunit <= 0.0:
            no_cofactor_thdp.append(str(row.get("Entry", idx)))

    df.to_excel(out_xlsx, index=False)

    unmatched_unique = sorted(set([n for n in unmatched_pnames if n]))
    if unmatched_unique:
        print("警告: 以下 Protein names 在 Substrate.xlsx 中未找到底物映射, 请检查:")
        for name in unmatched_unique:
            print("  ", name)

    if no_cofactor_thdp:
        print("警告: 以下条目在 Cofactor 列中未解析到 thiamine 信息, 已默认 ThDP_per_subunit=1.0:")
        for entry in sorted(set(no_cofactor_thdp)):
            print("  ", entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpp_xlsx", required=True, help="TPP_family.xlsx 路径")
    parser.add_argument("--substrate_xlsx", required=True, help="Substrate.xlsx 路径")
    parser.add_argument("--out_xlsx", required=True, help="输出整理后的 Excel 路径")
    args = parser.parse_args()

    enrich_tpp_table(args.tpp_xlsx, args.substrate_xlsx, args.out_xlsx)


if __name__ == "__main__":
    main()
