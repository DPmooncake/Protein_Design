# -*- coding: utf-8 -*-
"""
prepare_wet_supervision.py

作用:
    从 brenda_data.xlsx 中整理湿实验动力学数据, 构建用于监督头训练的表 wet_supervision.csv。
    主要操作:
        1) 读取 xlsx, 解析底物名称, 通过清洗规则生成 substrate_tag（严格对齐 pt 文件命名）;
        2) 构造 complex_uid = "{pdb_id}_{substrate_tag}", 与实际 PT/JSON/FASTA 文件名严格一致;
        3) 在每个 (complex_uid, literature_ids) 分组内, 以 WT/native 行为基准计算:
           delta_log_kcatKm = log10(kcatKmValue_mut) - log10(kcatKmValue_WT);
        4) 输出精简 csv, 只保留后续监督训练需要的字段。
        5) 可选: 传入 --pt_dir 时，检查 complex_uid.pt 是否存在，输出缺失列表用于审计。

输入:
    - --xlsx: 原始 brenda_data.xlsx 路径
    - --out:  输出 csv 路径
    - --pt_dir: (可选) pt 文件目录，用于一致性审计

输出:
    - wet_supervision.csv:
        complex_uid, pdb_id, substrate, substrate_tag, substrate_smiles,
        enzyme_variant, kcatKmValue, log10_kcatKm, delta_log_kcatKm

调用示例:
    python src/val/prepare_data.py \
        --xlsx data/brenda/paper_data.xlsx \
        --out data/test/wet_supervision.csv \
        --pt_dir data/val/graphs
"""

import argparse
import os
import re
import numpy as np
import pandas as pd


def substrate_to_pt_tag(name: str) -> str:
    """
    严格遵守“pdb_id_substrate”命名规则的 substrate 清洗:
      - 不强制小写（保留原始大小写，如 3_Indol_3_yl_pyruvate）
      - 不删除 "acid"
      - 将所有非字母数字字符统一替换为 "_"
      - 连续 "_" 压缩为单个
      - 去掉首尾 "_"

    目标:
      与你的 pt 文件名一致，例如:
        "2-oxopentanoic acid"   -> "2_oxopentanoic_acid"
        "2-oxoglutarate"        -> "2_oxoglutarate"
        "3-(Indol-3-yl)pyruvate"-> "3_Indol_3_yl_pyruvate"
        "phenylglyoxylate"      -> "phenylglyoxylate"
    """
    if not isinstance(name, str):
        return "unknown"
    s = name.strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "unknown"


def mark_wt_like(row: pd.Series) -> bool:
    """
    判断一行是否可以视为 WT 基准:
        - enzyme_source == "native" (不区分大小写), 或
        - enzyme_variant 文本为 WT / wild type 等。
    """
    src = str(row.get("enzyme_source", "")).strip().lower()
    var = str(row.get("enzyme_variant", "")).strip().lower()
    if src == "native":
        return True
    if var in ("wt", "wildtype", "wild type", "wild-type"):
        return True
    return False


def compute_delta_log(group: pd.DataFrame) -> pd.DataFrame:
    """
    在一个 (complex_uid, literature_ids) 分组内计算 delta_log_kcatKm。
    若该组没有任何 WT/native 行, 则返回空 DataFrame, 表示丢弃该组。
    """
    group = group.copy()
    wt_rows = group[group["is_wt_like"]]
    if wt_rows.empty:
        return pd.DataFrame(columns=group.columns)

    wt_log = np.log10(wt_rows["kcatKmValue"].values.astype(float))
    wt_log_mean = float(wt_log.mean())

    group["log10_kcatKm"] = np.log10(group["kcatKmValue"].values.astype(float))
    group["delta_log_kcatKm"] = group["log10_kcatKm"] - wt_log_mean
    return group


def audit_pt_files(df_out: pd.DataFrame, pt_dir: str) -> None:
    """
    审计: 检查 df_out['complex_uid'] 对应的 pt 文件是否存在。
    """
    if not pt_dir:
        return
    pt_dir = os.path.abspath(pt_dir)
    if not os.path.isdir(pt_dir):
        print(f"[WARN] pt_dir 不是目录: {pt_dir}")
        return

    uids = sorted(set(df_out["complex_uid"].astype(str).tolist()))
    missing = []
    for uid in uids:
        pt_path = os.path.join(pt_dir, uid + ".pt")
        if not os.path.exists(pt_path):
            missing.append(uid)

    print(f"[AUDIT] PT 目录: {pt_dir}")
    print(f"[AUDIT] complex_uid 总数: {len(uids)}")
    print(f"[AUDIT] 缺失 pt 数: {len(missing)}")
    if missing:
        print("[AUDIT] 缺失示例(最多50个):")
        for x in missing[:50]:
            print("  -", x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, required=True, help="原始 brenda_data.xlsx 路径")
    parser.add_argument("--out", type=str, required=True, help="输出 wet_supervision.csv 路径")
    parser.add_argument("--pt_dir", type=str, default="", help="(可选) pt 文件目录，用于一致性审计")
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx, engine="openpyxl")

    required_cols = [
        "enzyme_name",
        "organism",
        "substrate",
        "substrate_smiles",
        "kcatKmValue",
        "enzyme_variant",
        "enzyme_source",
        "literature_ids",
        "uniprot_id",
        "pdb_id",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    df = df.copy()
    df["kcatKmValue"] = pd.to_numeric(df["kcatKmValue"], errors="coerce")
    df = df.dropna(subset=["kcatKmValue"])

    # 关键修改：substrate_tag 与 complex_uid 按 pt 文件命名规则生成
    df["substrate_tag"] = df["substrate"].apply(substrate_to_pt_tag)
    df["pdb_id"] = df["pdb_id"].astype(str).str.strip()
    df["complex_uid"] = df["pdb_id"] + "_" + df["substrate_tag"]

    df["is_wt_like"] = df.apply(mark_wt_like, axis=1)

    group_cols = ["complex_uid", "literature_ids"]
    df_out = (
        df.groupby(group_cols, group_keys=False)
        .apply(compute_delta_log)
        .reset_index(drop=True)
    )
    df_out = df_out.dropna(subset=["delta_log_kcatKm"])

    keep_cols = [
        "complex_uid",
        "pdb_id",
        "substrate",
        "substrate_tag",
        "substrate_smiles",
        "enzyme_variant",
        "kcatKmValue",
        "log10_kcatKm",
        "delta_log_kcatKm",
    ]
    df_out = df_out[keep_cols]

    df_out.to_csv(args.out, index=False)
    print(f"[OK] 已保存湿实验监督数据: {args.out}")
    print(f"[INFO] 样本数: {len(df_out)}")
    print(df_out.head())

    # 可选审计
    if args.pt_dir:
        audit_pt_files(df_out, args.pt_dir)


if __name__ == "__main__":
    main()
