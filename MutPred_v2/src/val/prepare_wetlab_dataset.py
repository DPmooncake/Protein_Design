# -*- coding: utf-8 -*-
"""
prepare_wetlab_dataset.py

作用:
    读取湿实验数据表 (例如来自 BRENDA/文献的 2JCL 数据),
    以 kcatKmValue 为指标, 为每个突变体计算相对于 WT 的 fold-change 和对数 fold-change,
    并解析 enzyme_variant 字段得到突变位点与氨基酸信息, 输出整理后的 CSV,
    方便后续与 MutPred_v2 的预测分数做 Spearman / AUROC 等评估。

输入:
    一个 CSV 文件, 至少包含以下列:
        ecNumber, enzyme_name, organism, substrate, substrate_smiles,
        kcatKmValue, enzyme_variant, enzyme_source, literature_ids,
        uniprot_id, pdb_id, pdb_path

输出:
    一个新的 CSV 文件, 在原有列基础上新增:
        is_wt:           bool, 是否 WT
        n_mut:          int, 突变位点个数 (WT 为 0)
        mut_positions:  str, 逗号分隔的突变位点数字, 如 "55" 或 "32,55"
        mut_from_aas:   str, 逗号分隔的 WT 氨基酸, 如 "S,E"
        mut_to_aas:     str, 逗号分隔的 突变后氨基酸, 如 "A,D"
        ref_kcatKm:     float, 对应 pdb_id 组内 WT 的 kcatKmValue (若有多个 WT 取平均)
        fold_change:    float, kcatKmValue / ref_kcatKm
        log_fold_change_ln:   float, ln(fold_change)
        log_fold_change_log10:float, log10(fold_change)
        label_3class:   str, "beneficial"/"neutral"/"deleterious"
        label_binary:   int, 1 表示 beneficial, 0 表示非 beneficial (neutral+deleterious)

调用格式示例:
    python src/val/prepare_wetlab_dataset.py \
        --input data/val/2JLC.csv \
        --output data/val/2JLC_processed.csv
"""

import argparse
import math
from typing import List, Tuple

import numpy as np
import pandas as pd


WT_TOKENS = {"WT", "WILD-TYPE", "WILDTYPE", "WILD TYPE", "NATIVE"}


def parse_enzyme_variant(variant: str) -> Tuple[bool, List[int], List[str], List[str]]:
    """
    解析 enzyme_variant 字段.

    约定:
        - WT 行: "WT", "wild-type", "native" 等 -> 视为无突变
        - 单/多点突变: 类似 "E55D", "S32A,E55D", "S32A/E55D", "S32A E55D" 等
          只考虑最常见的格式: 单字母氨基酸 + 数字位置 + 单字母氨基酸.

    返回:
        is_wt:        bool, 是否 WT
        positions:    List[int], 突变位点列表
        wt_aas:       List[str], WT 氨基酸列表
        mut_aas:      List[str], 突变后氨基酸列表
    """
    if not isinstance(variant, str):
        return False, [], [], []

    raw = variant.strip()
    if not raw:
        return False, [], [], []

    upper = raw.upper()
    if upper in WT_TOKENS:
        return True, [], [], []

    # 替换常见分隔符为逗号
    for sep in ["/", " ", ";", "+"]:
        upper = upper.replace(sep, ",")
    parts = [p for p in upper.split(",") if p]

    positions: List[int] = []
    wt_aas: List[str] = []
    mut_aas: List[str] = []

    for p in parts:
        # 期望形如 "E55D"
        if len(p) < 3:
            # 太短的不认, 跳过
            continue
        wt = p[0]
        mut = p[-1]
        pos_str = p[1:-1]
        if not pos_str.isdigit():
            # 如果不是纯数字, 也跳过
            continue
        pos = int(pos_str)
        positions.append(pos)
        wt_aas.append(wt)
        mut_aas.append(mut)

    if len(positions) == 0:
        # 无法解析, 视为非 WT 但无突变信息
        return False, [], [], []

    return False, positions, wt_aas, mut_aas


def label_from_fold_change(
    fold: float,
    th_beneficial: float = 1.2,
    th_deleterious: float = 0.8,
) -> Tuple[str, int]:
    """
    根据 fold-change 打标签:
        fold >= th_beneficial  -> beneficial,  label_binary = 1
        fold <= th_deleterious -> deleterious, label_binary = 0
        其它                   -> neutral,     label_binary = 0
    """
    if math.isnan(fold) or fold <= 0:
        return "unknown", 0

    if fold >= th_beneficial:
        return "beneficial", 1
    if fold <= th_deleterious:
        return "deleterious", 0
    return "neutral", 0


def process_wetlab_csv(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    # 解析变体
    is_wt_list = []
    n_mut_list = []
    pos_str_list = []
    wt_str_list = []
    mut_str_list = []

    for v in df["enzyme_variant"]:
        is_wt, positions, wt_aas, mut_aas = parse_enzyme_variant(v)
        is_wt_list.append(is_wt)
        n_mut_list.append(len(positions))
        pos_str_list.append(",".join(str(p) for p in positions) if positions else "")
        wt_str_list.append(",".join(wt_aas) if wt_aas else "")
        mut_str_list.append(",".join(mut_aas) if mut_aas else "")

    df["is_wt"] = is_wt_list
    df["n_mut"] = n_mut_list
    df["mut_positions"] = pos_str_list
    df["mut_from_aas"] = wt_str_list
    df["mut_to_aas"] = mut_str_list

    # 以 pdb_id 分组, 计算每组 WT 的参考 kcatKm
    ref_values = []
    for pdb_id, group in df.groupby("pdb_id"):
        wt_rows = group[group["is_wt"]]

        if len(wt_rows) == 0:
            # 若没有 WT, 给该组全体 ref_kcatKm 设 NaN
            ref = np.nan
        else:
            # 若有多个 WT, 取平均
            ref = wt_rows["kcatKmValue"].astype(float).mean()

        ref_values.extend([ref] * len(group))

    # 上面分组是按 group 顺序生成的, 需要对齐到原 df 的 index
    # 更稳妥的写法是 merge, 这里改成 merge 版本:
    df["kcatKmValue"] = df["kcatKmValue"].astype(float)

    ref_df = (
        df[df["is_wt"]]
        .groupby("pdb_id", as_index=False)["kcatKmValue"]
        .mean()
        .rename(columns={"kcatKmValue": "ref_kcatKm"})
    )
    df = df.merge(ref_df, on="pdb_id", how="left")

    # 计算 fold-change 与 log fold-change
    def safe_div(row):
        val = row["kcatKmValue"]
        ref = row["ref_kcatKm"]
        if pd.isna(ref) or ref <= 0:
            return np.nan
        return float(val) / float(ref)

    df["fold_change"] = df.apply(safe_div, axis=1)

    def safe_log(x: float, base: float = math.e) -> float:
        if isinstance(x, (float, int)) and x > 0:
            return math.log(x, base)
        return float("nan")

    df["log_fold_change_ln"] = df["fold_change"].apply(lambda x: safe_log(x, math.e))
    df["log_fold_change_log10"] = df["fold_change"].apply(lambda x: safe_log(x, 10.0))

    # 打标签
    labels_3class = []
    labels_binary = []
    for x in df["fold_change"]:
        label3, label_bin = label_from_fold_change(x)
        labels_3class.append(label3)
        labels_binary.append(label_bin)

    df["label_3class"] = labels_3class
    df["label_binary"] = labels_binary

    # 保存
    df.to_csv(output_path, index=False)
    print(f"处理完成, 共 {len(df)} 条记录.")
    print(f"输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="整理湿实验 kcat/Km 数据, 生成 fold-change 和标签")
    parser.add_argument("--input", type=str, required=True, help="原始 CSV 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 CSV 文件路径")
    args = parser.parse_args()

    process_wetlab_csv(args.input, args.output)


if __name__ == "__main__":
    main()
