# -*- coding: utf-8 -*-
"""
脚本作用:
    从 EC_number.xlsx 中读取 “EC number” 列,
    将其中出现的所有 EC 号拆分、去重后导出为一个新的 Excel 文件,
    每个唯一的 EC 号占一行。

输入:
    --in_xlsx   EC_number.xlsx 文件路径, 必须包含列名 "EC number"
                每个单元格中可以是一个或多个 EC 号, 用分号 ';' 分隔, 允许有空格
输出:
    --out_xlsx  输出的唯一 EC 号列表, 每行一个 EC 号, 列名为 "EC_number"

调用示例:
    python src/data_clean/extract_unique_ec.py \
        --in_xlsx data/fasta/EC_number.xlsx \
        --out_xlsx data/fasta/unique_EC_numbers.xlsx
"""

import argparse
from typing import Set, List

import pandas as pd


def extract_unique_ec(in_xlsx: str, out_xlsx: str) -> None:
    # 读取原始表
    df = pd.read_excel(in_xlsx)

    if "EC number" not in df.columns:
        raise ValueError("输入文件中未找到 'EC number' 列, 请检查列名是否正确")

    unique_ec: Set[str] = set()

    for val in df["EC number"]:
        if not isinstance(val, str):
            # 非字符串 (NaN 等) 直接跳过
            continue

        # 按分号拆分, 去掉前后空格
        parts: List[str] = [p.strip() for p in val.split(";")]
        for ec in parts:
            if ec:
                unique_ec.add(ec)

    # 转成 DataFrame, 排序后导出
    ec_list = sorted(unique_ec)
    out_df = pd.DataFrame({"EC_number": ec_list})
    out_df.to_excel(out_xlsx, index=False)

    print(f"共提取到 {len(ec_list)} 个不重复的 EC number, 已写入: {out_xlsx}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_xlsx", required=True, help="包含 'EC number' 列的输入 Excel 文件路径")
    parser.add_argument("--out_xlsx", required=True, help="输出唯一 EC 号列表的 Excel 文件路径")
    args = parser.parse_args()

    extract_unique_ec(args.in_xlsx, args.out_xlsx)


if __name__ == "__main__":
    main()
