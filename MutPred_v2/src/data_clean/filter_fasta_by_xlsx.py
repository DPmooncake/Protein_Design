#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本作用:
    使用带有 Entry 列的 UniProt Excel 表格文件(.xlsx), 对多序列 FASTA 进行过滤:
      仅保留那些 Entry ID 出现在 Excel 中的序列, 其余序列全部删除。

输入:
    1) --fasta : 原始多序列 FASTA 文件路径
    2) --xlsx  : UniProt 元数据 Excel 文件路径(必须包含名为 "Entry" 的列)
    3) --out   : 过滤后输出的 FASTA 文件路径

输出:
    一个新的 FASTA 文件, 只包含 Excel 表中出现过的 Entry 对应的序列。

调用示例:
    python src/data_clean/filter_fasta_by_xlsx.py \
        --fasta data/fasta/TPP_family.fasta \
        --xlsx data/fasta/TPP_family_clean.xlsx \
        --out data/fasta/tpp_family_with_subunit.fasta

依赖:
    需要安装 pandas 和 openpyxl:
        pip install pandas openpyxl
"""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 UniProt Excel 表格(含 Entry 列)过滤 FASTA 序列"
    )
    parser.add_argument("--fasta", required=True, help="原始 FASTA 文件路径")
    parser.add_argument("--xlsx", required=True, help="UniProt 元数据 Excel 文件(.xlsx)")
    parser.add_argument("--out", required=True, help="输出过滤后 FASTA 文件路径")
    return parser.parse_args()


def extract_uid(header: str) -> str:
    """
    从 FASTA header 中提取 UniProt ID:
      - 若 header 形如 'sp|Q12345|...' 或 'tr|A0A0X1|...' 则取中间那段 accession;
      - 否则取第一个空格前的 token.
    """
    if "|" in header:
        parts = header.split("|")
        if len(parts) > 1:
            return parts[1]
        return header.split()[0]
    return header.split()[0]


def load_entry_whitelist_from_xlsx(path: str):
    """
    从 Excel 文件(.xlsx)中读取 Entry 列, 返回 Entry 集合。
    """
    df = pd.read_excel(path)
    # 去除列名两端空格, 防止 ' Entry ' 这种情况
    df.columns = [str(c).strip() for c in df.columns]
    if "Entry" not in df.columns:
        raise ValueError(f"Excel 文件中未找到 'Entry' 列, 实际列名为: {list(df.columns)}")

    entries = set(str(x).strip() for x in df["Entry"] if str(x).strip())
    return entries


def filter_fasta_by_entry(fasta_in: str, fasta_out: str, whitelist):
    """
    逐条读取 FASTA, 如果该序列的 UniProt ID 在 whitelist 中, 则写入输出文件。
    返回: (保留序列数量, 删除序列数量)
    """
    kept = 0
    removed = 0

    with open(fasta_in, "r", encoding="utf-8") as fin, open(
        fasta_out, "w", encoding="utf-8"
    ) as fout:
        header = None
        seq_lines = []

        def flush_record(h, seq_list):
            nonlocal kept, removed
            if h is None:
                return
            uid = extract_uid(h)
            if uid in whitelist:
                kept += 1
                fout.write(">" + h + "\n")
                for line in seq_list:
                    fout.write(line + "\n")
            else:
                removed += 1

        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush_record(header, seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())

        flush_record(header, seq_lines)

    return kept, removed


def main():
    args = parse_args()

    # 1. 从 xlsx 读取 Entry 白名单
    whitelist = load_entry_whitelist_from_xlsx(args.xlsx)
    print(f"从 Excel {args.xlsx} 读取到 Entry 数量: {len(whitelist)}")

    # 2. 按 Entry 过滤 FASTA
    kept, removed = filter_fasta_by_entry(args.fasta, args.out, whitelist)

    print("过滤完成:")
    print(f"  保留序列数: {kept}")
    print(f"  删除序列数: {removed}")
    print(f"  输出 FASTA: {args.out}")


if __name__ == "__main__":
    main()
