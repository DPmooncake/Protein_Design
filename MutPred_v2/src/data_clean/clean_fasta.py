#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能:
  - 从一个多序列 FASTA 中进行基础清洗和去冗余:
    1) 去除含有未知/非法氨基酸残基的序列 (只保留 20 种标准氨基酸);
    2) 仅保留长度在 [min_len, max_len] 区间内的序列;
    3) 调用 MMseqs2 按序列相似度阈值 min_id 聚类, 仅导出每个簇的代表序列。
  - 不做结构域筛选, 不做 Fragment/partial 字段过滤, 只实现你现在讨论的三条规则。

输入:
  - 一个包含所有蛋白序列的 FASTA 文件 (--in)

输出 (都写到 --outdir 下):
  - clean.fasta: 经过“未知残基过滤 + 长度过滤”后的序列集合
  - rep.fasta:   用 MMseqs2 聚类后得到的代表序列 (即相似度 >= min_id 的簇只保留一个代表)
  - clusters.tsv: 每条序列所属的簇信息 (方便以后追踪)

调用示例:
  python src/data_clean/clean_fasta.py \
      --in data/fasta/TPP_family.fasta \
      --outdir data/fasta/TPP_family_mmseqs \
      --min_len 300 --max_len 700 \
      --min_id 0.8 --min_cov 0.8 \
      --threads 8

依赖:
  - Python 3
  - MMseqs2 (命令行可直接调用 "mmseqs")
"""

import os
import re
import argparse
import subprocess
import sys
from typing import Iterator, Tuple, List

# 只允许标准 20 种氨基酸, 其它视为“未知/非法”
LEGAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """
    逐条读取 FASTA, 返回 (header, seq)
    header 不含起始的 '>' 字符。
    """
    name = None
    seq_chunks: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_chunks)
                name = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if name is not None:
            yield name, "".join(seq_chunks)


def write_fasta(records: Iterator[Tuple[str, str]], path: str) -> None:
    """
    将 (header, seq) 序列写入 FASTA 文件。
    """
    with open(path, "w") as w:
        for h, s in records:
            w.write(f">{h}\n")
            for i in range(0, len(s), 80):
                w.write(s[i : i + 80] + "\n")


def clean_records(in_fa: str, min_len: int, max_len: int) -> Iterator[Tuple[str, str]]:
    """
    按三条规则清洗序列:
      1) 去除含非法氨基酸的序列;
      2) 仅保留长度在 [min_len, max_len] 的序列;
      3) 不在这里做序列去重, 去重交给 MMseqs2 聚类完成。
    """
    for header, seq in read_fasta(in_fa):
        # 1) 过滤非法氨基酸
        if any(ch not in LEGAL_AA for ch in seq):
            continue

        # 2) 过滤长度
        if not (min_len <= len(seq) <= max_len):
            continue

        # header 原样保留即可
        yield header, seq


def run(cmd):
    """
    简单封装 subprocess 调用, 打印命令方便检查。
    """
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="使用 MMseqs2 对 TPP 家族 FASTA 做基础清洗和去冗余的脚本"
    )
    parser.add_argument("--in", dest="in_fa", required=True, help="输入 FASTA 文件")
    parser.add_argument("--outdir", required=True, help="输出目录")
    parser.add_argument(
        "--min_len", type=int, default=300, help="保留序列的最小长度, 默认 300"
    )
    parser.add_argument(
        "--max_len", type=int, default=700, help="保留序列的最大长度, 默认 700"
    )
    parser.add_argument(
        "--min_id",
        type=float,
        default=0.9,
        help="MMseqs2 聚类时的最小序列相似度 (0~1), 默认 0.9",
    )
    parser.add_argument(
        "--min_cov",
        type=float,
        default=0.8,
        help="MMseqs2 聚类时的最小覆盖度 (0~1), 默认 0.8 (cov-mode=1)",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="MMseqs2 使用的线程数, 默认 8"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    clean_fa = os.path.join(args.outdir, "clean.fasta")
    rep_fa = os.path.join(args.outdir, "rep.fasta")
    clu_tsv = os.path.join(args.outdir, "clusters.tsv")
    tmpdir = os.path.join(args.outdir, "mmseqs_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    # 1) 基础清洗: 非法 AA + 长度
    cleaned = list(clean_records(args.in_fa, args.min_len, args.max_len))
    if not cleaned:
        raise SystemExit("清洗后没有剩余序列, 请检查长度阈值或原始数据。")
    write_fasta(cleaned, clean_fa)
    print(f"清洗后序列数: {len(cleaned)}", file=sys.stderr)

    # 2) MMseqs2 聚类去冗余
    db = os.path.join(args.outdir, "db")
    clu = os.path.join(args.outdir, "clu")
    repdb = os.path.join(args.outdir, "repdb")

    # 2.1 创建 MMseqs2 数据库
    run(["mmseqs", "createdb", clean_fa, db])

    # 2.2 聚类
    run(
        [
            "mmseqs",
            "cluster",
            db,
            clu,
            tmpdir,
            "--min-seq-id",
            str(args.min_id),
            "-c",
            str(args.min_cov),
            "--cov-mode",
            "2",
            "--cluster-mode",
            "0",
            "--threads",
            str(args.threads),
        ]
    )

    # 2.3 导出簇映射表
    run(["mmseqs", "createtsv", db, db, clu, clu_tsv])

    # 2.4 取代表序列
    run(["mmseqs", "createsubdb", clu, db, repdb])
    run(["mmseqs", "convert2fasta", repdb, rep_fa])

    print(
        f"\n完成:\n- 清洗后 FASTA: {clean_fa}\n- 代表序列: {rep_fa}\n- 聚类映射: {clu_tsv}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
