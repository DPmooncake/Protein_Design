# -*- coding: utf-8 -*-
# 作用:
#   读取 xlsx，按命名规则 name = pdb_id_substrate(清洗后)；
#   将所有序列合并输出到一个 FASTA 文件；
#   若 fasta 中包含 ';'（异源多聚体），只保留大亚基(分号前)并忽略小亚基；
#   FASTA 序列单行输出，不换行（适配 ESM-2）。
#
# 输入:
#   1) xlsx 文件，至少包含列:
#      pdb_id, substrate, fasta, Subunit, substrate_smiles
#
# 输出:
#   1) 单个 FASTA 文件 out_fasta，格式:
#      >pdbid_substrate
#      SEQUENCE
#
# 调用格式:
#   python src/val/coll_fasta.py --xlsx data/brenda/paper_data.xlsx --out data/test/paper_data.fasta

import argparse
import re
from typing import Any, Optional, Tuple

import pandas as pd


def _sanitize_for_name(text: str) -> str:
    """
    清洗 substrate，用于生成 pdb_id_substrate
    """
    s = (text or "").strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "NA"


def _parse_fasta_big_subunit_only(fasta_cell: Any) -> Optional[str]:
    """
    返回大亚基序列:
      - 若包含 ';'，只取分号前
      - 否则返回整条序列
    """
    if fasta_cell is None:
        return None
    s = str(fasta_cell).strip()
    if not s or s.lower() == "nan":
        return None
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        return parts[0] if parts else None
    return s


def read_xlsx_as_df(xlsx_path: str, sheet: Optional[str]) -> pd.DataFrame:
    if sheet is None:
        return pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    if str(sheet).lower() == "all":
        dfs = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
        return pd.concat(list(dfs.values()), ignore_index=True)
    return pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")


def convert_xlsx_to_one_fasta(
    xlsx_path: str,
    out_fasta: str,
    sheet: Optional[str] = None,
) -> None:
    df = read_xlsx_as_df(xlsx_path, sheet)

    required = {"pdb_id", "substrate", "fasta", "Subunit", "substrate_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")

    # 与 json / pt 完全一致的组合去重 key
    df["_key"] = df.apply(
        lambda r: (
            str(r["pdb_id"]).strip(),
            str(r["substrate"]).strip(),
            str(r["fasta"]).strip(),
            str(r["substrate_smiles"]).strip(),
        ),
        axis=1,
    )
    df = df.drop_duplicates("_key").reset_index(drop=True)

    written, skipped = 0, 0
    with open(out_fasta, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pdb_id = str(row["pdb_id"]).strip()
            substrate = str(row["substrate"]).strip()

            seq = _parse_fasta_big_subunit_only(row["fasta"])
            if not pdb_id or not seq:
                skipped += 1
                continue

            name = f"{pdb_id}_{_sanitize_for_name(substrate)}"
            seq = re.sub(r"\s+", "", seq)

            f.write(f">{name}\n")
            f.write(seq + "\n")
            written += 1

    print(f"[OK] FASTA 输出完成: {out_fasta}")
    print(f"[INFO] 写入序列数: {written}, 跳过: {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()

    convert_xlsx_to_one_fasta(
        xlsx_path=args.xlsx,
        out_fasta=args.out,
        sheet=args.sheet,
    )


if __name__ == "__main__":
    main()
