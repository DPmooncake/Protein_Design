# -*- coding: utf-8 -*-
# 作用:
#   将湿实验数据集(xlsx)转换为符合 Protenix 输入规范的 JSON。
#   支持批量输出：每一种“fasta + 底物(smiles)”组合生成一个 json 文件。
#
# 输入:
#   1) xlsx 文件，至少包含表头(允许大小写/空格差异):
#      ecNumber, enzyme_name, organism, substrate, substrate_smiles, kcatKmValue,
#      enzyme_variant, enzyme_source, literature_ids, uniprot_id, pdb_id, fasta, Subunit
#   2) 输出路径 --out:
#      - 若为目录：批量生成多个 json（每个组合一个文件）
#      - 若以 .json 结尾：生成单个合并 json（包含所有条目）
#
# 输出:
#   - 批量模式：out_dir/<pdb_id>_<substrate>.json，每个文件内容为 [ {name, sequences} ]
#   - 单文件模式：out.json，内容为 list[ {name, sequences} ]
#
# 调用格式:
#   1) 批量输出到目录:
#      python src/val/fasta2json.py --xlsx data/brenda/brenda_data.xlsx --out val/json
#   2) 输出单个合并文件:
#      python src/val/fasta2json.py --xlsx data/brenda/brenda_data.xlsx --out val/protenix_input.json
#   3) 指定 sheet:
#      python src/val/fasta2json.py --xlsx xxx.xlsx --out val/json --sheet Sheet1
#      python src/val/fasta2json.py --xlsx xxx.xlsx --out val/json --sheet all
#
# 规则:
#   - fasta 若包含 ';' 视为异源多聚体：大亚基在前，小亚基在后；
#     若 Subunit=2 代表 α2β2，则两条蛋白链 count 都取 2。
#   - 默认 TPP 与 MG 的 count 均等于 Subunit；
#   - 底物默认对接 1 个(count=1)；
#   - name = "<pdb_id>_<substrate>" (substrate 会做轻量清洗以适配文件名)；
#   - “每一种 fasta + 底物”组合只生成一个 json：默认去重保留第一条。

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

TPP_SMILES = "n1c(C)nc(N)c(c1)CN1CSC(=C1C)CCO[P@](=O)([O-])OP(=O)([O-])[O-]"


def _norm_colname(c: str) -> str:
    """统一列名：去空白并保持原样大小写不重要时可统一小写。"""
    return str(c).strip()


def _sanitize_for_name(text: str) -> str:
    """
    将 substrate 等字符串清洗为适合做文件名/条目名的片段：
    - 首尾去空白
    - 非字母数字替换为 '_'
    - 连续 '_' 压缩
    """
    s = (text or "").strip()
    s = re.sub(r"[^\w]+", "_", s)  # \w: [A-Za-z0-9_]
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "NA"


def _parse_fasta_sequences(fasta_cell: Any) -> List[str]:
    """
    解析 fasta 列:
    - 若包含 ';' => 异源多聚体，返回 [alpha_seq, beta_seq]
    - 否则返回 [seq]
    """
    if fasta_cell is None:
        return []
    s = str(fasta_cell).strip()
    if not s or s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split(";")]
    parts = [p for p in parts if p]
    return parts


def _to_int(v: Any, default: int = 1) -> int:
    try:
        x = int(v)
        return x if x > 0 else default
    except Exception:
        return default


def row_to_protenix_item(row: pd.Series) -> Optional[Dict[str, Any]]:
    pdb_id = str(row.get("pdb_id", "")).strip()
    substrate = str(row.get("substrate", "")).strip()
    substrate_smiles = str(row.get("substrate_smiles", "")).strip()
    fasta_cell = row.get("fasta", None)
    subunit = _to_int(row.get("Subunit", 1), default=1)

    if not pdb_id or pdb_id.lower() == "nan":
        return None
    if not substrate or substrate.lower() == "nan":
        substrate = "NA"
    if not substrate_smiles or substrate_smiles.lower() == "nan":
        return None

    seqs = _parse_fasta_sequences(fasta_cell)
    if len(seqs) == 0:
        return None
    if len(seqs) > 2:
        # 当前需求只描述最多两条链(异源二聚/四聚等)。>2 不做猜测。
        return None

    name = f"{pdb_id}_{_sanitize_for_name(substrate)}"

    sequences_block: List[Dict[str, Any]] = []

    # proteinChain
    if len(seqs) == 1:
        sequences_block.append({
            "proteinChain": {
                "sequence": seqs[0],
                "count": subunit
            }
        })
    else:
        # 异源多聚体：两条链 count 都取 Subunit (如 α2β2 => 2 和 2)
        sequences_block.append({
            "proteinChain": {
                "sequence": seqs[0],  # 大亚基
                "count": subunit
            }
        })
        sequences_block.append({
            "proteinChain": {
                "sequence": seqs[1],  # 小亚基
                "count": subunit
            }
        })

    # 默认 TPP 与 MG 数量 = Subunit
    sequences_block.append({
        "ligand": {
            "ligand": TPP_SMILES,
            "count": subunit
        }
    })
    sequences_block.append({
        "ion": {
            "ion": "MG",
            "count": subunit
        }
    })

    # 底物默认 1 个
    sequences_block.append({
        "ligand": {
            "ligand": substrate_smiles,
            "count": 1
        }
    })

    return {
        "name": name,
        "sequences": sequences_block
    }


def read_xlsx_as_df(xlsx_path: str, sheet: Optional[str]) -> pd.DataFrame:
    """
    读取 xlsx：
    - sheet is None: 读第一个 sheet（避免 pandas 返回 dict）
    - sheet == 'all': 读所有 sheet 并合并
    - 其他: 按名称/索引读取
    """
    if sheet is None:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
        return df

    sheet_str = str(sheet).strip()
    if sheet_str.lower() == "all":
        dfs = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
        # dfs: Dict[str, DataFrame]
        merged = pd.concat(list(dfs.values()), ignore_index=True)
        return merged

    # 允许 sheet 为数字索引字符串
    try:
        sheet_idx = int(sheet_str)
        df = pd.read_excel(xlsx_path, sheet_name=sheet_idx, engine="openpyxl")
        return df
    except Exception:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_str, engine="openpyxl")
        return df


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_colname(c) for c in df.columns]

    required_cols = {"pdb_id", "substrate", "substrate_smiles", "fasta", "Subunit"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")
    return df


def build_key_for_batch(row: pd.Series) -> Tuple[str, str, str, int, str]:
    """
    批量输出时用于去重与命名的 key。

    你说“每一种 fasta + 底物组合生成一种 json”，
    但为了避免不同 pdb/substrate 同时同 smiles 导致覆盖，这里把:
      (pdb_id, substrate, fasta, Subunit, substrate_smiles)
    都纳入 key，保证文件名与条目一致且不互相覆盖。
    """
    pdb_id = str(row.get("pdb_id", "")).strip()
    substrate = str(row.get("substrate", "")).strip()
    fasta = str(row.get("fasta", "")).strip()
    subunit = _to_int(row.get("Subunit", 1), default=1)
    substrate_smiles = str(row.get("substrate_smiles", "")).strip()
    return (pdb_id, substrate, fasta, subunit, substrate_smiles)


def convert_xlsx_to_protenix_json(
    xlsx_path: str,
    out_path: str,
    sheet: Optional[str] = None,
    drop_na: bool = True,
) -> None:
    df = read_xlsx_as_df(xlsx_path, sheet=sheet)
    df = ensure_required_columns(df)

    is_dir_out = (not out_path.lower().endswith(".json"))
    if is_dir_out:
        os.makedirs(out_path, exist_ok=True)

        # 去重：按 fasta+底物组合（这里用更稳健的 key，避免覆盖）
        df["_key"] = df.apply(build_key_for_batch, axis=1)
        df_unique = df.drop_duplicates(subset=["_key"], keep="first").reset_index(drop=True)

        made, skipped = 0, 0
        for _, row in df_unique.iterrows():
            item = row_to_protenix_item(row)
            if item is None:
                if drop_na:
                    skipped += 1
                    continue
                skipped += 1
                continue

            # 文件名：与 name 一致
            file_name = f"{item['name']}.json"
            file_path = os.path.join(out_path, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([item], f, ensure_ascii=False, indent=2)

            made += 1

        print(f"[OK] 批量输出目录: {out_path}")
        print(f"[INFO] 生成文件数: {made}; 跳过条目数: {skipped}; 去重后条目数: {len(df_unique)}")

    else:
        # 单文件：所有条目合并输出
        items: List[Dict[str, Any]] = []
        skipped = 0
        for _, row in df.iterrows():
            item = row_to_protenix_item(row)
            if item is None:
                if drop_na:
                    skipped += 1
                    continue
                skipped += 1
                continue
            items.append(item)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        print(f"[OK] 输出: {out_path}")
        print(f"[INFO] 生成条目数: {len(items)}; 跳过行数: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="输入 xlsx 路径")
    parser.add_argument("--out", required=True, help="输出 json 文件路径(.json) 或 输出目录(批量)")
    parser.add_argument("--sheet", default=None, help="sheet 名称/索引；'all' 表示全部；默认第一个 sheet")
    parser.add_argument("--drop_na", action="store_true", help="丢弃关键字段缺失的行(默认开启)")
    args = parser.parse_args()

    convert_xlsx_to_protenix_json(
        xlsx_path=args.xlsx,
        out_path=args.out,
        sheet=args.sheet,
        drop_na=True
    )


if __name__ == "__main__":
    main()
