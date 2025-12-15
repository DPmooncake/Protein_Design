# 功能：使用 BRENDA SOAP API 批量获取指定 EC 号的 Kcat/KM 数据，并通过 Cofactor 表筛选出含 thiamine diphosphate
#      的 TPP 依赖酶，同时提取 Protein Variants(Engineering) 中的突变信息，整理为一个 CSV 文件。
# 输入：
#   1) BRENDA 账号邮箱 (--email)
#   2) BRENDA 账号明文密码 (--password)，脚本内部会进行 SHA-256 加密
#   3) TPP 家族 EC 号文件 (--ec-xlsx)，为 xlsx 格式，至少包含一列 EC 号
#   4) EC 号所在列名 (--ec-column)，默认 "EC_number"
#   5) 输出 CSV 路径 (--output-csv)
# 可选：
#   6) 是否跳过 Cofactor 检查（--skip-cofactor-filter），默认 False，即强制要求 Cofactor 中包含 thiamine diphosphate
# 输出：
#   一个 CSV 文件，每行对应一个 Kcat/KM 记录，包含字段：
#   ecNumber, organism, substrate, kcatKmValue, kcatKmValueMaximum, commentary_kcatKm,
#   ligandStructureId, literature_ids, mutations_description, has_tpp_cofactor, enzyme_synonyms
# 依赖：
#   pip install zeep pandas openpyxl
# 调用示例：
"""
   python src/brenda_data/extract.py \
       --email "2817919375@qq.com" \
       --password "wr28170506" \
       --ec-xlsx data/fasta/unique_EC_numbers.xlsx \
       --ec-column EC_number \
       --output-csv data/brenda/brenda_data.csv
"""

import argparse
import hashlib
import time
from typing import List, Dict, Any

import pandas as pd
from zeep import Client, Settings, exceptions as zeep_exceptions
from zeep.helpers import serialize_object


BRENDA_WSDL = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"


def init_brenda_client() -> Client:
    """初始化 zeep 客户端。"""
    settings = Settings(strict=False, xml_huge_tree=True)
    client = Client(BRENDA_WSDL, settings=settings)
    return client


def sha256_hex(password: str) -> str:
    """对 BRENDA 明文密码进行 SHA-256 加密。"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def call_brenda_method(client: Client, method_name: str, parameters: tuple) -> Any:
    """
    通用的 BRENDA SOAP 调用封装，带简单异常处理和重试。
    返回值为 zeep 的对象或列表，后续用 serialize_object 转换。
    """
    method = getattr(client.service, method_name)
    max_retry = 3
    for trial in range(max_retry):
        try:
            result = method(*parameters)
            return result
        except (
            zeep_exceptions.TransportError,
            zeep_exceptions.XMLSyntaxError,
            zeep_exceptions.ValidationError,
        ) as e:
            print(f"[WARN] {method_name} 调用失败，第 {trial + 1} 次重试，错误：{e}")
            time.sleep(2.0)
    print(f"[ERROR] {method_name} 多次重试仍失败，跳过本次请求。")
    return None


def get_cofactor_entries(client: Client, email: str, password_hex: str, ec: str) -> List[Dict[str, Any]]:
    """获取某个 EC 的 Cofactor 记录。"""
    params = (
        email,
        password_hex,
        f"ecNumber*{ec}",
        "cofactor*",
        "commentary*",
        "organism*",
        "ligandStructureId*",
        "literature*",
    )
    result = call_brenda_method(client, "getCofactor", params)
    if result is None:
        return []
    data = serialize_object(result)
    if isinstance(data, dict):
        data = [data]
    return data or []


def ec_has_tpp(cofactor_entries: List[Dict[str, Any]]) -> bool:
    """判断 Cofactor 记录中是否包含 thiamine diphosphate。"""
    keywords = [
        "thiamine diphosphate",
        "thiamin diphosphate",
        "thiamine-diphosphate",
        "thiamin-diphosphate",
        "thiamine pyrophosphate",
        "thiamin pyrophosphate",
    ]
    for entry in cofactor_entries:
        cof = (entry.get("cofactor") or "").lower()
        for kw in keywords:
            if kw in cof:
                return True
    return False


def get_kcatkm_entries(client: Client, email: str, password_hex: str, ec: str) -> List[Dict[str, Any]]:
    """获取某个 EC 的 Kcat/KM 记录。"""
    params = (
        email,
        password_hex,
        f"ecNumber*{ec}",
        "kcatKmValue*",
        "kcatKmValueMaximum*",
        "substrate*",
        "commentary*",
        "organism*",
        "ligandStructureId*",
        "literature*",
    )
    result = call_brenda_method(client, "getKcatKmValue", params)
    if result is None:
        return []
    data = serialize_object(result)
    if isinstance(data, dict):
        data = [data]
    return data or []


def get_engineering_entries(client: Client, email: str, password_hex: str, ec: str) -> List[Dict[str, Any]]:
    """获取某个 EC 的 Engineering（蛋白突变）记录。当前版本未导出，但可保留接口供扩展。"""
    params = (
        email,
        password_hex,
        f"ecNumber*{ec}",
        "engineering*",
        "commentary*",
        "organism*",
        "literature*",
    )
    result = call_brenda_method(client, "getEngineering", params)
    if result is None:
        return []
    data = serialize_object(result)
    if isinstance(data, dict):
        data = [data]
    return data or []


def build_organism_to_mutations(engineering_entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    将 Engineering 记录按 organism 聚合。
    返回 {organism: [mutation_desc1, mutation_desc2, ...]}。
    当前版本不写入 CSV，仅保留数据结构供后续使用。
    """
    mapping: Dict[str, List[str]] = {}
    for entry in engineering_entries:
        org = entry.get("organism") or ""
        eng = entry.get("engineering") or ""
        if not eng:
            continue
        mapping.setdefault(org, []).append(eng)
    return mapping


def get_sequence_entries(client: Client, email: str, password_hex: str, ec: str) -> List[Dict[str, Any]]:
    """获取某个 EC 的 Sequence 记录，用于提取 UniProt accession（firstAccessionCode）。"""
    params = (
        email,
        password_hex,
        f"ecNumber*{ec}",
        "sequence*",
        "noOfAminoAcids*",
        "firstAccessionCode*",
        "source*",
        "id*",
        "organism*",
    )
    result = call_brenda_method(client, "getSequence", params)
    if result is None:
        return []
    data = serialize_object(result)
    if isinstance(data, dict):
        data = [data]
    return data or []


def build_organism_to_uniprot(sequence_entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    将 Sequence 记录按 organism 聚合出 UniProt ID（firstAccessionCode）。
    返回 {organism: accession}。
    规则：
        1) 每个 organism 只保留一个 accession（取去重后的第一个），避免变成一长串；
        2) 不在这里跨物种合并，避免“全 EC 的所有 UniProt 全挤进一格”的情况。
    """
    mapping: Dict[str, str] = {}
    tmp: Dict[str, List[str]] = {}
    for entry in sequence_entries:
        org = entry.get("organism") or ""
        acc = entry.get("firstAccessionCode") or ""
        if not org or not acc:
            continue
        tmp.setdefault(org, []).append(str(acc).strip())
    for org, accs in tmp.items():
        uniq = sorted(set(accs))
        if uniq:
            mapping[org] = uniq[0]  # 只取第一个 accession，和网页显示一致
    return mapping


def load_ec_list_from_excel(path: str, ec_column: str) -> List[str]:
    """从 Excel 文件读取 EC 号列表，并去重。"""
    df = pd.read_excel(path)
    if ec_column not in df.columns:
        raise ValueError(f"在文件 {path} 中未找到列名 '{ec_column}'")
    ecs = (
        df[ec_column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    ecs = ecs[ecs != ""]
    return sorted(set(ecs.tolist()))


def sanitize_commentary(text: str) -> str:
    """
    将 commentary 中的特殊符号做成更通用的 ASCII 表达，避免在某些中文 Excel 编码下显示为问号。
    例如：'55°C' -> '55 degC'。
    """
    if not text:
        return text
    text = text.replace("°C", " degC").replace("Â°C", " degC")
    text = text.replace("°", " deg")
    return text


def main():
    """
    命令行参数示例：
        python extract.py \
            --email "your_email@example.com" \
            --password "YourPassword" \
            --ec-xlsx data/TPP_family.xlsx \
            --ec-column EC_number \
            --output-csv data/brenda_tpp_kcatkm.csv
    """
    parser = argparse.ArgumentParser(description="从 BRENDA 批量下载 TPP 依赖酶家族的 Kcat/KM 与 UniProt 数据")
    parser.add_argument("--email", required=True, help="BRENDA 注册邮箱")
    parser.add_argument("--password", required=True, help="BRENDA 明文密码，将在脚本内部做 SHA-256 加密")
    parser.add_argument("--ec-xlsx", required=True, help="包含 TPP 家族 EC 号的 xlsx 文件路径")
    parser.add_argument("--ec-column", default="EC_number", help="EC 号所在列名，默认 EC_number")
    parser.add_argument("--output-csv", required=True, help="输出结果 CSV 路径")
    parser.add_argument(
        "--skip-cofactor-filter",
        action="store_true",
        help="如果指定该参数，则不通过 Cofactor 表检查 thiamine diphosphate，直接用提供的 EC 列表",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="两次 EC 请求之间的睡眠时间（秒），默认 0.5，防止高频访问报错",
    )
    args = parser.parse_args()

    email = args.email
    password_hex = sha256_hex(args.password)

    print("初始化 BRENDA SOAP 客户端...")
    client = init_brenda_client()

    print(f"从 {args.ec_xlsx} 读取 EC 列表...")
    ec_list = load_ec_list_from_excel(args.ec_xlsx, args.ec_column)
    print(f"共读取到 {len(ec_list)} 个 EC 号。")

    rows: List[Dict[str, Any]] = []

    for idx, ec in enumerate(ec_list, start=1):
        print(f"\n[{idx}/{len(ec_list)}] 处理 EC {ec}")

        has_tpp = True
        if not args.skip_cofactor_filter:
            cofactor_entries = get_cofactor_entries(client, email, password_hex, ec)
            has_tpp = ec_has_tpp(cofactor_entries)
            print(f"  Cofactor 记录数：{len(cofactor_entries)}，是否含 TPP：{has_tpp}")
            if not has_tpp:
                print("  不含 thiamine diphosphate，跳过该 EC。")
                time.sleep(args.sleep)
                continue

        engineering_entries = get_engineering_entries(client, email, password_hex, ec)
        print(f"  Engineering 记录数：{len(engineering_entries)}")
        org_to_mut = build_organism_to_mutations(engineering_entries)

        sequence_entries = get_sequence_entries(client, email, password_hex, ec)
        print(f"  Sequence 记录数：{len(sequence_entries)}")
        org_to_uniprot = build_organism_to_uniprot(sequence_entries)

        kcatkm_entries = get_kcatkm_entries(client, email, password_hex, ec)
        print(f"  Kcat/KM 记录数：{len(kcatkm_entries)}")

        for entry in kcatkm_entries:
            org = entry.get("organism") or ""

            # 精确按 organism 匹配 UniProt；若找不到则置空，不再跨物种拼接
            if org and org in org_to_uniprot:
                uniprot_id = org_to_uniprot[org]
            else:
                uniprot_id = ""

            lit_ids = entry.get("literature") or []
            if isinstance(lit_ids, (list, tuple, set)):
                lit_str = ";".join(str(x) for x in lit_ids)
            else:
                lit_str = str(lit_ids) if lit_ids is not None else ""

            commentary_raw = entry.get("commentary", "")
            commentary_clean = sanitize_commentary(commentary_raw)

            row = {
                "ecNumber": entry.get("ecNumber", ec),
                "substrate": entry.get("substrate", ""),
                "kcatKmValue": entry.get("kcatKmValue", ""),
                "commentary_kcatKm": commentary_clean,
                "literature_ids": lit_str,
                "uniprot_id": uniprot_id,
            }
            rows.append(row)

        time.sleep(args.sleep)

    if not rows:
        print("未获得任何 Kcat/KM 记录，脚本结束。")
        return

    df = pd.DataFrame(rows)

    # 按 EC 号和文献 ID 排序，让同一篇论文的数据自然聚到一起
    df = df.sort_values(by=["ecNumber", "literature_ids", "substrate", "kcatKmValue"])

    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\n完成，共写出 {len(df)} 条 Kcat/KM 记录到 {args.output_csv}")


if __name__ == "__main__":
    main()