# -*- coding: utf-8 -*-
"""
作用:
    使用 ESM-2 从一个包含多条蛋白序列的 FASTA 文件中提取 embedding。
    对于每条序列, 输出一个 {seq_id}.pt 文件, 内含:
        - seq_id: 序列 ID
        - seq:   氨基酸序列字符串
        - residue_emb:  残基级别 embedding, 形状 [L, D]
        - seq_emb:      序列级别 embedding, 形状 [D] (对残基 embedding 求平均)

输入:
    1) --fasta:  多序列 FASTA 文件路径 (例如 data/fasta/rep.fasta)；
    2) --out-dir: 输出目录 (例如 results/esm2_t36_3B_features)；
    3) --model-name: 使用的 ESM-2 预训练模型名称 (例如 esm2_t36_3B_UR50D)；
    4) --repr-layer: 取第几层的表示 (对 esm2_t36_3B_UR50D 通常用 36)；
    5) --batch-size: batch 大小 (默认 4, 根据显存调整)。

输出:
    在 out-dir 下生成:
        - 每条序列一个 {seq_id}.pt 文件 (torch.save 的 dict, 见上);
        - 一个 index.tsv, 记录 seq_id、长度、embedding 文件名。

调用格式示例:
python src/embedding/extract_esm2_embedding.py \
    --fasta data/fasta/TPP_family_mmseqs/rep.fasta \
    --out-dir results/sequence_features \
    --model-name esm2_t36_3B_UR50D \
    --repr-layer 36 \
    --batch-size 4

python src/embedding/extract_esm2_embedding.py \
    --fasta data/test/paper_data.fasta \
    --out-dir data/test/sequence_features \
    --model-name esm2_t36_3B_UR50D \
    --repr-layer 36 \
    --batch-size 4
"""

import os
import argparse
from typing import List, Tuple

import torch
from tqdm import tqdm
import esm


def parse_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    读取多序列 FASTA, 返回列表 [(seq_id, seq), ...]。

    规则:
        - 每条记录以 '>' 开头的行为 header;
        - seq_id 取 header 第一段 (空格前);
        - 若该段形如 'sp|P33287|PDC_NEUCR', 则取中间 'P33287' 作为 seq_id。
    """
    records: List[Tuple[str, str]] = []
    with open(fasta_path, "r") as f:
        seq_id = None
        seq_chunks = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None and seq_chunks:
                    records.append((seq_id, "".join(seq_chunks)))
                    seq_chunks = []
                header = line[1:]
                first_token = header.split()[0]
                if "|" in first_token:
                    parts = first_token.split("|")
                    if len(parts) >= 2 and parts[1]:
                        seq_id = parts[1]
                    else:
                        seq_id = parts[0]
                else:
                    seq_id = first_token
            else:
                seq_chunks.append(line)
        if seq_id is not None and seq_chunks:
            records.append((seq_id, "".join(seq_chunks)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True, help="输入的多序列 FASTA 文件路径")
    parser.add_argument("--out-dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--model-name",
        type=str,
        default="esm2_t36_3B_UR50D",
        help="ESM-2 模型名称, 需与 esm.pretrained 中的名称一致",
    )
    parser.add_argument(
        "--repr-layer",
        type=int,
        default=36,
        help="取第几层的表示 (不同模型层数不同, 需与 model-name 对应)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch 大小, 根据显存调整",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "index.tsv")
    index_f = open(index_path, "w", encoding="utf-8")
    index_f.write("seq_id\tlength\tfile\n")

    print("读取 FASTA...")
    records = parse_fasta(args.fasta)
    print(f"共读取到 {len(records)} 条序列")

    print(f"加载 ESM-2 模型: {args.model_name}")
    model, alphabet = getattr(esm.pretrained, args.model_name)()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_converter = alphabet.get_batch_converter()

    for i in tqdm(range(0, len(records), args.batch_size), desc="Extracting embeddings"):
        batch_records = records[i : i + args.batch_size]
        batch_labels = [r[0] for r in batch_records]
        batch_seqs = [r[1] for r in batch_records]

        batch_data = list(zip(batch_labels, batch_seqs))
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[args.repr_layer], return_contacts=False)
            reps = out["representations"][args.repr_layer]  # [B, L_token, D]

        for j, (seq_id, seq) in enumerate(batch_records):
            tokens = batch_tokens[j]       # [L_token]
            rep = reps[j]                  # [L_token, D]

            L = len(seq)
            # 去掉开头 BOS, 截取前 L 个残基
            residue_emb = rep[1 : 1 + L]   # [L, D]

            # 序列级别 embedding: 对残基 embedding 取平均
            seq_emb = residue_emb.mean(dim=0)  # [D]

            file_name = f"{seq_id}.pt"
            save_path = os.path.join(args.out_dir, file_name)
            torch.save(
                {
                    "seq_id": seq_id,
                    "seq": seq,
                    "residue_emb": residue_emb.cpu(),
                    "seq_emb": seq_emb.cpu(),
                },
                save_path,
            )

            index_f.write(f"{seq_id}\t{L}\t{file_name}\n")

    index_f.close()
    print(f"完成, 索引文件已保存到: {index_path}")


if __name__ == "__main__":
    main()
