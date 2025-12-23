# -*- coding: utf-8 -*-
"""
train_wet_head.py

作用:
    在 MutPred_v2 的“自监督掩码氨基酸预测”训练框架上，加入“湿实验监督头”，
    使用湿实验的 Δlog(kcat/KM) 对模型打分 score=Δlog10P 进行趋势塑形（Lsign），并可同时拟合幅度（Lreg）。

重要统一约束（你要求的“彻底统一”）:
    1) AA 顺序唯一：AA_ORDER="ARNDCQEGHILKMFPSTWYV"
    2) checkpoint 只保存/只使用 model_state 作为 backbone 权重

本版本关键修改（按你最新要求）:
    - split 不再划分测试集：仅按 8:2 划分 train/val
      * split_dir 下仅写入 train.txt 与 val.txt
      * 若 split_dir 中仍存在旧的 test.txt，会被忽略，不影响运行

调用格式:
python src/train_wet_head.py \
  --graph-dir data/val/graphs \
  --wet-csv data/val/wet_supervision.csv \
  --save-dir results/wet_head_ckpt/egnn_100_noz \
  --ckpt results/mutpred_v2_ckpt/global_graph_global_mask_egnn_100/mutpred_v2_best.pt \
  --epochs 30 \
  --batch-size 2 \
  --mask-ratio 0.15 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --mask-mode global \
  --wet-weight 1.0 \
  --lambda-sign 1.0 \
  --freeze-backbone 0

输入:
    --graph-dir: 图 .pt 目录（自监督训练用）
    --wet-csv: wet_supervision.csv（湿实验监督用）
    --save-dir: 输出 checkpoint 与 log 的目录
    --ckpt: 可选，自监督预训练 checkpoint（必须含 model_state）
输出:
    save-dir/mutpred_v2_wet_best.pt
    save-dir/training_log.csv
"""

import os
import re
import math
import argparse
from typing import List, Tuple, Optional, Dict, Any, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

from models.mutpred_v2_model import MutPredV2Model

# =========================
# 常量：必须与构图脚本一致
# =========================
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
AA_DIM = 20
NODE_TYPE_PROTEIN = 0


class PocketGraphDataset(Dataset):
    def __init__(self, graph_dir: str, uid_list: List[str]):
        super().__init__()
        self.graph_dir = graph_dir
        self.uids = uid_list
        if len(self.uids) == 0:
            raise RuntimeError(f"数据集为空: {graph_dir}")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[idx]
        path = os.path.join(self.graph_dir, uid + ".pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"图文件不存在: {path}")
        graph = torch.load(path)
        graph["uid"] = uid
        return graph


def collate_graphs(batch: List[dict]) -> dict:
    import torch as _torch

    node_feats_list = []
    node_esm_list = []
    node_type_list = []
    pocket_mask_list = []
    pos_list = []
    node_chain_idx_list = []
    node_resseq_list = []

    edge_index_list = []
    edge_attr_list = []

    prot_idx_list = []
    sub_idx_list = []
    tpp_idx_list = []
    mg_idx_list = []

    batch_id_list = []

    node_offset = 0
    for b_id, g in enumerate(batch):
        x = g["x"]
        x_esm = g["x_esm"]
        node_type = g["node_type"]
        pocket_mask = g["pocket_mask"]
        pos = g["pos"]
        node_chain_idx = g["node_chain_idx"]
        node_resseq = g["node_resseq"]

        edge_index = g["edge_index"]
        edge_attr = g["edge_attr"]

        N_i = x.size(0)

        node_feats_list.append(x)
        node_esm_list.append(x_esm)
        node_type_list.append(node_type)
        pocket_mask_list.append(pocket_mask)
        pos_list.append(pos)
        node_chain_idx_list.append(node_chain_idx)
        node_resseq_list.append(node_resseq)

        batch_id_list.append(_torch.full((N_i,), b_id, dtype=_torch.long))

        ei = edge_index + node_offset
        edge_index_list.append(ei)
        edge_attr_list.append(edge_attr)

        prot_idx_list.append([idx + node_offset for idx in g["prot_node_idx"]])
        sub_idx_list.append([idx + node_offset for idx in g["sub_node_idx"]])
        tpp_idx_list.append([idx + node_offset for idx in g.get("tpp_node_idx", [])])
        mg_idx_list.append([idx + node_offset for idx in g.get("mg_node_idx", [])])

        node_offset += N_i

    merged = {
        "x": _torch.cat(node_feats_list, dim=0),
        "x_esm": _torch.cat(node_esm_list, dim=0),
        "node_type": _torch.cat(node_type_list, dim=0),
        "pocket_mask": _torch.cat(pocket_mask_list, dim=0),
        "pos": _torch.cat(pos_list, dim=0),
        "node_chain_idx": _torch.cat(node_chain_idx_list, dim=0),
        "node_resseq": _torch.cat(node_resseq_list, dim=0),
        "edge_index": _torch.cat(edge_index_list, dim=1),
        "edge_attr": _torch.cat(edge_attr_list, dim=0),
        "prot_node_idx": sum(prot_idx_list, []),
        "sub_node_idx": sum(sub_idx_list, []),
        "tpp_node_idx": sum(tpp_idx_list, []),
        "mg_node_idx": sum(mg_idx_list, []),
        "batch": _torch.cat(batch_id_list, dim=0),
    }
    return merged


def build_labels_from_x_pocket(x: torch.Tensor, node_type: torch.Tensor, pocket_mask: torch.Tensor) -> torch.Tensor:
    aa_onehot = x[:, :AA_DIM]
    aa_idx = aa_onehot.argmax(dim=-1)
    labels = torch.full_like(aa_idx, fill_value=-100)
    valid = (node_type == NODE_TYPE_PROTEIN) & pocket_mask
    labels[valid] = aa_idx[valid]
    return labels


def build_labels_from_x_global(x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
    aa_onehot = x[:, :AA_DIM]
    aa_idx = aa_onehot.argmax(dim=-1)
    labels = torch.full_like(aa_idx, fill_value=-100)
    labels[node_type == NODE_TYPE_PROTEIN] = aa_idx[node_type == NODE_TYPE_PROTEIN]
    return labels


def apply_random_mask(x: torch.Tensor, x_esm: torch.Tensor, labels: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    device = x.device
    N = x.size(0)

    valid = (labels != -100)
    if int(valid.sum().item()) == 0:
        return torch.full_like(labels, fill_value=-100)

    rand = torch.rand(N, device=device)
    mask = (rand < mask_ratio) & valid
    if int(mask.sum().item()) == 0:
        valid_idx = valid.nonzero(as_tuple=False).view(-1)
        pick = valid_idx[torch.randint(len(valid_idx), (1,), device=device)]
        mask[pick] = True

    x[:, :AA_DIM][mask] = 0.0
    x_esm[mask] = 0.0

    masked_labels = torch.full_like(labels, fill_value=-100)
    masked_labels[mask] = labels[mask]
    return masked_labels


def parse_variants(v: str) -> Optional[List[Tuple[str, int, str]]]:
    if not isinstance(v, str):
        return None
    v = v.strip().upper()
    if v in ("WT", "WILDTYPE", "WILD TYPE", "WILD-TYPE", "NATIVE"):
        return None
    for sep in [";", "/", "+", ","]:
        v = v.replace(sep, " ")
    tokens = [tok for tok in v.split() if tok]
    out: List[Tuple[str, int, str]] = []
    for tok in tokens:
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", tok)
        if m is None:
            return None
        wt_aa, pos_str, mut_aa = m.groups()
        out.append((wt_aa, int(pos_str), mut_aa))
    return out if out else None


class WetPairDataset(Dataset):
    """
    输出单条样本 dict:
        {
          "wt": {x,x_esm,pos,edge_index,edge_attr,node_type,node_resseq},
          "mut_sites": List[{"wt_idx":int, "mut_idx":int, "node_idx": LongTensor[K]}],
          "delta": FloatTensor[1]  (Δexp)
          "y": FloatTensor[1]      (sign(Δexp) in {+1,-1})
          "uid": str
        }
    """
    def __init__(self, csv_path: str, graph_dir: str, allowed_uids: Optional[Set[str]] = None):
        super().__init__()
        self.graph_dir = graph_dir
        df = pd.read_csv(csv_path)

        required_cols = ["complex_uid", "enzyme_variant", "delta_log_kcatKm"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"wet_supervision.csv 缺少列: {col}，当前列={list(df.columns)}")

        variants: List[List[Tuple[str, int, str]]] = []
        keep_idx: List[int] = []
        for i, v in enumerate(df["enzyme_variant"].tolist()):
            info = parse_variants(v)
            if info is None:
                continue
            uid = str(df.loc[i, "complex_uid"])
            if allowed_uids is not None and uid not in allowed_uids:
                continue
            keep_idx.append(i)
            variants.append(info)

        self.df = df.iloc[keep_idx].reset_index(drop=True)
        self.variants = variants

        if len(self.df) == 0:
            raise RuntimeError("WetPairDataset 为空：请检查 allowed_uids 过滤是否过严，或 wet_csv 是否有突变样本。")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        uid = str(row["complex_uid"])
        delta = float(row["delta_log_kcatKm"])
        y = 1.0 if delta > 0 else -1.0

        graph_path = os.path.join(self.graph_dir, f"{uid}.pt")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"找不到图文件: {graph_path}")

        g = torch.load(graph_path)

        wt = {
            "x": g["x"].clone(),
            "x_esm": g["x_esm"].clone(),
            "pos": g["pos"].clone(),
            "edge_index": g["edge_index"].clone(),
            "edge_attr": g["edge_attr"].clone(),
            "node_type": g["node_type"].clone(),
            "node_resseq": g["node_resseq"].clone(),
        }

        prot_mask = (wt["node_type"] == NODE_TYPE_PROTEIN)
        aa_onehot_wt = wt["x"][:, :AA_DIM].clone()

        mut_sites: List[Dict[str, Any]] = []
        mut_list = self.variants[idx]

        for wt_aa, res_pos, mut_aa in mut_list:
            wt_idx = AA_TO_IDX.get(wt_aa, None)
            mut_idx = AA_TO_IDX.get(mut_aa, None)
            if wt_idx is None or mut_idx is None:
                raise ValueError(f"AA_ORDER 中找不到氨基酸: {wt_aa} 或 {mut_aa}")

            res_mask = (wt["node_resseq"] == int(res_pos))
            wt_mask = (aa_onehot_wt[:, wt_idx] > 0.5)
            node_mask = prot_mask & res_mask & wt_mask
            node_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)

            if node_idx.numel() == 0:
                node_mask = prot_mask & res_mask
                node_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)

            mut_sites.append({"wt_idx": wt_idx, "mut_idx": mut_idx, "node_idx": node_idx})

        return {
            "wt": wt,
            "mut_sites": mut_sites,
            "y": torch.tensor([y], dtype=torch.float32),
            "delta": torch.tensor([delta], dtype=torch.float32),
            "uid": uid,
        }


class WetSupervisionHead(nn.Module):
    """
    delta_pred = a * score + b
    a = softplus(alpha) > 0
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self.alpha)
        return a * score + self.b


def mask_mut_sites_inplace(x: torch.Tensor, x_esm: torch.Tensor, mut_sites: List[Dict[str, Any]]) -> None:
    idx_all: List[torch.Tensor] = []
    for s in mut_sites:
        node_idx = s["node_idx"]
        if torch.is_tensor(node_idx) and node_idx.numel() > 0:
            idx_all.append(node_idx)
    if not idx_all:
        return
    idx = torch.cat(idx_all, dim=0).unique()
    x[idx, :AA_DIM] = 0.0
    x_esm[idx] = 0.0


def compute_score_from_logits_wt(
    logits_wt: torch.Tensor,
    mut_sites: List[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    """
    score = mean_over_sites( mean_chains(log10 P(mutAA)) - mean_chains(log10 P(wtAA)) )
    """
    if logits_wt.dim() != 2 or logits_wt.size(-1) != AA_DIM:
        raise ValueError("logits_wt 形状应为 [N, 20]")

    logp_ln = F.log_softmax(logits_wt, dim=-1)
    ln10 = math.log(10.0)

    site_scores: List[torch.Tensor] = []
    for s in mut_sites:
        wt_idx = int(s["wt_idx"])
        mut_idx = int(s["mut_idx"])
        node_idx = s["node_idx"]
        if not torch.is_tensor(node_idx) or node_idx.numel() == 0:
            continue
        lp_mut = logp_ln[node_idx, mut_idx].mean()
        lp_wt = logp_ln[node_idx, wt_idx].mean()
        score_site = (lp_mut - lp_wt) / ln10
        site_scores.append(score_site)

    if not site_scores:
        return torch.tensor(0.0, device=device)

    return torch.stack(site_scores, dim=0).mean().view(())


def make_or_load_splits(
    graph_dir: str,
    split_dir: str,
    train_ratio: float = 0.8,
    seed: int = 1234,
) -> Tuple[List[str], List[str]]:
    """
    仅划分 train/val，不再生成 test。
    - train_ratio 默认 0.8，对应 val_ratio=0.2
    - split_dir 下只写入 train.txt 与 val.txt
    - 若存在旧的 test.txt，会被忽略
    """
    os.makedirs(split_dir, exist_ok=True)
    train_txt = os.path.join(split_dir, "train.txt")
    val_txt = os.path.join(split_dir, "val.txt")

    def read_list(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def write_list(path: str, xs: List[str]) -> None:
        with open(path, "w") as f:
            for x in xs:
                f.write(x + "\n")

    if os.path.exists(train_txt) and os.path.exists(val_txt):
        return read_list(train_txt), read_list(val_txt)

    all_files = [f for f in os.listdir(graph_dir) if f.endswith(".pt")]
    uids = [os.path.splitext(f)[0] for f in all_files]
    uids.sort()
    if len(uids) < 2:
        raise RuntimeError(f"在 {graph_dir} 中图文件数量不足以划分 train/val（至少需要 2 个 uid），当前={len(uids)}")

    import random
    rnd = random.Random(seed)
    rnd.shuffle(uids)

    n = len(uids)
    n_train = int(round(n * float(train_ratio)))
    n_train = max(1, min(n_train, n - 1))
    n_val = n - n_train
    if n_val <= 0:
        n_train = n - 1
        n_val = 1

    train_uids = uids[:n_train]
    val_uids = uids[n_train:]

    write_list(train_txt, train_uids)
    write_list(val_txt, val_uids)
    return train_uids, val_uids


def spearman_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) == 0 or len(ys) == 0 or len(xs) != len(ys):
        return float("nan")
    sx = pd.Series(xs).rank(method="average")
    sy = pd.Series(ys).rank(method="average")
    v = sx.corr(sy, method="pearson")
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return float("nan")
    return float(v)


@torch.no_grad()
def evaluate_wet(
    backbone: MutPredV2Model,
    wet_head: WetSupervisionHead,
    wet_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    backbone.eval()
    wet_head.eval()

    scores: List[float] = []
    deltas: List[float] = []
    preds: List[float] = []
    sign_ok = 0
    n = 0

    for batch in wet_loader:
        wt = batch["wt"]
        mut_sites = batch["mut_sites"]
        delta_exp = float(batch["delta"].view(-1)[0].item())

        wt_x = wt["x"].to(device)
        wt_xesm = wt["x_esm"].to(device)
        wt_pos = wt["pos"].to(device)
        wt_ei = wt["edge_index"].to(device)
        wt_ea = wt["edge_attr"].to(device)

        mask_mut_sites_inplace(wt_x, wt_xesm, mut_sites)

        logits_wt = backbone(wt_x, wt_xesm, wt_ei, wt_ea, wt_pos)
        score = compute_score_from_logits_wt(logits_wt, mut_sites, device=device)
        delta_pred = wet_head(score).view(())

        score_f = float(score.item())
        pred_f = float(delta_pred.item())

        scores.append(score_f)
        deltas.append(delta_exp)
        preds.append(pred_f)

        if delta_exp > 0 and score_f > 0:
            sign_ok += 1
        elif delta_exp < 0 and score_f < 0:
            sign_ok += 1
        n += 1

    sp = spearman_corr(scores, deltas)
    mse = float(pd.Series([(p - d) ** 2 for p, d in zip(preds, deltas)]).mean()) if n > 0 else float("nan")
    sign_acc = float(sign_ok / n) if n > 0 else float("nan")

    return {
        "spearman_score_delta": sp,
        "mse_delta_pred": mse,
        "sign_acc": sign_acc,
        "n": float(n),
    }


def collate_keep_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return batch[0]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--graph-dir", type=str, required=True, help="图 .pt 目录")
    ap.add_argument("--save-dir", type=str, required=True, help="输出目录")
    ap.add_argument("--split-dir", type=str, default=None, help="split 目录（默认 graph_dir 上一级/splits）")

    ap.add_argument("--ckpt", type=str, default=None, help="可选：加载自监督预训练 checkpoint（必须含 model_state）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--mask-mode", type=str, default="global", choices=["pocket", "global"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)

    # 固定 8:2：只保留 train_ratio（val_ratio = 1 - train_ratio）
    ap.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例（默认 0.8，对应验证集 0.2）")

    ap.add_argument("--wet-csv", type=str, required=True, help="wet_supervision.csv 路径")
    ap.add_argument("--wet-graph-dir", type=str, default=None, help="wet 对应图目录（默认同 --graph-dir）")
    ap.add_argument("--wet-weight", type=float, default=1.0, help="总损失中湿实验损失的权重")
    ap.add_argument("--lambda-sign", type=float, default=1.0, help="L_wet = L_reg + lambda_sign * L_sign")
    ap.add_argument("--freeze-backbone", type=int, default=0, help="1=冻结 backbone；0=允许 wet 反传进 backbone")

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.wet_graph_dir is None:
        args.wet_graph_dir = args.graph_dir

    if args.split_dir is None:
        graph_dir_abs = os.path.abspath(args.graph_dir)
        args.split_dir = os.path.join(os.path.dirname(graph_dir_abs), "splits")
    os.makedirs(args.split_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    train_uids, val_uids = make_or_load_splits(
        graph_dir=args.graph_dir,
        split_dir=args.split_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    train_uid_set = set(train_uids)
    val_uid_set = set(val_uids)
    print(f"图数据 split(8:2): train_uid={len(train_uids)}, val_uid={len(val_uids)}")

    train_dataset = PocketGraphDataset(args.graph_dir, train_uids)
    val_dataset = PocketGraphDataset(args.graph_dir, val_uids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    wet_train_dataset = WetPairDataset(args.wet_csv, args.wet_graph_dir, allowed_uids=train_uid_set)
    wet_val_dataset = WetPairDataset(args.wet_csv, args.wet_graph_dir, allowed_uids=val_uid_set)
    print(f"湿实验 split(8:2): wet_train={len(wet_train_dataset)}, wet_val={len(wet_val_dataset)}")

    wet_train_loader = DataLoader(
        wet_train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_keep_single,
    )
    wet_val_loader = DataLoader(
        wet_val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_keep_single,
    )

    sample_graph = train_dataset[0]
    struct_dim = sample_graph["x"].shape[1]
    esm_dim = sample_graph["x_esm"].shape[1]
    edge_dim = sample_graph["edge_attr"].shape[1]

    backbone = MutPredV2Model(
        struct_dim=struct_dim,
        esm_dim=esm_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=AA_DIM,
    ).to(device)
    wet_head = WetSupervisionHead().to(device)

    if args.ckpt is not None and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise KeyError(
                f"--ckpt 指向的文件缺少 'model_state': {args.ckpt}\n"
                "请使用 train_global.py 的输出 mutpred_v2_best.pt。"
            )
        state = ckpt["model_state"]
        backbone.load_state_dict(state, strict=True)
        print(f"加载 ckpt(model_state): {args.ckpt}")

    if int(args.freeze_backbone) == 1:
        for p in backbone.parameters():
            p.requires_grad = False
        print("[INFO] freeze_backbone=1: backbone 已冻结。")

    params = list(wet_head.parameters())
    if int(args.freeze_backbone) == 0:
        params += [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_wet_val_spearman = -1e9
    best_path = os.path.join(args.save_dir, "mutpred_v2_wet_best.pt")
    log_path = os.path.join(args.save_dir, "training_log.csv")

    with open(log_path, "w") as f:
        f.write(
            "epoch,"
            "self_val_loss,self_val_acc,"
            "wet_train_spearman,wet_train_mse,wet_train_sign_acc,wet_train_n,"
            "wet_val_spearman,wet_val_mse,wet_val_sign_acc,wet_val_n\n"
        )

    for epoch in range(1, args.epochs + 1):
        # (1) 自监督训练
        backbone.train(int(args.freeze_backbone) == 0)
        wet_head.train()

        for graph in tqdm(train_loader, desc=f"Epoch {epoch} 自监督训练", leave=False):
            x = graph["x"].to(device)
            x_esm = graph["x_esm"].to(device)
            edge_index = graph["edge_index"].to(device)
            edge_attr = graph["edge_attr"].to(device)
            pos = graph["pos"].to(device)
            node_type = graph["node_type"].to(device)
            pocket_mask = graph["pocket_mask"].to(device)

            if args.mask_mode == "pocket":
                labels = build_labels_from_x_pocket(x, node_type, pocket_mask)
            else:
                labels = build_labels_from_x_global(x, node_type)

            masked_labels = apply_random_mask(x, x_esm, labels, args.mask_ratio)
            num_supervised = int((masked_labels != -100).sum().item())
            if num_supervised == 0:
                continue

            logits = backbone(x, x_esm, edge_index, edge_attr, pos)
            loss = F.cross_entropy(logits, masked_labels, ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # (2) 湿实验训练：logits_wt -> score -> Lreg/Lsign
        backbone.train(int(args.freeze_backbone) == 0)
        wet_head.train()

        for batch in tqdm(wet_train_loader, desc=f"Epoch {epoch} 湿实验训练", leave=False):
            wt = batch["wt"]
            mut_sites = batch["mut_sites"]
            y = batch["y"].to(device).view(1)
            delta_exp = batch["delta"].to(device).view(1)

            wt_x = wt["x"].to(device)
            wt_xesm = wt["x_esm"].to(device)
            wt_pos = wt["pos"].to(device)
            wt_ei = wt["edge_index"].to(device)
            wt_ea = wt["edge_attr"].to(device)

            mask_mut_sites_inplace(wt_x, wt_xesm, mut_sites)

            logits_wt = backbone(wt_x, wt_xesm, wt_ei, wt_ea, wt_pos)
            score = compute_score_from_logits_wt(logits_wt, mut_sites, device=device)
            delta_pred = wet_head(score).view(1)

            L_reg = F.mse_loss(delta_pred, delta_exp)
            L_sign = F.softplus(-y * score).mean()
            L_wet = L_reg + float(args.lambda_sign) * L_sign

            optimizer.zero_grad()
            (float(args.wet_weight) * L_wet).backward()
            optimizer.step()

        # (3) 自监督验证
        backbone.eval()
        with torch.no_grad():
            self_val_loss_sum = 0.0
            self_val_nodes = 0
            self_val_correct = 0

            for graph in tqdm(val_loader, desc=f"Epoch {epoch} 自监督验证", leave=False):
                x = graph["x"].to(device)
                x_esm = graph["x_esm"].to(device)
                edge_index = graph["edge_index"].to(device)
                edge_attr = graph["edge_attr"].to(device)
                pos = graph["pos"].to(device)
                node_type = graph["node_type"].to(device)
                pocket_mask = graph["pocket_mask"].to(device)

                if args.mask_mode == "pocket":
                    labels = build_labels_from_x_pocket(x, node_type, pocket_mask)
                else:
                    labels = build_labels_from_x_global(x, node_type)

                masked_labels = apply_random_mask(x, x_esm, labels, args.mask_ratio)
                num_supervised = int((masked_labels != -100).sum().item())
                if num_supervised == 0:
                    continue

                logits = backbone(x, x_esm, edge_index, edge_attr, pos)
                loss = F.cross_entropy(logits, masked_labels, ignore_index=-100)

                self_val_loss_sum += float(loss.item()) * num_supervised
                self_val_nodes += num_supervised

                preds = logits.argmax(dim=-1)
                correct = (preds == masked_labels) & (masked_labels != -100)
                self_val_correct += int(correct.sum().item())

            self_val_loss = self_val_loss_sum / max(self_val_nodes, 1)
            self_val_acc = self_val_correct / max(self_val_nodes, 1)

        # (4) 湿实验评估：全量 wet_train + wet_val
        wet_train_metrics = evaluate_wet(backbone, wet_head, wet_train_loader, device)
        wet_val_metrics = evaluate_wet(backbone, wet_head, wet_val_loader, device)

        print(
            f"[Epoch {epoch}] "
            f"自监督验证: loss={self_val_loss:.4f} acc={self_val_acc:.4f} | "
            f"湿实验训练: spearman(score,Δexp)={wet_train_metrics['spearman_score_delta']:.4f} "
            f"mse={wet_train_metrics['mse_delta_pred']:.4f} sign_acc={wet_train_metrics['sign_acc']:.4f} n={int(wet_train_metrics['n'])} | "
            f"湿实验验证: spearman(score,Δexp)={wet_val_metrics['spearman_score_delta']:.4f} "
            f"mse={wet_val_metrics['mse_delta_pred']:.4f} sign_acc={wet_val_metrics['sign_acc']:.4f} n={int(wet_val_metrics['n'])}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},"
                f"{self_val_loss:.6f},{self_val_acc:.6f},"
                f"{wet_train_metrics['spearman_score_delta']:.6f},{wet_train_metrics['mse_delta_pred']:.6f},{wet_train_metrics['sign_acc']:.6f},{int(wet_train_metrics['n'])},"
                f"{wet_val_metrics['spearman_score_delta']:.6f},{wet_val_metrics['mse_delta_pred']:.6f},{wet_val_metrics['sign_acc']:.6f},{int(wet_val_metrics['n'])}\n"
            )

        cur_wet_val_spearman = wet_val_metrics["spearman_score_delta"]
        if not (math.isnan(cur_wet_val_spearman) or math.isinf(cur_wet_val_spearman)):
            if cur_wet_val_spearman > best_wet_val_spearman:
                best_wet_val_spearman = cur_wet_val_spearman

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": backbone.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "args": vars(args),
                        "aa_order": AA_ORDER,
                        "best_wet_val_spearman": best_wet_val_spearman,
                        "self_val_loss": self_val_loss,
                        "self_val_acc": self_val_acc,
                        "wet_train_metrics": wet_train_metrics,
                        "wet_val_metrics": wet_val_metrics,
                    },
                    best_path,
                )
                print(f"发现更优模型（wet_val_spearman={best_wet_val_spearman:.4f}），已保存到: {best_path}")

    print(f"训练结束，best wet_val_spearman={best_wet_val_spearman:.4f}")
    print(f"日志: {log_path}")


if __name__ == "__main__":
    main()
