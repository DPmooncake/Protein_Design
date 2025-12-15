# -*- coding: utf-8 -*-
"""
train_mutpred_v2.py

作用:
    使用口袋/全局图 .pt 文件训练 MutPred_v2 自监督模型, 任务为 masked amino-acid 预测。
    模型输入为:
        - 结构/类型节点特征 x (来自 build_graphs.py),
        - ESM 残基级别 embedding x_esm,
        - 边索引 edge_index 及边特征 edge_attr。
    模型输出为每个节点对 20 种氨基酸的 logits (20 维, 对应 20 种标准氨基酸)。

数据划分:
    - 对 graph_dir 中的所有 {uid}.pt 做 train/val/test 划分 (按 uid 划分);
    - 若 split_dir 下已存在 train.txt / val.txt / test.txt, 则直接读取并复用该划分;
    - 若不存在, 则根据给定随机种子生成一次划分并写入上述三个 txt 文件以便复现。

batch 处理:
    - 使用 collate_graphs 将一批图拼接成一个“大图”:
        - 节点特征按节点维拼接;
        - 边索引加节点 offset 后按边维拼接;
        - 新增 "batch" 向量记录每个节点属于哪个子图 (当前未使用)。

新增功能:
    - 掩码/监督模式 mask_mode:
        * "pocket": 仅在蛋白 pocket 节点上构建标签与掩码 (原始行为);
        * "global": 在所有蛋白节点上构建标签与掩码 (全局随机 mask)。

输出:
    - split_dir/train.txt, val.txt, test.txt: 每行一个 uid, 对应 {uid}.pt;
    - 训练过程中仅保存一个最佳模型:
        save-dir 下:
        - mutpred_v2_best.pt (包含 model_state 等信息)。
    - 训练日志:
        save-dir/training_log.csv, 每行包含 epoch, train_loss, val_loss, val_acc;
    - 训练曲线:
        save-dir/loss_curve.png, save-dir/acc_curve.png

调用格式示例:
    # 口袋图 + 口袋监督
    python src/train_mutpred_v2.py \
        --graph-dir data/graphs_pocket \
        --save-dir results/mutpred_v2_ckpt/pocket_100 \
        --epochs 100 \
        --batch-size 32 \
        --mask-ratio 0.15 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --num-workers 2 \
        --mask-mode pocket

    # 全局图 + 口袋监督
    python src/train_global.py \
        --graph-dir data/graphs_global \
        --save-dir results/mutpred_v2_ckpt/global_graph_pocket_mask_100 \
        --epochs 100 \
        --batch-size 2 \
        --mask-ratio 0.15 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --num-workers 2 \
        --mask-mode pocket

    # 全局图 + 全局监督
    python src/train_global.py \
        --graph-dir data/graphs_global \
        --save-dir results/mutpred_v2_ckpt/global_graph_global_mask_300 \
        --epochs 300 \
        --batch-size 4 \
        --mask-ratio 0.15 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --num-workers 4 \
        --mask-mode global
"""

import os
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.mutpred_v2_model import MutPredV2Model

# 必须与构图脚本保持一致
AA_DIM = 20
NODE_TYPE_PROTEIN = 0  # build_graphs.py 中 NODE_TYPE_MAP["protein"] 的值


class PocketGraphDataset(Dataset):
    """
    基于 uid 列表的图数据集:
        - graph_dir 下存放若干 {uid}.pt;
        - uid_list 中的每个 uid 对应一个样本。
    """

    def __init__(self, graph_dir: str, uid_list: List[str]):
        super().__init__()
        self.graph_dir = graph_dir
        self.uids = uid_list
        if len(self.uids) == 0:
            raise RuntimeError(f"数据集为空: {graph_dir}")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        path = os.path.join(self.graph_dir, uid + ".pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"图文件不存在: {path}")
        graph = torch.load(path)
        graph["uid"] = uid
        return graph


def collate_graphs(batch: List[dict]) -> dict:
    """
    将一批图 (list[graph_dict]) 拼接成一个大图, 支持 batch_size > 1 的训练。

    输入每个 graph 包含字段:
        - x:             [N_i, F_struct]
        - x_esm:         [N_i, D_esm]
        - node_type:     [N_i]
        - pocket_mask:   [N_i]
        - pos:           [N_i, 3]
        - node_chain_idx:[N_i]
        - node_resseq:   [N_i]
        - edge_index:    [2, E_i]
        - edge_attr:     [E_i, F_edge]
        - prot_node_idx: List[int]
        - sub_node_idx:  List[int]
        - tpp_node_idx:  List[int]
        - mg_node_idx:   List[int]

    输出 merged_graph:
        同上字段, 但所有节点/边已拼接, 并新增:
        - batch: [N_total], 每个元素为该节点属于的子图 id (0..B-1)。
    """
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


def build_labels_from_x_pocket(
    x: torch.Tensor,
    node_type: torch.Tensor,
    pocket_mask: torch.Tensor,
) -> torch.Tensor:
    """
    口袋监督版本:
        - 假定 x 的前 AA_DIM 维为 20 种氨基酸 one-hot;
        - 仅在蛋白且属于 pocket 的节点上有有效标签;
        - 其它节点标签设为 -100, 以便 cross_entropy(ignore_index=-100) 忽略。
    """
    aa_onehot = x[:, :AA_DIM]
    aa_idx = aa_onehot.argmax(dim=-1)  # 0..19

    labels = torch.full_like(aa_idx, fill_value=-100)
    is_protein = (node_type == NODE_TYPE_PROTEIN)
    valid = is_protein & pocket_mask
    labels[valid] = aa_idx[valid]
    return labels


def build_labels_from_x_global(
    x: torch.Tensor,
    node_type: torch.Tensor,
) -> torch.Tensor:
    """
    全局监督版本:
        - 同样从 x 的前 AA_DIM 维恢复氨基酸;
        - 所有蛋白节点 (node_type == NODE_TYPE_PROTEIN) 都有标签;
        - 其他节点标签为 -100。
    """
    aa_onehot = x[:, :AA_DIM]
    aa_idx = aa_onehot.argmax(dim=-1)

    labels = torch.full_like(aa_idx, fill_value=-100)
    is_protein = (node_type == NODE_TYPE_PROTEIN)
    labels[is_protein] = aa_idx[is_protein]
    return labels


def apply_random_mask(
    x: torch.Tensor,
    x_esm: torch.Tensor,
    labels: torch.Tensor,
    mask_ratio: float,
) -> torch.Tensor:
    """
    在有效监督节点 (labels != -100) 中随机掩码一部分, 并返回仅在被掩码位置保留的标签。

    掩码策略:
        - 有效监督节点: labels != -100;
        - 在这些节点上以 mask_ratio 的概率进行掩码;
        - 对被掩码节点:
            - x 前 20 维 (氨基酸 one-hot) 置 0;
            - x_esm 置 0;
        - masked_labels 在被掩码位置等于原始标签, 其它位置为 -100。
    """
    device = x.device
    N = x.size(0)

    valid = (labels != -100)
    num_valid = int(valid.sum().item())
    if num_valid == 0:
        return torch.full_like(labels, fill_value=-100)

    rand = torch.rand(N, device=device)
    mask = (rand < mask_ratio) & valid

    if mask.sum().item() == 0:
        valid_idx = valid.nonzero(as_tuple=False).view(-1)
        rand_idx = valid_idx[torch.randint(len(valid_idx), (1,))]
        mask[rand_idx] = True

    x[:, :AA_DIM][mask] = 0.0
    x_esm[mask] = 0.0

    masked_labels = torch.full_like(labels, fill_value=-100)
    masked_labels[mask] = labels[mask]
    return masked_labels


def train_one_epoch(
    model: MutPredV2Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
    mask_mode: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for graph in tqdm(loader, desc="Train", leave=False):
        x = graph["x"].to(device)
        x_esm = graph["x_esm"].to(device)
        edge_index = graph["edge_index"].to(device)
        edge_attr = graph["edge_attr"].to(device)
        node_type = graph["node_type"].to(device)
        pocket_mask = graph["pocket_mask"].to(device)

        if mask_mode == "pocket":
            labels = build_labels_from_x_pocket(x, node_type, pocket_mask)
        elif mask_mode == "global":
            labels = build_labels_from_x_global(x, node_type)
        else:
            raise ValueError(f"未知 mask_mode: {mask_mode}")

        masked_labels = apply_random_mask(x, x_esm, labels, mask_ratio)

        num_supervised = (masked_labels != -100).sum().item()
        if num_supervised == 0:
            continue

        logits = model(x, x_esm, edge_index, edge_attr)
        loss = F.cross_entropy(logits, masked_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * num_supervised
        total_nodes += num_supervised

    avg_loss = total_loss / max(total_nodes, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: MutPredV2Model,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    mask_mode: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_nodes = 0
    total_correct = 0

    for graph in tqdm(loader, desc="Val", leave=False):
        x = graph["x"].to(device)
        x_esm = graph["x_esm"].to(device)
        edge_index = graph["edge_index"].to(device)
        edge_attr = graph["edge_attr"].to(device)
        node_type = graph["node_type"].to(device)
        pocket_mask = graph["pocket_mask"].to(device)

        if mask_mode == "pocket":
            labels = build_labels_from_x_pocket(x, node_type, pocket_mask)
        elif mask_mode == "global":
            labels = build_labels_from_x_global(x, node_type)
        else:
            raise ValueError(f"未知 mask_mode: {mask_mode}")

        masked_labels = apply_random_mask(x, x_esm, labels, mask_ratio)

        num_supervised = (masked_labels != -100).sum().item()
        if num_supervised == 0:
            continue

        logits = model(x, x_esm, edge_index, edge_attr)
        loss = F.cross_entropy(logits, masked_labels, ignore_index=-100)

        total_loss += loss.item() * num_supervised
        total_nodes += num_supervised

        preds = logits.argmax(dim=-1)
        correct = (preds == masked_labels) & (masked_labels != -100)
        total_correct += correct.sum().item()

    avg_loss = total_loss / max(total_nodes, 1)
    avg_acc = total_correct / max(total_nodes, 1)
    return avg_loss, avg_acc


def make_or_load_splits(
    graph_dir: str,
    split_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1234,
) -> Tuple[List[str], List[str], List[str]]:
    """
    生成或读取 train/val/test 划分:
        - 若 split_dir 下存在 train.txt, val.txt, test.txt, 则直接读取 uid 列表;
        - 否则对 graph_dir 下所有 {uid}.pt 根据比例随机划分一次并写入 txt。
    """
    os.makedirs(split_dir, exist_ok=True)
    train_txt = os.path.join(split_dir, "train.txt")
    val_txt = os.path.join(split_dir, "val.txt")
    test_txt = os.path.join(split_dir, "test.txt")

    def read_list(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def write_list(path: str, xs: List[str]) -> None:
        with open(path, "w") as f:
            for x in xs:
                f.write(x + "\n")

    if os.path.exists(train_txt) and os.path.exists(val_txt) and os.path.exists(test_txt):
        train_uids = read_list(train_txt)
        val_uids = read_list(val_txt)
        test_uids = read_list(test_txt)
        return train_uids, val_uids, test_uids

    all_files = [f for f in os.listdir(graph_dir) if f.endswith(".pt")]
    uids = [os.path.splitext(f)[0] for f in all_files]
    uids.sort()

    if len(uids) == 0:
        raise RuntimeError(f"在 {graph_dir} 中未找到任何 .pt 图文件")

    import random
    rnd = random.Random(seed)
    rnd.shuffle(uids)

    n = len(uids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_uids = uids[:n_train]
    val_uids = uids[n_train:n_train + n_val]
    test_uids = uids[n_train + n_val:]

    write_list(train_txt, train_uids)
    write_list(val_txt, val_uids)
    write_list(test_txt, test_uids)

    return train_uids, val_uids, test_uids


def main():
    parser = argparse.ArgumentParser(description="Train MutPred_v2 self-supervised model")
    parser.add_argument("--graph-dir", type=str, required=True, help="图 .pt 文件目录")
    parser.add_argument("--save-dir", type=str, required=True, help="模型保存目录")
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="划分 txt 保存目录; 若未指定, 默认为 graph_dir 的上一级目录下 splits/ 子目录",
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="batch 大小(图的数量)")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="掩码比例")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--hidden-dim", type=int, default=256, help="GNN 隐藏维度")
    parser.add_argument("--num-layers", type=int, default=4, help="GNN 层数")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例(剩余为测试集)")
    parser.add_argument("--seed", type=int, default=1234, help="划分随机种子")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 的 num_workers (建议 >0 以提高数据加载速度)",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="pocket",
        choices=["pocket", "global"],
        help="自监督掩码/监督范围: pocket=仅口袋蛋白节点; global=所有蛋白节点",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 默认的 split_dir: graph_dir 的上一级目录下的 splits 子目录
    if args.split_dir is None:
        graph_dir_abs = os.path.abspath(args.graph_dir)
        parent = os.path.dirname(graph_dir_abs)  # 例如 data/graphs -> data
        args.split_dir = os.path.join(parent, "splits")
    os.makedirs(args.split_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 划分或读取已有划分
    train_uids, val_uids, test_uids = make_or_load_splits(
        graph_dir=args.graph_dir,
        split_dir=args.split_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Train: {len(train_uids)}, Val: {len(val_uids)}, Test: {len(test_uids)}")
    print(f"划分文件位于: {args.split_dir}")

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

    sample_graph = train_dataset[0]
    struct_dim = sample_graph["x"].shape[1]
    esm_dim = sample_graph["x_esm"].shape[1]
    edge_dim = sample_graph["edge_attr"].shape[1]

    model = MutPredV2Model(
        struct_dim=struct_dim,
        esm_dim=esm_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=AA_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_path = os.path.join(args.save_dir, "mutpred_v2_best.pt")

    # 准备日志文件
    log_path = os.path.join(args.save_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_acc\n")

    # 用于画图的历史记录
    epoch_list = []
    train_hist = []
    val_hist = []
    acc_hist = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch} =====")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mask_ratio=args.mask_ratio,
            mask_mode=args.mask_mode,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            mask_ratio=args.mask_ratio,
            mask_mode=args.mask_mode,
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val   loss: {val_loss:.4f}, Val masked acc: {val_acc:.4f}")

        # 追加到 CSV
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_acc:.6f}\n")

        # 存到内存里用于画图
        epoch_list.append(epoch)
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        acc_hist.append(val_acc)

        # 只保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_path,
            )
            print(f"发现更优模型, 已保存到: {best_path}")

    print(f"\n训练结束. 最优 epoch = {best_epoch}, 最优 val_loss = {best_val_loss:.4f}")
    print(f"最终模型路径: {best_path}")
    print(f"训练日志已写入: {log_path}")

    # 画 loss 曲线
    if len(epoch_list) > 0:
        plt.figure()
        plt.plot(epoch_list, train_hist, label="train_loss")
        plt.plot(epoch_list, val_hist, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        loss_fig_path = os.path.join(args.save_dir, "loss_curve.png")
        plt.savefig(loss_fig_path, dpi=200)
        plt.close()
        print(f"loss 曲线已保存到: {loss_fig_path}")

        # 画 val_acc 曲线
        plt.figure()
        plt.plot(epoch_list, acc_hist, label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Val masked accuracy")
        plt.title("Validation Masked Accuracy")
        plt.legend()
        acc_fig_path = os.path.join(args.save_dir, "acc_curve.png")
        plt.savefig(acc_fig_path, dpi=200)
        plt.close()
        print(f"accuracy 曲线已保存到: {acc_fig_path}")


if __name__ == "__main__":
    main()
