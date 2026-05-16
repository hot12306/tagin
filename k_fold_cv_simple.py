"""
简化版K折交叉验证脚本
与现有代码结构兼容
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from global_local_fusion_gate import GlobalLocalFusionTaskModelGate
from utile import set_seed, get_data, get_loaders, train_func, evaluate_model_on_dataset, calculate_metrics
from dataset import MyDataset
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data

warnings.filterwarnings("ignore", category=UserWarning)

# ==================== 参数配置 ====================
parser = argparse.ArgumentParser(description='K-Fold Cross Validation')
parser.add_argument('--seed', type=int, default=2023, help="random seed")
parser.add_argument('--dataset_name', type=str, default='bindingdB',
                    choices=['bindingdB', 'ttd', 'drugbank1w'])
parser.add_argument('--k_folds', type=int, default=5, help='number of folds')
parser.add_argument('--ratio', type=float, default=1, help='pos/neg ratio')
parser.add_argument('--task', type=str, default='SP', choices=['SD', 'ST', 'SP'])

# 模型参数
parser.add_argument('--dm', type=bool, default=True)
parser.add_argument('--dm_layers', type=int, default=2)
parser.add_argument('--dm_heads', type=int, default=4)
parser.add_argument('--dm_residua', type=bool, default=True)
parser.add_argument('--dm_graph', type=bool, default=True)
parser.add_argument('--sf', type=bool, default=True)
parser.add_argument('--hops', type=int, default=3)
parser.add_argument('--s_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=256)

# 训练参数
parser.add_argument('--train_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--use_global', type=bool, default=True)
parser.add_argument('--use_local', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=50)

args = parser.parse_args()
set_seed(args.seed)


def split_data_kfold(edge_label_index, edge_label, k_folds=5, seed=2023):
    """
    将边数据划分为K折（保持正负样本比例）
    返回: list of (train_mask, val_mask, test_mask)
    """
    num_edges = edge_label.shape[0]

    # 分离正负样本索引
    pos_indices = np.where(edge_label.numpy() == 1)[0]
    neg_indices = np.where(edge_label.numpy() == 0)[0]

    kf_pos = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    kf_neg = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_masks = []
    for (pos_train_val_idx, pos_test_idx), (neg_train_val_idx, neg_test_idx) in zip(
            kf_pos.split(pos_indices), kf_neg.split(neg_indices)
    ):
        # 分别对正负样本进行train/val划分 (8:2)
        pos_train_size = int(0.8 * len(pos_train_val_idx))
        pos_train_idx = pos_train_val_idx[:pos_train_size]
        pos_val_idx = pos_train_val_idx[pos_train_size:]

        neg_train_size = int(0.8 * len(neg_train_val_idx))
        neg_train_idx = neg_train_val_idx[:neg_train_size]
        neg_val_idx = neg_train_val_idx[neg_train_size:]

        # 创建mask
        train_mask = np.zeros(num_edges, dtype=bool)
        val_mask = np.zeros(num_edges, dtype=bool)
        test_mask = np.zeros(num_edges, dtype=bool)

        # 正样本
        train_mask[pos_indices[pos_train_idx]] = True
        val_mask[pos_indices[pos_val_idx]] = True
        test_mask[pos_indices[pos_test_idx]] = True

        # 负样本
        train_mask[neg_indices[neg_train_idx]] = True
        val_mask[neg_indices[neg_val_idx]] = True
        test_mask[neg_indices[neg_test_idx]] = True

        fold_masks.append((train_mask, val_mask, test_mask))

    return fold_masks



def create_split_data(base_data, edge_label_index, edge_label, train_mask, val_mask, test_mask):
    """根据mask创建划分后的数据"""

    train_data = Data(
        x=base_data.x,
        edge_index=base_data.edge_index,
        edge_label=edge_label[train_mask],
        edge_label_index=edge_label_index[:, train_mask],
        num_nodes=base_data.num_nodes
    )

    val_data = Data(
        x=base_data.x,
        edge_index=base_data.edge_index,
        edge_label=edge_label[val_mask],
        edge_label_index=edge_label_index[:, val_mask],
        num_nodes=base_data.num_nodes
    )

    test_data = Data(
        x=base_data.x,
        edge_index=base_data.edge_index,
        edge_label=edge_label[test_mask],
        edge_label_index=edge_label_index[:, test_mask],
        num_nodes=base_data.num_nodes
    )

    return train_data, val_data, test_data


def train_fold(model, train_data, val_data, args, device, fold_idx):
    """训练一个fold"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    # 准备训练数据
    num_edges = train_data.edge_label.shape[0]
    sample_indices = torch.arange(num_edges)
    links = train_data.edge_label_index.t()  # [num_edges, 2]

    for epoch in range(args.train_epoch):
        # 训练
        model.train()
        total_loss = 0

        # 随机打乱训练样本
        perm = torch.randperm(num_edges)

        # 分批训练
        batch_size = args.batch_size
        for i in range(0, num_edges, batch_size):
            batch_indices = perm[i:i + batch_size]

            optimizer.zero_grad()

            # 确保所有数据在正确的设备上
            train_data = train_data.to(device)
            out = model(train_data, sample_indices.to(device), links.to(device), batch_indices.to(device))

            loss = torch.nn.functional.binary_cross_entropy(
                out.squeeze(), train_data.edge_label[batch_indices].float().to(device)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_data_device = val_data.to(device)
            val_num_edges = val_data.edge_label.shape[0]
            val_sample_indices = torch.arange(val_num_edges).to(device)
            val_links = val_data.edge_label_index.t().to(device)

            # 分批预测以避免内存问题
            val_preds = []
            for i in range(0, val_num_edges, batch_size):
                val_batch_indices = torch.arange(i, min(i + batch_size, val_num_edges)).to(device)
                val_out = model(val_data_device, val_sample_indices, val_links, val_batch_indices)
                val_preds.append(val_out.cpu())

            pred = torch.cat(val_preds).numpy()
            labels = val_data.edge_label.cpu().numpy()
            val_auc = roc_auc_score(labels, pred)

        # 早停
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print(f"Fold {fold_idx}, Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 50 == 0:
            num_batches = (num_edges + batch_size - 1) // batch_size
            print(f"Fold {fold_idx}, Epoch {epoch + 1}: Loss={total_loss / num_batches:.4f}, Val_AUC={val_auc:.4f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_fold(model, test_data, device):
    """评估一个fold"""
    model.eval()

    batch_size = 512
    num_edges = test_data.edge_label.shape[0]
    sample_indices = torch.arange(num_edges).to(device)
    links = test_data.edge_label_index.t().to(device)
    test_data_device = test_data.to(device)

    with torch.no_grad():
        # 分批预测
        preds = []
        for i in range(0, num_edges, batch_size):
            batch_indices = torch.arange(i, min(i + batch_size, num_edges)).to(device)
            out = model(test_data_device, sample_indices, links, batch_indices)
            preds.append(out.cpu())

        pred = torch.cat(preds).numpy()
        labels = test_data.edge_label.cpu().numpy()

        auc = roc_auc_score(labels, pred)
        ap = average_precision_score(labels, pred)

        pred_binary = (pred > 0.5).astype(int)
        acc, sen, pre, spe, f1, mcc = calculate_metrics(labels, pred_binary)

        # 确保所有指标都是标量，使用 .item() 提取numpy标量
        acc = acc.item() if hasattr(acc, 'item') else float(acc)
        sen = sen.item() if hasattr(sen, 'item') else float(sen)
        pre = pre.item() if hasattr(pre, 'item') else float(pre)
        spe = spe.item() if hasattr(spe, 'item') else float(spe)
        f1 = f1.item() if hasattr(f1, 'item') else float(f1)
        mcc = mcc.item() if hasattr(mcc, 'item') else float(mcc)

    return auc, ap, acc, sen, pre, spe, f1, mcc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"数据集: {args.dataset_name}")
    print(f"K折数: {args.k_folds}")

    # 保存目录
    save_root = f"save_kfold/{args.dataset_name}"
    os.makedirs(save_root, exist_ok=True)

    # 加载完整数据集
    print("\n加载数据集...")
    dataset = MyDataset(args)
    base_data = dataset.data

    # 使用RandomLinkSplit添加edge_label和edge_label_index
    from torch_geometric.transforms import RandomLinkSplit
    transform = RandomLinkSplit(
        is_undirected=True,
        num_val=0.1,
        num_test=0.1,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        disjoint_train_ratio=0.3
    )

    # 先进行一次完整的划分以获取带标签的数据结构
    _, full_data_with_labels, _ = transform(base_data)

    # 现在合并所有边用于K折划分
    # 注意：这里我们需要重新构建包含所有边的数据结构
    # 为了简化，我们直接从原始图开始，手动创建正负样本边

    print(f"\n构建带标签的边数据...")

    # 从原始数据构建正样本边
    pos_edges = base_data.edge_index  # [2, num_pos_edges]
    num_pos = pos_edges.shape[1]

    # 生成负样本边（简单的随机采样）
    num_neg = int(num_pos * args.ratio)
    num_nodes = base_data.num_nodes

    # 随机生成负样本边
    neg_edges = torch.zeros(2, num_neg, dtype=torch.long)
    existing_edges = set()
    for i in range(pos_edges.shape[1]):
        existing_edges.add((pos_edges[0, i].item(), pos_edges[1, i].item()))

    neg_count = 0
    attempts = 0
    max_attempts = num_neg * 100  # 防止无限循环

    while neg_count < num_neg and attempts < max_attempts:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        attempts += 1

        # 确保不是已存在的边且不是自环
        if (src, dst) not in existing_edges and src != dst:
            neg_edges[0, neg_count] = src
            neg_edges[1, neg_count] = dst
            neg_count += 1

    print(f"正样本边数: {num_pos}")
    print(f"负样本边数: {neg_count}")

    # 合并正负样本边
    all_edges = torch.cat([pos_edges, neg_edges], dim=1)  # [2, num_total]
    all_labels = torch.cat([
        torch.ones(num_pos),
        torch.zeros(neg_count)
    ])  # [num_total]

    # K折划分
    print(f"\n进行{args.k_folds}折划分...")
    fold_masks = split_data_kfold(all_edges, all_labels, args.k_folds, args.seed)

    all_results = []

    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(fold_masks):
        print(f"\n{'='*60}")
        print(f"第 {fold_idx + 1}/{args.k_folds} 折")
        print(f"{'='*60}")

        # 创建数据划分
        train_data, val_data, test_data = create_split_data(
            base_data, all_edges, all_labels,
            train_mask, val_mask, test_mask
        )

        print(f"训练集: {train_data.edge_label.shape[0]} 条边 (正样本: {train_data.edge_label.sum().item():.0f})")
        print(f"验证集: {val_data.edge_label.shape[0]} 条边 (正样本: {val_data.edge_label.sum().item():.0f})")
        print(f"测试集: {test_data.edge_label.shape[0]} 条边 (正样本: {test_data.edge_label.sum().item():.0f})")

        # 创建模型
        model = GlobalLocalFusionTaskModelGate(
            args,
            base_data.x.shape[1]
        ).to(device)
        
        # 创建数据加载器
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
        
        # 训练
        print(f"\n开始训练...")
        model = train_fold(model, train_data, val_data, args, device, fold_idx + 1)

        # 测试
        print(f"\n测试...")
        metrics = evaluate_fold(model, test_data, device)
        auc, ap, acc, sen, pre, spe, f1, mcc = metrics
        
        print(f"\n第 {fold_idx + 1} 折结果:")
        print(f"  AUC:  {auc:.4f}")
        print(f"  AP:   {ap:.4f}")
        print(f"  ACC:  {acc:.4f}")
        print(f"  SEN:  {sen:.4f}")
        print(f"  PRE:  {pre:.4f}")
        print(f"  SPE:  {spe:.4f}")
        print(f"  F1:   {f1:.4f}")
        print(f"  MCC:  {mcc:.4f}")
        
        all_results.append({
            'fold': fold_idx + 1,
            'auc': auc, 'ap': ap, 'acc': acc,
            'sen': sen, 'pre': pre, 'spe': spe,
            'f1': f1, 'mcc': mcc
        })
        
        # 保存模型
        model_path = os.path.join(save_root, f'best_model_fold_{fold_idx + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"\n模型已保存: {model_path}")
    
    # 最终统计
    print(f"\n{'='*60}")
    print("K折交叉验证最终结果 (Mean ± Std)")
    print(f"{'='*60}")
    
    metrics_names = ['auc', 'ap', 'acc', 'sen', 'pre', 'spe', 'f1', 'mcc']
    for metric in metrics_names:
        values = [r[metric] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper():>6}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.loc[len(results_df)] = ['Mean'] + [np.mean([r[m] for r in all_results]) for m in metrics_names]
    results_df.loc[len(results_df)] = ['Std'] + [np.std([r[m] for r in all_results]) for m in metrics_names]
    
    csv_path = os.path.join(save_root, f'kfold_{args.k_folds}fold_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n结果已保存: {csv_path}")


if __name__ == '__main__':
    main()
