# utile.py
import tqdm
import torch
import random
import numpy as np
import tensorflow as tf
from dataset import MyDataset, HashDataset
from torch.utils.data import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tf.random.set_seed(seed)


def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)

    # 修复MCC计算中的数值溢出问题
    denominator = np.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))
    if denominator == 0:
        mcc = 0
    else:
        mcc = (TP * TN - FP * FN) / denominator

    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data(args):
    dataset = MyDataset(args)

    # 调整数据划分参数 (保证训练边数量)
    transform = RandomLinkSplit(
        is_undirected=True,
        num_val=0.1,
        num_test=0.1,
        add_negative_train_samples=True,  # 开启自动负采样
        neg_sampling_ratio=1.0,
        disjoint_train_ratio=0.3
    )

    # 重新设计数据划分流程
    train_data, val_data, test_data = transform(dataset.data)

    # 验证边数量有效性
    print(f"训练边: {train_data.edge_label.sum().item()}")
    print(f"验证边: {val_data.edge_label.sum().item()}")
    print(f"测试边: {test_data.edge_label.sum().item()}")

    if args.task == 'SD':
        print('SD')

        test_drug_set = set(test_data['edge_label_index'][1].tolist())
        train_drug_list = train_data['edge_label_index'][1].tolist()
        edge_label_index = train_data['edge_label_index'].tolist()
        edge_label = train_data['edge_label'].tolist()
        for drug in train_drug_list:
            if drug in test_drug_set:
                index_to_remove = train_drug_list.index(drug)
                edge_label_index = [sl[:index_to_remove] + sl[index_to_remove + 1:] for sl in edge_label_index]
                edge_label = edge_label[:index_to_remove] + edge_label[index_to_remove + 1:]
        train_data['edge_label_index'] = torch.LongTensor(edge_label_index)
        train_data['edge_label'] = torch.Tensor(edge_label)

        # 添加过滤前检查
        original_train_edges = len(train_data.edge_label)
        test_drug_set = set(test_data['edge_label_index'][1].tolist())

        # 使用集合运算加速过滤
        overlap_drugs = set(train_drug_list) & test_drug_set
        mask = torch.tensor([drug not in overlap_drugs for drug in train_drug_list])

        train_data.edge_label_index = train_data.edge_label_index[:, mask]
        train_data.edge_label = train_data.edge_label[mask]

        print(f"过滤后保留训练边: {len(train_data.edge_label)}/{original_train_edges}")
    elif args.task == 'ST':
        print('ST')
        test_target_set = set(test_data['edge_label_index'][0].tolist())
        train_target_list = train_data['edge_label_index'][0].tolist()
        edge_label_index = train_data['edge_label_index'].tolist()
        edge_label = train_data['edge_label'].tolist()
        for drug in train_target_list:
            if drug in test_target_set:
                index_to_remove = train_target_list.index(drug)
                edge_label_index = [sl[:index_to_remove] + sl[index_to_remove + 1:] for sl in edge_label_index]
                edge_label = edge_label[:index_to_remove] + edge_label[index_to_remove + 1:]
        train_data['edge_label_index'] = torch.LongTensor(edge_label_index)
        train_data['edge_label'] = torch.Tensor(edge_label)
    else:
        print("SP")

    splits = {'train': train_data, 'val': test_data, 'test': test_data}
    save_test_pairs(args, test_data)
    return dataset, splits


def save_test_pairs(args, test_data):
    """保存测试集的药物-靶标对"""
    import pandas as pd
    import os

    # 获取测试集的边索引和标签
    edge_index = test_data['edge_label_index']  # [2, num_edges]
    labels = test_data['edge_label']  # [num_edges]

    # 分离正样本和负样本
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_edges = edge_index[:, pos_mask].t().numpy()  # [num_pos, 2]
    neg_edges = edge_index[:, neg_mask].t().numpy()  # [num_neg, 2]

    # 获取节点ID映射（需要从dataset中获取）
    # 这里需要根据你的实际情况调整
    save_dir = f"save/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 保存为正样本和负样本文件
    np.savez(os.path.join(save_dir, 'test_pairs.npz'),
             pos_edges=pos_edges,
             neg_edges=neg_edges,
             labels=labels.numpy(),
             edge_index=edge_index.numpy())

    print(f"测试集对已保存至: {save_dir}/test_pairs.npz")
    print(f"正样本数: {len(pos_edges)}, 负样本数: {len(neg_edges)}")

def get_pos_neg_edges(data):
    pos_edges = data['edge_label_index'][:, data['edge_label'] == 1].t()
    neg_edges = data['edge_label_index'][:, data['edge_label'] == 0].t()
    return pos_edges, neg_edges


def get_hashed_train_val_test_datasets(args, train_data, val_data, test_data):
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    train_dataset = HashDataset(args, train_data, pos_train_edge, neg_train_edge)
    val_dataset = HashDataset(args, val_data, pos_val_edge, neg_val_edge)
    test_dataset = HashDataset(args, test_data, pos_test_edge, neg_test_edge)
    return train_dataset, val_dataset, test_dataset


def get_loaders(args, splits):
    train_data, val_data, test_data = splits['train'], splits['val'], splits['test']
    train_dataset, val_dataset, test_dataset = get_hashed_train_val_test_datasets(args, train_data, val_data, test_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader, test_loader


def train_model(data_loader, model, optimizer, loss_fn):
    model.train()
    data = data_loader.dataset
    labels = torch.tensor(data.labels)
    sample_indices = torch.randperm(len(labels))[:len(labels)]
    links = data.links[sample_indices]
    labels = labels[sample_indices]
    total_loss = 0
    for batch_count, indices in enumerate(DataLoader(range(len(links)), batch_size=512, shuffle=True)):
        optimizer.zero_grad()
        logits = model(data, sample_indices, links, indices)
        loss = loss_fn(logits.view(-1), labels[indices].squeeze(0).cuda().to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss


@torch.no_grad()
def test_model(data_loader, model):
    model.eval()
    data = data_loader.dataset
    labels = torch.tensor(data.labels)
    sample_indices = torch.arange(0, len(labels))
    links = data.links[sample_indices]
    preds = []
    for batch_count, indices in enumerate(DataLoader(range(len(links)), batch_size=8192, shuffle=False)):
        # logits = model(data, sample_indices, links, indices, target='test')
        logits = model(data, sample_indices, links, indices)
        preds.append(logits.view(-1).cpu())
    pred = torch.cat(preds)
    labels = labels[:len(pred)]

    # 返回原始预测分数和标签，而非格式化字符串
    return pred.numpy(), labels.numpy()


def train_func(args, train_loader, val_loader, test_loader, model, optimizer, loss_fn):
    early_stop = 0
    best_auc = 0
    output_result = []
    best_y_true = None
    best_y_scores = None

    for epoch in tqdm.tqdm(range(args.train_epoch)):
        early_stop += 1
        loss = train_model(train_loader, model, optimizer, loss_fn)

        # 获取测试集预测结果
        y_scores, y_true = test_model(test_loader, model)

        # 计算评估指标
        AUC = roc_auc_score(y_true, y_scores)
        AP = average_precision_score(y_true, y_scores)

        # 二值化预测结果计算其他指标
        temp = torch.tensor(y_scores)
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        accuracy, sensitivity, precision, specificity, F1_score, mcc = calculate_metrics(y_true, temp.numpy())

        result = ['AUC:{:.4f}'.format(AUC), 'AP:{:.4f}'.format(AP),
                  'acc:{:.4f}'.format(accuracy.item()), 'sen:{:.4f}'.format(sensitivity.item()),
                  'pre:{:.4f}'.format(precision.item()), 'spe:{:.4f}'.format(specificity.item()),
                  'f1:{:.4f}'.format(F1_score.item()), 'mcc:{:.4f}'.format(mcc.item())]

        print(result)

        if AUC > best_auc:
            early_stop = 0
            best_auc = AUC
            output_result = result
            # 保存最佳预测结果用于绘图
            best_y_true = y_true
            best_y_scores = y_scores

        if early_stop == 10:
            print("Early Stopping", output_result)
            break

    # 返回结果以及用于绘制曲线的数据
    return output_result, best_y_true, best_y_scores


# 新增：评估模型在特定数据集上的性能
@torch.no_grad()
def evaluate_model_on_dataset(data_loader, model):
    """评估模型在特定数据集上的性能，返回预测值和真实标签"""
    model.eval()
    data = data_loader.dataset
    labels = torch.tensor(data.labels)
    sample_indices = torch.arange(0, len(labels))
    links = data.links[sample_indices]
    preds = []
    for batch_count, indices in enumerate(DataLoader(range(len(links)), batch_size=8192, shuffle=False)):
        logits = model(data, sample_indices, links, indices)
        preds.append(logits.view(-1).cpu())
    pred = torch.cat(preds)
    labels = labels[:len(pred)]

    return pred.numpy(), labels.numpy()
