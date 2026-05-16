# main.py
import os
import torch
import argparse
import numpy as np
from global_local_fusion_gate import GlobalLocalFusionTaskModelGate
from utile import set_seed, get_data, get_loaders, train_func, evaluate_model_on_dataset
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json

warnings.filterwarnings("ignore", category=UserWarning)

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())  # 需输出True
print("CUDA版本:", torch.version.cuda)  # 需显示具体版本（如11.7）
print("当前GPU设备:", torch.cuda.get_device_name(0))  # 应显示显卡型号

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help="random seed of dataset and model")
parser.add_argument('--dataset_name', type=str, default='ttd',
                    choices=['bindingdB', 'ttd', 'drugbank'])
parser.add_argument('--num_test', type=float, default=0.2, help='ratio of test datasets')
parser.add_argument('--ratio', type=float, default=1, help='ratio of positive samples and negative samples')
parser.add_argument('--task', type=str, default='SP', choices=['SD', 'ST', 'SP'])

parser.add_argument('--dm', type=bool, default=True, help="Whether to use diffusion model")
parser.add_argument('--dm_layers', type=int, default=2, help="The number of layers in the diffusion model")
parser.add_argument('--dm_heads', type=int, default=4, help="The number of heads in the diffusion model")
parser.add_argument('--dm_residua', type=bool, default=True, help="Whether to use residua in the diffusion model")
parser.add_argument('--dm_graph', type=bool, default=True, help="Whether to use graph in the diffusion model")

parser.add_argument('--sf', type=bool, default=True, help="Whether to use subgraph features")
parser.add_argument('--hops', type=int, default=3, help="k-hop subgraph[1,2,3]")
parser.add_argument('--s_dim', type=int, default=128, help="feature dimension of subgraph")

parser.add_argument('--hidden_dim', type=int, default=256)

parser.add_argument('--train_times', type=int, default=10, help='number of training times')
parser.add_argument('--train_epoch', type=int, default=1000, help='number of training epoch')
parser.add_argument('--batch_size', type=int, default=512, help='batch size of dataset')
parser.add_argument('--use_global', type=bool, default=True)
parser.add_argument('--use_local', type=bool, default=False)

# 新增参数：是否保存最佳模型的预测结果用于后续整合
parser.add_argument('--save_best_predictions', type=bool, default=True,
                    help='Whether to save best model predictions for later integration')
args = parser.parse_args()
set_seed(args.seed)


def print_result(result):
    metrics = ['auc', 'ap', 'acc', 'sen', 'pre', 'spe', 'F1', 'mcc']
    metric_values = [[] for _ in range(len(metrics))]
    for i in result:
        for j, val in enumerate(i):
            metric_values[j].append(float(val[-6:]))
    metric_values = [np.array(m) for m in metric_values]
    formatted_metrics = []
    for metric, values in zip(metrics, metric_values):
        mean = "{:.4f}".format(values.mean())
        std = "{:.4f}".format(np.std(values))
        formatted_metrics.append(f"{metric}: {mean} ± {std}")
    print(*formatted_metrics)


save_root = f"save/{args.dataset_name}"
os.makedirs(save_root, exist_ok=True)
# file_name = f'{args.dataset_name.upper()}'+'-num_perm-128'+'.csv'
file_name = f'{args.dataset_name.upper()}.csv'
file_dir = save_root + '/' + file_name


# 在main.py中添加以下函数
def save_result(result, filename=file_dir):
    """保存所有训练结果到CSV文件"""
    import csv
    from datetime import datetime

    # 准备表头
    headers = ['Run', 'auc', 'ap', 'acc', 'sen', 'pre', 'spe', 'F1', 'mcc']

    # 准备数据行
    rows = []
    for i, run_result in enumerate(result):
        row = [i + 1] + [float(val[-6:]) for val in run_result]
        rows.append(row)

    # 添加统计行
    stats = ['Mean'] + list(np.mean(rows, axis=0)[1:])
    stds = ['Std'] + list(np.std(rows, axis=0)[1:])

    # 写入文件
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(['Dataset:', args.dataset_name])
        writer.writerow(['Parameters:', str(vars(args))])
        writer.writerow([])  # 空行分隔
        writer.writerow(headers)
        writer.writerows(rows)
        writer.writerow([])  # 空行分隔
        writer.writerow(stats)
        writer.writerow(stds)


# 新增：保存最佳模型的函数
def save_best_model(model, filename):
    """保存最佳模型权重"""
    torch.save(model.state_dict(), filename)
    print(f"最佳模型已保存至: {filename}")


# 新增：保存最佳模型预测结果的函数
def save_best_predictions(y_true, y_scores, dataset_name, best_auc, save_root):
    """保存最佳模型的预测结果用于后续整合"""
    predictions_file = os.path.join(save_root, f'best_predictions_{dataset_name}.npz')
    metadata_file = os.path.join(save_root, f'best_predictions_{dataset_name}_metadata.json')

    # 保存预测结果
    np.savez(predictions_file, y_true=y_true, y_scores=y_scores)

    # 保存元数据
    metadata = {
        'dataset_name': dataset_name,
        'best_auc': float(best_auc),
        'num_samples': len(y_true)
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"最佳模型预测结果已保存至: {predictions_file}")
    print(f"元数据已保存至: {metadata_file}")


if __name__ == '__main__':
    # 将需要隔离的代码段移入此处
    args = parser.parse_args()
    set_seed(args.seed)
    all_result = []
    c = 1

    # 记录所有训练中的最佳AUC得分
    best_auc_scores = []

    # 记录全局最佳预测结果
    global_best_auc = 0
    global_best_predictions = None

    for _ in range(args.train_times):
        print("第", c, "次训练开始")
        c += 1
        dataset, splits = get_data(args)
        train_loader, val_loader, test_loader = get_loaders(args, splits)

        # model = Model(args, dataset.num_features).cuda()
        model = GlobalLocalFusionTaskModelGate(args, dataset.num_features).cuda()
        # model = SimplifiedImprovedModel(args, dataset.num_features).cuda()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0005)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # 修改调用方式以获取预测值
        result, y_true, y_scores = train_func(args, train_loader, val_loader, test_loader, model, optimizer, loss_fn)
        all_result.append(result)

        # 提取本次训练的最佳验证集AUC得分（假设result格式中auc在第一位）
        # 根据print_result函数推断，每个result元素应该是一个包含各指标字符串的列表
        # 我们需要提取auc值（第一个元素）
        best_val_auc = float(result[0][-6:])  # 假设auc值在字符串的最后6位
        best_auc_scores.append(best_val_auc)

        # 检查是否为全局最佳模型
        if best_val_auc > global_best_auc:
            global_best_auc = best_val_auc
            global_best_predictions = (y_true, y_scores)

        # 保存每次训练的最佳模型
        model_save_path = os.path.join(save_root, f'best_model_run_{c - 1}.pth')
        save_best_model(model, model_save_path)

    print_result(all_result)
    save_result(all_result)

    # 找出所有训练中的最佳模型并保存其副本
    overall_best_idx = np.argmax(best_auc_scores)
    best_model_path = os.path.join(save_root, f'best_model_run_{overall_best_idx + 1}.pth')
    if os.path.exists(best_model_path):
        import shutil

        final_best_path = os.path.join(save_root, 'best_model_overall.pth')
        shutil.copy(best_model_path, final_best_path)
        print(f"整体最佳模型(AUC={best_auc_scores[overall_best_idx]:.4f})已复制到: {final_best_path}")

    # 保存全局最佳模型的预测结果
    if args.save_best_predictions and global_best_predictions is not None:
        save_best_predictions(global_best_predictions[0], global_best_predictions[1],
                              args.dataset_name, global_best_auc, save_root)
