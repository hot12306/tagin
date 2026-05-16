# integrate_model_comparison.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import argparse
import json


def load_best_predictions(model_name, dataset_name="drugbank", save_root_base="save"):
    """加载指定模型在DrugBank数据集上的最佳预测结果"""
    # 每个模型的结果存储在save/model_name/目录下
    save_root = os.path.join(save_root_base, model_name)

    # 首先尝试加载测试预测结果文件
    predictions_file = os.path.join(save_root, f'test_predictions_{dataset_name}.npz')
    metadata_file = os.path.join(save_root, f'test_predictions_{dataset_name}_metadata.json')

    # 如果测试预测结果不存在，尝试加载最佳预测结果文件
    if not os.path.exists(predictions_file):
        predictions_file = os.path.join(save_root, f'best_predictions_{dataset_name}.npz')
        metadata_file = os.path.join(save_root, f'best_predictions_{dataset_name}_metadata.json')

    if not os.path.exists(predictions_file):
        print(f"未找到 {model_name} 在 {dataset_name} 上的预测结果文件: {predictions_file}")
        return None, None

    try:
        # 加载预测结果
        data = np.load(predictions_file)
        print(f"{model_name} 文件包含的键: {list(data.keys())}")

        # 尝试不同的键名组合
        if 'y_true' in data and 'y_scores' in data:
            y_true = data['y_true']
            y_scores = data['y_scores']
        elif 'y_true' in data and 'y_pred' in data:
            y_true = data['y_true']
            y_scores = data['y_pred']
        elif 'true' in data and 'scores' in data:
            y_true = data['true']
            y_scores = data['scores']
        elif 'labels' in data and 'scores' in data:
            y_true = data['labels']
            y_scores = data['scores']
        elif 'y_true' in data and 'probs' in data:
            y_true = data['y_true']
            y_scores = data['probs']
        else:
            print(f"{model_name} 文件格式不支持，包含的键: {list(data.keys())}")
            return None, None

        # 加载元数据
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        print(
            f"加载 {model_name} 在 {dataset_name} 上的预测结果 (AUC: {metadata.get('auc', metadata.get('best_auc', 'N/A'))})")
        return (y_true, y_scores), metadata
    except Exception as e:
        print(f"加载 {model_name} 在 {dataset_name} 上的预测结果失败: {e}")
        return None, None


def plot_model_comparison_curves(model_predictions, model_names, model_metadata, output_dir="model_comparison_results"):
    """绘制不同模型在DrugBank数据集上的ROC和PR曲线比较"""
    os.makedirs(output_dir, exist_ok=True)

    # 定义颜色（使用更鲜明的颜色）
    colors = ['#FF8C00', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # 统一使用实线
    linestyle = '-'

    # 绘制ROC曲线
    plt.figure(figsize=(12, 8))

    valid_models = []
    roc_auc_values = []  # 存储ROC AUC值

    for idx, (model_name, predictions_data) in enumerate(zip(model_names, model_predictions)):
        if predictions_data is None:
            continue

        y_true, y_scores = predictions_data
        metadata = model_metadata[idx] if idx < len(model_metadata) else {}

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_auc_values.append(roc_auc)  # 保存ROC AUC值

        # 绘制曲线
        color = colors[idx % len(colors)]
        # 将my_model显示为TAGIN-DTI
        display_name = 'TAGIN-DTI' if model_name.lower() == 'my_model' else model_name.upper()
        label = f'{display_name}'
        # 检查元数据中的AUC字段
        auc_value = metadata.get('auc', metadata.get('best_auc', roc_auc))
        label += f' (AUC-ROC = {auc_value:.4f})'

        plt.plot(fpr, tpr, color=color, lw=2.5, linestyle=linestyle, label=label)
        valid_models.append(model_name)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random (AUC-ROC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves Comparison: Different Models on DrugBank Dataset', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 保存图像
    roc_filename = os.path.join(output_dir, 'model_comparison_roc_curves.png')
    plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线比较图已保存至: {roc_filename}")

    # 绘制PR曲线
    plt.figure(figsize=(12, 8))

    pr_auc_values = []  # 存储PR AUC值
    for idx, (model_name, predictions_data) in enumerate(
            zip(valid_models, [p for p in model_predictions if p is not None])):
        y_true, y_scores = predictions_data
        metadata = model_metadata[idx] if idx < len(model_metadata) else {}

        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        pr_auc_values.append(pr_auc)  # 保存PR AUC值

        # 绘制曲线
        color = colors[idx % len(colors)]
        # 将my_model显示为TAGIN-DTI
        display_name = 'TAGIN-DTI' if model_name.lower() == 'my_model' else model_name.upper()
        label = f'{display_name}'
        # 检查元数据中的AP字段，优先使用AP值
        ap_value = metadata.get('ap', metadata.get('auc_pr', metadata.get('best_ap', pr_auc)))
        label += f' (AUC-PR = {ap_value:.4f})'

        plt.plot(recall, precision, color=color, lw=2.5, linestyle=linestyle, label=label)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves Comparison: Different Models on DrugBank Dataset', fontsize=16, pad=20)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 保存图像
    pr_filename = os.path.join(output_dir, 'model_comparison_pr_curves.png')
    plt.savefig(pr_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR曲线比较图已保存至: {pr_filename}")


def print_model_comparison_metrics(model_predictions, model_names, model_metadata):
    """打印不同模型在DrugBank数据集上的评估指标比较"""
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("\n模型在DrugBank数据集上的性能比较:")
    print("=" * 60)

    results = []
    for model_name, predictions_data, metadata in zip(model_names, model_predictions, model_metadata):
        if predictions_data is None:
            print(f"{model_name.upper()}: 无数据")
            continue

        y_true, y_scores = predictions_data

        # 计算指标
        roc_auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        # 将my_model显示为TAGIN-DTI
        display_name = 'TAGIN-DTI' if model_name.lower() == 'my_model' else model_name.upper()
        print(f"{display_name}:")
        # 检查元数据中的AUC字段
        auc_value = metadata.get('auc', metadata.get('best_auc', roc_auc))
        print(f"  AUC-ROC: {auc_value:.4f}")
        results.append((model_name, auc_value, ap))
        print(f"  AUC-PR:  {ap:.4f}")
        print()

    # 按AUC-ROC排序并显示排名
    if results:
        print("模型性能排名 (按AUC-ROC):")
        print("-" * 30)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        for i, (model_name, auc_roc, auc_pr) in enumerate(sorted_results, 1):
            # 将my_model显示为TAGIN-DTI
            display_name = 'TAGIN-DTI' if model_name.lower() == 'my_model' else model_name.upper()
            print(f"{i}. {display_name}: AUC-ROC = {auc_roc:.4f}, AUC-PR = {auc_pr:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['deepdta', 'gat', 'iifdti', 'transformercpi', 'my_model'],
                        help='Models to compare (should match the directory names in save folder)')
    parser.add_argument('--dataset', type=str, default='drugbank',
                        help='Dataset to compare models on')
    parser.add_argument('--save_root', type=str, default='save',
                        help='Root directory where results are saved')
    parser.add_argument('--output_dir', type=str, default='model_comparison_results',
                        help='Directory to save comparison results')

    args = parser.parse_args()

    # 加载所有模型在DrugBank数据集上的最佳预测结果
    model_predictions = []
    model_metadata = []
    valid_model_names = []

    for model_name in args.models:
        predictions_data, metadata = load_best_predictions(model_name, args.dataset, args.save_root)
        if predictions_data is not None:
            model_predictions.append(predictions_data)
            model_metadata.append(metadata)
            valid_model_names.append(model_name)

    if not model_predictions:
        print("未找到任何模型的预测结果，退出")
        exit(1)

    # 打印模型比较指标
    print_model_comparison_metrics(model_predictions, valid_model_names, model_metadata)

    # 绘制模型比较曲线
    plot_model_comparison_curves(model_predictions, valid_model_names, model_metadata, args.output_dir)

    print(f"\n模型比较结果已保存至: {args.output_dir}")
