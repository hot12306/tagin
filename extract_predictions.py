# extract_complete_predictions.py - 提取包含药物ID、靶标ID、真实值、预测值、预测标签的完整预测结果
import numpy as np
import pandas as pd
import os
import sys
import json
import torch
import argparse

sys.path.append(os.path.dirname(__file__))

from dataset import MyDataset
from utile import get_data


def extract_complete_predictions(dataset_name_='bindingdB', task_='SP', threshold=0.5):
    """
    提取完整的预测结果，包含：
    1. 药物ID (drug_id)
    2. 靶标ID (target_id)
    3. 真实值 (true_label)
    4. 预测概率 (predicted_probability)
    5. 预测标签 (predicted_label)

    参数:
        dataset_name_: 数据集名称 ('bindingdB', 'drugbank', 'ttd')
        task_: 数据划分方式 ('SP', 'SD', 'ST')
        threshold: 分类阈值，默认0.5
    """

    print("=" * 80)
    print(f"提取 {dataset_name_} 数据集的完整预测结果 (task={task_})")
    print("=" * 80)

    # ========================
    # 1. 加载预测结果
    # ========================
    save_dir = f'save/{dataset_name_}'
    predictions_file = os.path.join(save_dir, f'best_predictions_{dataset_name_}.npz')
    metadata_file = os.path.join(save_dir, f'best_predictions_{dataset_name_}_metadata.json')

    if not os.path.exists(predictions_file):
        print(f"❌ 错误: 预测文件不存在 {predictions_file}")
        return None

    print(f"\n📂 加载预测文件: {predictions_file}")
    pred_data = np.load(predictions_file)

    y_true = pred_data['y_true']
    y_scores = pred_data['y_scores']

    # 加载元数据
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"📊 元数据:")
        print(f"   数据集: {metadata['dataset_name']}")
        print(f"   最佳AUC: {metadata['best_auc']:.4f}")
        print(f"   样本数: {metadata['num_samples']}")

    print(f"\n✅ 预测数据统计:")
    print(f"   总样本数: {len(y_true)}")
    print(f"   正样本数: {sum(y_true == 1)}")
    print(f"   负样本数: {sum(y_true == 0)}")
    print(f"   预测概率范围: [{y_scores.min():.4f}, {y_scores.max():.4f}]")
    print(f"   预测概率均值: {y_scores.mean():.4f}")

    # ========================
    # 2. 加载数据集以获取ID映射和测试集边信息
    # ========================
    print(f"\n🔄 加载数据集以获取ID映射...")

    class Args:
        dataset_name = dataset_name_
        seed = 2023
        task = task_
        num_test = 0.2
        ratio = 1

    args = Args()

    try:
        dataset, splits = get_data(args)
        test_data = splits['test']
        print(f"✅ 数据集加载成功 (task={task_})")

        data_metadata = dataset.data.metadata
        drug_ids = data_metadata['drug_ids']
        target_ids = data_metadata['target_ids']
        num_drugs = data_metadata['num_drugs']
        num_proteins = data_metadata['num_proteins']

        print(f"   药物数量: {num_drugs}")
        print(f"   靶标数量: {num_proteins}")

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    edge_index = test_data['edge_label_index']
    labels = test_data['edge_label'].numpy()

    print(f"   测试集边数: {len(labels)}")
    print(f"   测试集正样本: {sum(labels == 1)}")
    print(f"   测试集负样本: {sum(labels == 0)}")

    if len(y_true) != len(labels):
        print(f"⚠️  警告: 预测结果数量 ({len(y_true)}) 与测试集边数 ({len(labels)}) 不一致")

    # ========================
    # 3. 转换节点索引为真实ID
    # ========================
    print(f"\n🔍 正在转换节点索引为真实ID...")

    def convert_edge_to_ids(node1_idx, node2_idx, drug_ids, target_ids, num_drugs):
        """将节点索引转换为药物和靶标ID"""
        if node1_idx < num_drugs:
            # node1 是药物，node2 是靶标
            drug_id = drug_ids[node1_idx]
            target_id = target_ids[node2_idx - num_drugs]
        else:
            # node1 是靶标，node2 是药物
            drug_id = drug_ids[node2_idx]
            target_id = target_ids[node1_idx - num_drugs]

        return drug_id, target_id

    # 处理所有边
    all_records = []

    for i in range(len(y_true)):
        # 从测试集中获取对应的边索引
        if i < edge_index.shape[1]:
            node1_idx = int(edge_index[0, i])
            node2_idx = int(edge_index[1, i])
        else:
            # 如果超出范围，跳过
            print(f"⚠️  警告: 第 {i} 条边超出测试集范围，跳过")
            continue

        try:
            drug_id, target_id = convert_edge_to_ids(
                node1_idx, node2_idx, drug_ids, target_ids, num_drugs
            )

            true_label = int(y_true[i])
            predicted_prob = float(y_scores[i])
            predicted_label = 1 if predicted_prob >= threshold else 0

            all_records.append({
                'drug_id': drug_id,
                'target_id': target_id,
                'true_label': true_label,
                'predicted_probability': round(predicted_prob, 6),
                'predicted_label': predicted_label
            })
        except Exception as e:
            print(f"⚠️  警告: 第 {i} 条边转换失败: {e}")
            continue

    print(f"✅ 成功转换 {len(all_records)} 条边的预测结果")

    # ========================
    # 4. 创建DataFrame并保存
    # ========================
    df = pd.DataFrame(all_records)

    # 确保列顺序正确
    df = df[['drug_id', 'target_id', 'true_label', 'predicted_probability', 'predicted_label']]

    # 保存完整预测结果
    output_csv = os.path.join(save_dir, 'complete_predictions.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 完整预测结果已保存至: {output_csv}")

    # 分别保存正负样本的预测
    pos_df = df[df['true_label'] == 1]
    neg_df = df[df['true_label'] == 0]

    pos_csv = os.path.join(save_dir, 'positive_predictions_complete.csv')
    neg_csv = os.path.join(save_dir, 'negative_predictions_complete.csv')

    pos_df.to_csv(pos_csv, index=False, encoding='utf-8-sig')
    neg_df.to_csv(neg_csv, index=False, encoding='utf-8-sig')

    print(f"💾 正样本预测: {pos_csv} (共 {len(pos_df)} 条)")
    print(f"💾 负样本预测: {neg_csv} (共 {len(neg_df)} 条)")

    # ========================
    # 5. 显示统计信息
    # ========================
    print("\n" + "=" * 80)
    print("📊 统计信息")
    print("=" * 80)
    print(f"总预测数量: {len(df)}")
    print(f"正样本数量: {len(pos_df)}")
    print(f"负样本数量: {len(neg_df)}")
    print(f"正负样本比例: {len(pos_df) / len(neg_df):.2f}")

    # 计算准确率
    accuracy = (df['true_label'] == df['predicted_label']).mean()
    print(f"\n✅ 分类准确率 (threshold={threshold}): {accuracy:.4f}")

    # 计算TP, TN, FP, FN
    TP = ((df['true_label'] == 1) & (df['predicted_label'] == 1)).sum()
    TN = ((df['true_label'] == 0) & (df['predicted_label'] == 0)).sum()
    FP = ((df['true_label'] == 0) & (df['predicted_label'] == 1)).sum()
    FN = ((df['true_label'] == 1) & (df['predicted_label'] == 0)).sum()

    print(f"\n混淆矩阵:")
    print(f"   TP (真阳性): {TP}")
    print(f"   TN (真阴性): {TN}")
    print(f"   FP (假阳性): {FP}")
    print(f"   FN (假阴性): {FN}")

    # 唯一药物和靶标数量
    unique_drugs = df['drug_id'].nunique()
    unique_targets = df['target_id'].nunique()
    print(f"\n唯一药物数量: {unique_drugs}")
    print(f"唯一靶标数量: {unique_targets}")

    # 预测概率分布
    print("\n" + "=" * 80)
    print("📊 预测概率分布:")
    print("=" * 80)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(df['predicted_probability'], bins=bins)
    for i in range(len(bins) - 1):
        print(f"   [{bins[i]:.1f} - {bins[i + 1]:.1f}]: {hist[i]} 个样本")

    # ========================
    # 6. 显示示例数据
    # ========================
    print("\n" + "=" * 80)
    print("📋 前10个预测结果示例:")
    print("=" * 80)
    print(df.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("✅ 提取完成！")
    print("=" * 80)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='提取完整预测结果（含真实ID）')
    parser.add_argument('--dataset_name', type=str, default='bindingdB',
                        choices=['bindingdB', 'drugbank', 'ttd'])
    parser.add_argument('--task', type=str, default='SP',
                        choices=['SP', 'SD', 'ST'],
                        help='数据划分方式: SP=随机, SD=药物冷启动, ST=靶标冷启动')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='分类阈值')
    args_extract = parser.parse_args()

    df = extract_complete_predictions(
        dataset_name_=args_extract.dataset_name,
        task_=args_extract.task,
        threshold=args_extract.threshold
    )
