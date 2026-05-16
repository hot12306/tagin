# extract_test_pairs_with_ids.py - 提取测试集对并转换为真实ID
import numpy as np
import pandas as pd
import os
import sys
import torch
from torch_geometric.transforms import RandomLinkSplit

# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

from my_model.dataset import MyDataset


def extract_test_pairs_with_real_ids(dataset_name='bindingdB'):
    """从 test_pairs.npz 提取数据并转换为真实的 DrugBank ID 和 UniProt ID"""

    print("=" * 80)
    print(f"开始提取 {dataset_name} 数据集的测试集对")
    print("=" * 80)

    # 1. 加载 npz 文件
    save_dir = f'save/{dataset_name}'
    npz_file = os.path.join(save_dir, 'test_pairs.npz')

    if not os.path.exists(npz_file):
        print(f"❌ 错误: 文件不存在 {npz_file}")
        return

    print(f"\n📂 加载文件: {npz_file}")
    data = np.load(npz_file)

    pos_edges = data['pos_edges']  # [862, 2]
    neg_edges = data['neg_edges']  # [862, 2]
    labels = data['labels']  # [1724]
    edge_index = data['edge_index']  # [2, 1724]

    print(f"✅ 数据统计:")
    print(f"   正样本边: {pos_edges.shape[0]} 条")
    print(f"   负样本边: {neg_edges.shape[0]} 条")
    print(f"   总标签数: {labels.shape[0]}")
    print(f"   总边索引数: {edge_index.shape[1]}")

    # 2. 重新加载数据集以获取ID映射
    print(f"\n🔄 加载数据集以获取ID映射...")

    class Args:
        dataset_name = dataset_name
        seed = 2023

    args = Args()

    try:
        dataset = MyDataset(args)
        print(f"✅ 数据集加载成功")

        # 获取元数据
        metadata = dataset.data.metadata
        drug_ids = metadata['drug_ids']  # 药物ID列表 (DBxxxxx)
        target_ids = metadata['target_ids']  # 靶标ID列表 (UniProt ID)
        num_drugs = metadata['num_drugs']
        num_proteins = metadata['num_proteins']

        print(f"   药物数量: {num_drugs}")
        print(f"   靶标数量: {num_proteins}")
        print(f"   前5个药物ID: {drug_ids[:5]}")
        print(f"   前5个靶标ID: {target_ids[:5]}")

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # 3. 转换节点索引为真实ID
    print(f"\n🔍 正在转换节点索引为真实ID...")

    def convert_edge_to_ids(node1_idx, node2_idx, drug_ids, target_ids, num_drugs):
        """将节点索引转换为药物和靶标ID"""

        # 判断哪个是药物，哪个是靶标
        # 药物索引范围: [0, num_drugs-1]
        # 靶标索引范围: [num_drugs, num_drugs+num_proteins-1]

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

    # 合并正负样本的边索引
    all_edges = np.concatenate([pos_edges, neg_edges], axis=0)
    all_labels = np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])

    for i in range(len(all_edges)):
        node1_idx = int(all_edges[i, 0])
        node2_idx = int(all_edges[i, 1])
        label = int(all_labels[i])

        try:
            drug_id, target_id = convert_edge_to_ids(
                node1_idx, node2_idx, drug_ids, target_ids, num_drugs
            )

            all_records.append({
                'drugbank_id': drug_id,
                'uniprot_id': target_id,
                'label': label,
                'type': 'positive' if label == 1 else 'negative',
                'node1_idx': node1_idx,
                'node2_idx': node2_idx
            })
        except Exception as e:
            print(f"⚠️  警告: 第 {i} 条边转换失败: {e}")
            continue

    print(f"✅ 成功转换 {len(all_records)} 条边")

    # 4. 创建 DataFrame 并保存
    df = pd.DataFrame(all_records)

    # 保存完整数据
    output_csv = os.path.join(save_dir, 'test_pairs_with_real_ids.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 完整测试对已保存至: {output_csv}")

    # 分别保存正样本和负样本
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]

    pos_csv = os.path.join(save_dir, 'test_positive_pairs_with_ids.csv')
    neg_csv = os.path.join(save_dir, 'test_negative_pairs_with_ids.csv')

    pos_df.to_csv(pos_csv, index=False, encoding='utf-8-sig')
    neg_df.to_csv(neg_csv, index=False, encoding='utf-8-sig')

    print(f"💾 正样本已保存至: {pos_csv} (共 {len(pos_df)} 条)")
    print(f"💾 负样本已保存至: {neg_csv} (共 {len(neg_df)} 条)")

    # 5. 显示统计信息
    print("\n" + "=" * 80)
    print("📊 统计信息")
    print("=" * 80)
    print(f"总测试对数量: {len(df)}")
    print(f"正样本数量: {len(pos_df)}")
    print(f"负样本数量: {len(neg_df)}")
    print(f"正负样本比例: {len(pos_df) / len(neg_df):.2f}")

    # 显示唯一药物和靶标数量
    unique_drugs = df['drugbank_id'].nunique()
    unique_targets = df['uniprot_id'].nunique()
    print(f"唯一药物数量: {unique_drugs}")
    print(f"唯一靶标数量: {unique_targets}")

    # 6. 显示示例数据
    print("\n" + "=" * 80)
    print("📋 前10个测试对示例:")
    print("=" * 80)
    print(df.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("✅ 提取完成！")
    print("=" * 80)

    return df


if __name__ == '__main__':
    # 可以修改这里的数据集名称
    dataset_name = 'bindingdB'  # 可选: 'ttd', 'bindingdb', 'drugbank1w'

    df = extract_test_pairs_with_real_ids(dataset_name)
