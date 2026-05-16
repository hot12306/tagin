import pandas as pd


def merge_drug_features():
    """
    将drug_sequences_.csv的seq_features合并到drug_features_aligned.csv的origin列中
    按照drug_id和db_id的对应关系进行匹配
    """
    try:
        # 读取输入文件
        drug_sequences = pd.read_csv('../../../../../dataset/DrugBank/PPI/protein_sequence.csv')
        drug_features = pd.read_csv('../seq/protein_features_aligned.csv')

        # 创建drug_id到seq_features的映射字典
        seq_dict = dict(zip(drug_sequences['uniprot_id'], drug_sequences['seq_features']))

        # 将seq_features合并到origin列
        drug_features['origin'] = drug_features['uniprot_id'].map(seq_dict)

        # 处理可能的空值
        drug_features['origin'] = drug_features['origin'].fillna('')

        # 保存结果
        output_file = 'protein_features.csv'
        drug_features.to_csv(output_file, index=False)

        print(f"合并完成，结果已保存到: {output_file}")
        print(f"处理记录数: {len(drug_features)}")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except KeyError as e:
        print(f"CSV文件中缺少必要的列: {e}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    merge_drug_features()
