import csv

# 文件路径
interaction_file = r'e:\研究生\研三\论文\toJournal\my_model\dataset\bindingDB\both\drug_interactions_smiles_final_filtered.csv'
features_file = r'e:\研究生\研三\论文\toJournal\my_model\dataset\bindingDB\both\adjusted_drug_features.csv'
output_file = r'e:\研究生\研三\论文\toJournal\my_model\dataset\bindingDB\both\related_drug_pairs.txt'

# 读取 adjusted_drug_features.csv 获取所有药物列表
drug_set = set()
with open(features_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        if row:
            drug_set.add(row[0])  # db_id

print(f"adjusted_drug_features.csv 中共有 {len(drug_set)} 个药物")

# 读取药物相互作用文件并生成相关药物对
related_pairs = set()

with open(interaction_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        if len(row) >= 2:
            drug_id = row[0]
            related_drugs = row[1].split('|')
            
            # 只考虑当前药物也在 features 文件中的情况
            if drug_id in drug_set:
                for related_drug in related_drugs:
                    # 确保相关药物也在 features 文件中，且避免重复对（按字典序排序）
                    if related_drug in drug_set:
                        pair = tuple(sorted([drug_id, related_drug]))
                        if drug_id != related_drug:  # 排除自环
                            related_pairs.add(pair)

print(f"共找到 {len(related_pairs)} 对相关药物")

# 写入结果文件
with open(output_file, 'w', encoding='utf-8') as f:
    for pair in sorted(related_pairs):
        f.write(f"{pair[0]}\t{pair[1]}\n")

print(f"结果已保存到 {output_file}")