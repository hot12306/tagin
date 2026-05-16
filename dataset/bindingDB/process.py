import pandas as pd

# 读取调整后的特征文件
adjusted_drugs = pd.read_csv('both/adjusted_drug_features.csv')['db_id'].tolist()
adjusted_proteins = pd.read_csv('both/adjusted_protein_features.csv')['uniprot_id'].tolist()

# 读取原始目标矩阵
drug_target_matrix = pd.read_csv('drug_target_matrix.csv')

# 删除adjusted_drug_features.csv中不存在的行(药物)
filtered_matrix = drug_target_matrix[drug_target_matrix.iloc[:, 0].isin(adjusted_drugs)]

# 删除adjusted_protein_features.csv中不存在的列(蛋白质)
# 第一列是drug_id，需要保留
columns_to_keep = [filtered_matrix.columns[0]] + adjusted_proteins
filtered_matrix = filtered_matrix[columns_to_keep]

# 保存结果
filtered_matrix.to_csv('filtered_drug_target_matrix.csv', index=False)
