import pandas as pd

# 读取数据
df = pd.read_csv('../../../../dataset/DrugBank/DTI/DPI_pos_filtered.csv')

# 创建药物-靶标矩阵
drug_target_matrix = pd.crosstab(df['db_id'], df['target'])

# 将矩阵保存为CSV文件
drug_target_matrix.to_csv('drug_target_matrix.csv')

print("药物靶标矩阵已保存为 drug_target_matrix.csv")
