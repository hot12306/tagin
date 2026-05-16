import pandas as pd

# 1. 读取 drug_target_matrix 获取药物和靶标的顺序
dt_matrix = pd.read_csv("drug_target_matrix.csv", index_col=0)
drug_order = dt_matrix.index.tolist()  # 药物顺序
target_order = dt_matrix.columns.tolist()  # 靶标顺序

# 2. 调整 drug_normalized.csv 顺序
drug_features = pd.read_csv("drug_normalized.csv")
drug_features = drug_features.set_index("db_id")  # 设为索引方便对齐
drug_features = drug_features.loc[drug_order]  # 按 drug_target_matrix 的药物顺序排列
drug_features.to_csv("drug_normalized.csv")  # 保存调整后的文件

# 3. 调整 protein_normalized.csv 顺序
protein_features = pd.read_csv("protein_normalized.csv")
protein_features = protein_features.set_index("uniprot_id")  # 假设列名为 'protein_id'，按实际情况调整
protein_features = protein_features.loc[target_order]  # 按 drug_target_matrix 的靶标顺序排列
protein_features.to_csv("protein_normalized.csv")  # 保存调整后的文件

print("调整完成！文件已保存为 drug_normalized_aligned.csv 和 protein_normalized_aligned.csv")
