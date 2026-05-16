import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_excel("../dataset/DPI/DrugBank/drugbank-DPI_with_features.xlsx")

# ========================
# 1. 生成药物文件 drugs.csv
# ========================
drugs_df = df[["Drug_ID", "drug_features"]].drop_duplicates(subset=["Drug_ID"])
drugs_df.to_csv("dataset/drugs.csv", index=False)

# ==========================
# 2. 生成靶标文件 targets.csv
# ==========================
targets_df = df[["UniProt_ID", "protein_features"]].drop_duplicates(subset=["UniProt_ID"])
targets_df.to_csv("dataset/targets.csv", index=False)

# ========================================
# 3. 生成严格对齐的关系矩阵
# ========================================
# 从文件重新读取保证顺序一致性（重要！）
drug_order = pd.read_csv("dataset/drugbank/drugs.csv")["Drug_ID"].tolist()  # 行顺序
target_order = pd.read_csv("dataset/drugbank/targets.csv")["UniProt_ID"].tolist()  # 列顺序

# 创建空矩阵
adj_matrix = pd.DataFrame(
    np.zeros((len(drug_order), len(target_order)), dtype=int),
    index=drug_order,
    columns=target_order
)

# 填充关联关系
relation_pairs = set(zip(df["Drug_ID"], df["UniProt_ID"]))
for drug, target in relation_pairs:
    adj_matrix.at[drug, target] = 1

# 保存矩阵
adj_matrix.to_csv("dataset/drug_target_adjacency_matrix.csv", header=True)

print("矩阵已生成，保证与CSV文件行列严格对齐！")
