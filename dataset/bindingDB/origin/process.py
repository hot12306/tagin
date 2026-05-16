import pandas as pd

# 读取调整后的药物特征文件
adjusted_df = pd.read_csv('../both/adjusted_protein_features.csv')

# 读取原始药物特征文件
original_df = pd.read_csv('../../drugbank1w/origin/protein_features.csv', usecols=['uniprot_id', 'origin'])

# 合并origin列到调整后的特征文件
merged_df = adjusted_df.merge(original_df, on='uniprot_id', how='left')

# 保存结果
merged_df.to_csv('adjusted_protein_features.csv', index=False)
