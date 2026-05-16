import pandas as pd


def align_to_drug_target_matrix(
        drug_csv_path,
        protein_csv_path,
        matrix_csv_path,
        output_drug_csv_path,
        output_protein_csv_path
):
    """调整药物和蛋白质特征顺序，与 drug_target_matrix.csv 完全一致"""
    # 读取药物和蛋白质特征文件
    drug_df = pd.read_csv(drug_csv_path)
    protein_df = pd.read_csv(protein_csv_path)

    # 读取药物-蛋白质关联矩阵
    matrix_df = pd.read_csv(matrix_csv_path, index_col=0)

    # 获取药物顺序（矩阵的行名）
    drug_order = matrix_df.index.tolist()

    # 获取蛋白质顺序（矩阵的列名）
    protein_order = matrix_df.columns.tolist()

    # 调整药物特征顺序
    aligned_drug_df = drug_df.set_index('db_id').loc[drug_order].reset_index()

    # 调整蛋白质特征顺序
    aligned_protein_df = protein_df.set_index('uniprot_id').loc[protein_order].reset_index()

    # 保存对齐后的文件
    aligned_drug_df.to_csv(output_drug_csv_path, index=False)
    aligned_protein_df.to_csv(output_protein_csv_path, index=False)

    print(f"药物和蛋白质顺序已对齐到 drug_target_matrix.csv！输出文件：")
    print(f"- 药物: {output_drug_csv_path}")
    print(f"- 蛋白质: {output_protein_csv_path}")


# 示例调用
align_to_drug_target_matrix(
    drug_csv_path='drug_features_standardized.csv',
    protein_csv_path='protein_features_standardized.csv',
    matrix_csv_path='drug_target_matrix.csv',
    output_drug_csv_path='drug_features_aligned.csv',
    output_protein_csv_path='protein_features_aligned.csv'
)
