import networkx as nx
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 读取药物对
drug_pairs = []
with open(r'e:\研究生\研三\论文\toJournal\my_model\dataset\bindingDB\both\related_drug_pairs.txt', 'r',
          encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) >= 2:
                drug_pairs.append((parts[0], parts[1]))

# 读取蛋白质对
protein_pairs = []
with open(r'e:\研究生\研三\论文\toJournal\my_model\dataset\bindingDB\both\related_protein_pairs.txt', 'r',
          encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) >= 2:
                protein_pairs.append((parts[0], parts[1]))

# 读取药物 - 靶标矩阵
drug_target_pairs = []
drugs = set()
proteins = set()

with open(r'e:\研究生\研三\论文\toJournal\my_model\dataset\bindingDB\both\filtered_drug_target_matrix.csv', 'r',
          encoding='utf-8') as f:
    lines = f.readlines()
    header = lines[0].strip().split(',')
    targets = header[1:]

    # 取前 40 个药物
    for line in lines[1:41]:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            drug_id = parts[0]
            drugs.add(drug_id)
            for i, val in enumerate(parts[1:], 1):
                if val == '1':
                    target_id = header[i]
                    proteins.add(target_id)
                    drug_target_pairs.append((drug_id, target_id))

print(f"药物数量：{len(drugs)}")
print(f"靶标数量：{len(proteins)}")
print(f"药物 - 靶标对数量：{len(drug_target_pairs)}")

# 创建异构图
G = nx.Graph()

# 添加所有可能的节点（先不区分类型）
all_drugs_in_pairs = set()
all_proteins_in_pairs = set()

# 添加药物 - 药物边（取前 40 个），并记录有关系的药物
selected_drugs_set = set(drugs)
for d1, d2 in drug_pairs[:40]:
    if d1 in selected_drugs_set and d2 in selected_drugs_set:
        G.add_edge(d1, d2, relation='drug-drug')
        all_drugs_in_pairs.add(d1)
        all_drugs_in_pairs.add(d2)

# 添加蛋白质 - 蛋白质边（取前 50 个），并记录有关系的蛋白质
# 不过滤，直接添加所有蛋白质对
for p1, p2 in protein_pairs[:50]:
    G.add_edge(p1, p2, relation='protein-protein')
    all_proteins_in_pairs.add(p1)
    all_proteins_in_pairs.add(p2)

# 添加药物 - 靶标边（取前 70 个），并记录有关系的节点
for d, p in drug_target_pairs[:70]:
    G.add_edge(d, p, relation='drug-target')
    all_drugs_in_pairs.add(d)
    all_proteins_in_pairs.add(p)

# 为节点添加类型属性
for drug in all_drugs_in_pairs:
    if G.has_node(drug):
        G.nodes[drug]['type'] = 'drug'
for protein in all_proteins_in_pairs:
    if G.has_node(protein):
        G.nodes[protein]['type'] = 'protein'

# 移除孤立节点（没有边的节点）
isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
for node in isolated_nodes:
    G.remove_node(node)

print(f"\n图中节点数：{G.number_of_nodes()}")
print(f"图中边数：{G.number_of_edges()}")
print(f"移除的孤立节点数：{len(isolated_nodes)}")

# 统计边的类型
drug_drug_count = 0
protein_protein_count = 0
drug_target_count = 0
for u, v, data in G.edges(data=True):
    relation = data.get('relation', '')
    if relation == 'drug-drug':
        drug_drug_count += 1
    elif relation == 'protein-protein':
        protein_protein_count += 1
    else:
        drug_target_count += 1

print(f"药物 - 药物边：{drug_drug_count}")
print(f"蛋白质 - 蛋白质边：{protein_protein_count}")
print(f"药物 - 靶标边：{drug_target_count}")

# 绘制图形
plt.figure(figsize=(18, 14))

# 使用 spring 布局，增大排斥力避免节点重叠
pos = nx.spring_layout(G, k=0.8, iterations=100, scale=3.5)

# 设置节点颜色（参考图片风格：红色药物，绿色蛋白质）
node_colors = []
for node in G.nodes:
    if G.nodes[node]['type'] == 'drug':
        node_colors.append('#E74C3C')  # 红色 - 药物
    else:
        node_colors.append('#27AE60')  # 绿色 - 蛋白质

# 绘制节点（大小均匀，参考图片风格）
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000,
                       alpha=0.9, edgecolors='black', linewidths=1)

# 分别绘制不同类型的边（使用曲线）
# 药物-药物边（红色曲线）
drug_drug_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'drug-drug']
nx.draw_networkx_edges(G, pos, edgelist=drug_drug_edges, edge_color='#E74C3C',
                       alpha=0.65, width=1.8, arrows=True, connectionstyle='arc3,rad=0.1')
 
# 蛋白质-蛋白质边（蓝色曲线）
protein_protein_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'protein-protein']
nx.draw_networkx_edges(G, pos, edgelist=protein_protein_edges, edge_color='#3498DB',
                       alpha=0.65, width=1.8, arrows=True, connectionstyle='arc3,rad=0.1')

# 药物-靶标边（紫色曲线）
drug_target_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'drug-target']
nx.draw_networkx_edges(G, pos, edgelist=drug_target_edges, edge_color='#9B59B6',
                       alpha=0.65, width=1.8, arrows=True, connectionstyle='arc3,rad=0.15')

# 绘制标签（所有标签都显示）
nx.draw_networkx_labels(G, pos, font_size=7, font_color='#2C3E50', font_weight='normal')

# 添加图例
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#E74C3C', edgecolor='black', label='Drug Nodes'),
    Patch(facecolor='#27AE60', edgecolor='black', label='Protein Nodes'),
    Line2D([0], [0], color='#E74C3C', linewidth=2.5, label='Drug-Drug Relations'),
    Line2D([0], [0], color='#3498DB', linewidth=2.5, label='Protein-Protein Relations'),
    Line2D([0], [0], color='#9B59B6', linewidth=2.5, label='Drug-Target Relations')
]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)

# 添加一个圆圈包围所有节点
import numpy as np
positions = np.array([pos[node] for node in G.nodes()])
center_x = np.mean(positions[:, 0])
center_y = np.mean(positions[:, 1])
radius = np.max(np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)) + 0.3

circle = plt.Circle((center_x, center_y), radius, color='gray', linestyle='--', 
                    linewidth=2, fill=False, alpha=0.6)
plt.gca().add_patch(circle)

plt.title('Heterogeneous Graph: Drug-Protein Interaction Network', fontsize=14, pad=20)
plt.axis('off')
plt.tight_layout()

# 保存图片
plt.savefig(r'e:\研究生\研三\论文\toJournal\my_model\dataset\bindingDB\both\heterogeneous_graph.png',
            dpi=300, bbox_inches='tight')
print("\n图片已保存：heterogeneous_graph.png")

# 显示图片（添加异常处理以避免 PyCharm 后端问题）
try:
    plt.show()
except AttributeError:
    print("图片已成功保存，跳过显示（PyCharm 后端兼容性问题）")
finally:
    plt.close()
