# heterogeneous_network_visualization.py - 药物-靶标异构图可视化
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Patch
from networkx.algorithms import community

# 设置字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HeterogeneousDTINetwork:
    """药物-靶标异构图可视化"""

    def __init__(self, save_dir='save/bindingdB'):
        self.save_dir = save_dir
        self.G = nx.Graph()
        self.drug_info = {}  # 药物信息
        self.target_info = {}  # 靶标信息
        self.interaction_info = {}  # 相互作用信息

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")

        # 1. 加载测试集对（带ID）
        csv_file = os.path.join(self.save_dir, 'test_pairs_with_real_ids.csv')
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"文件不存在: {csv_file}")

        self.df_pairs = pd.read_csv(csv_file)
        print(f"✅ 加载测试集对: {len(self.df_pairs)} 条")

        # 2. 加载药物特征信息
        drug_features_file = 'dataset/bindingDB/origin/adjusted_drug_features.csv'
        if os.path.exists(drug_features_file):
            self.df_drugs = pd.read_csv(drug_features_file)
            print(f"✅ 加载药物信息: {len(self.df_drugs)} 个药物")

            # 提取药物信息
            for _, row in self.df_drugs.iterrows():
                drug_id = row['db_id']
                self.drug_info[drug_id] = {
                    'name': row.get('name', 'Unknown'),
                    'type': 'Drug'
                }

        # 3. 加载靶标特征信息
        target_features_file = 'dataset/bindingDB/origin/adjusted_protein_features.csv'
        if os.path.exists(target_features_file):
            self.df_targets = pd.read_csv(target_features_file)
            print(f"✅ 加载靶标信息: {len(self.df_targets)} 个靶标")

            # 提取靶标信息
            for _, row in self.df_targets.iterrows():
                target_id = row['uniprot_id']
                self.target_info[target_id] = {
                    'name': row.get('name', 'Unknown'),
                    'type': 'Target'
                }

        print("✅ 数据加载完成\n")

    def build_heterogeneous_network(self, use_positive_only=True, max_nodes=80):
        """
        构建异构图

        节点类型:
        - Drug: 药物（红色）
        - Target: 靶标（绿色）
        - Disease: 疾病（蓝色）- 如果有数据
        - Pathway: 通路（紫色）- 如果有数据

        边类型:
        - interacts_with: 药物-靶标相互作用
        - treats: 药物治疗疾病
        - targets: 靶标参与疾病
        """
        print("正在构建异构图...")

        self.G = nx.Graph()

        # 选择数据
        if use_positive_only:
            data = self.df_pairs[self.df_pairs['label'] == 1]
        else:
            data = self.df_pairs

        # 添加药物-靶标相互作用边
        for _, row in data.iterrows():
            drug_id = row['drugbank_id']
            target_id = row['uniprot_id']

            # 添加药物节点
            if not self.G.has_node(drug_id):
                self.G.add_node(drug_id,
                                node_type='drug',
                                label=drug_id[:10],
                                color='#E74C3C')  # 红色

            # 添加靶标节点
            if not self.G.has_node(target_id):
                self.G.add_node(target_id,
                                node_type='target',
                                label=target_id[:10],
                                color='#27AE60')  # 绿色

            # 添加相互作用边
            self.G.add_edge(drug_id, target_id,
                            edge_type='interacts_with',
                            weight=1.0)

        # 如果节点太多，选择重要节点
        if self.G.number_of_nodes() > max_nodes:
            print(f"⚠️ 节点数过多 ({self.G.number_of_nodes()})，选择度数最高的 {max_nodes} 个节点")
            degrees = dict(self.G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_names = [n[0] for n in top_nodes]
            self.G = self.G.subgraph(top_node_names)

        print(f"✅ 异构图构建完成:")
        print(f"   总节点数: {self.G.number_of_nodes()}")
        print(f"   总边数: {self.G.number_of_edges()}")

        # 统计节点类型
        node_types = {}
        for n, attr in self.G.nodes(data=True):
            ntype = attr.get('node_type', 'unknown')
            node_types[ntype] = node_types.get(ntype, 0) + 1

        for ntype, count in node_types.items():
            print(f"   {ntype}节点: {count}")

    def plot_heterogeneous_network(self, save_path=None, figsize=(20, 16)):
        """
        绘制异构图 - 二分图布局

        特点:
        - 药物节点聚在左侧
        - 靶标节点聚在右侧
        - 布局均匀分布
        """
        if self.G.number_of_nodes() == 0:
            self.build_heterogeneous_network()

        print("\n🎨 正在绘制异构图（二分图布局）...")

        fig, ax = plt.subplots(figsize=figsize)

        # 按节点类型分组
        drug_nodes = [n for n, attr in self.G.nodes(data=True)
                      if attr.get('node_type') == 'drug']
        target_nodes = [n for n, attr in self.G.nodes(data=True)
                        if attr.get('node_type') == 'target']

        # 自定义二分图布局：药物在左，靶标在右
        pos = {}

        # 药物节点均匀分布在左侧
        if drug_nodes:
            y_positions = np.linspace(1, 0, len(drug_nodes))
            for i, node in enumerate(drug_nodes):
                # 添加轻微的随机扰动避免完全重叠
                jitter = np.random.uniform(-0.05, 0.05)
                pos[node] = (0.25 + jitter, y_positions[i])

        # 靶标节点均匀分布在右侧
        if target_nodes:
            y_positions = np.linspace(1, 0, len(target_nodes))
            for i, node in enumerate(target_nodes):
                jitter = np.random.uniform(-0.05, 0.05)
                pos[node] = (0.75 + jitter, y_positions[i])

        # 使用力导向布局进行微调
        pos = nx.spring_layout(self.G, pos=pos, fixed=list(pos.keys()),
                               k=0.5, iterations=50, seed=42)

        # 绘制药物节点（红色，较大，圆形）
        nx.draw_networkx_nodes(self.G, pos,
                               nodelist=drug_nodes,
                               node_color='#E74C3C',
                               node_size=600,
                               alpha=0.85,
                               edgecolors='black',
                               linewidths=2.0,
                               ax=ax)

        # 绘制靶标节点（绿色，中等大小，方形）
        target_pos = {n: pos[n] for n in target_nodes}
        if target_pos:
            x_coords = [target_pos[n][0] for n in target_nodes]
            y_coords = [target_pos[n][1] for n in target_nodes]
            ax.scatter(x_coords, y_coords,
                       s=500,
                       c='#27AE60',
                       marker='s',
                       alpha=0.85,
                       edgecolors='black',
                       linewidths=2.0,
                       zorder=2)

        # 绘制边（灰色细线）
        nx.draw_networkx_edges(self.G, pos,
                               width=1.5,
                               alpha=0.5,
                               edge_color='gray',
                               style='solid',
                               ax=ax)

        # 添加标签
        degrees = dict(self.G.degree())
        avg_degree = np.mean(list(degrees.values()))

        # 标记所有度数 >= 平均度数一半的节点
        labeled_nodes = [n for n, d in degrees.items() if d >= avg_degree * 0.3]
        labels = {n: self.G.nodes[n].get('label', n[:8]) for n in labeled_nodes}

        nx.draw_networkx_labels(self.G, pos,
                                labels=labels,
                                font_size=8,
                                font_weight='bold',
                                font_color='black',
                                ax=ax)

        # 创建自定义图例
        legend_elements = [
            Patch(facecolor='#E74C3C', edgecolor='black', label='Drugs'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#27AE60',
                       markersize=12, label='Targets', linestyle='None'),
            plt.Line2D([0], [0], color='gray', lw=2.0, label='Drug-Target Interaction')
        ]

        ax.legend(handles=legend_elements,
                  loc='upper center',
                  fontsize=13,
                  framealpha=0.9,
                  fancybox=True,
                  shadow=True,
                  bbox_to_anchor=(0.5, 0.98))

        # 添加标题
        ax.set_title('Heterogeneous Drug-Target Interaction Network (Bipartite Layout)',
                     fontsize=20, fontweight='bold', pad=30)
        ax.axis('off')

        # 添加左右标签
        ax.text(0.25, 1.05, 'DRUGS', fontsize=16, fontweight='bold',
                ha='center', va='bottom', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#E74C3C', alpha=0.3))
        ax.text(0.75, 1.05, 'TARGETS', fontsize=16, fontweight='bold',
                ha='center', va='bottom', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#27AE60', alpha=0.3))

        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ 异构图已保存至: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'visualizations',
                                     'heterogeneous_network_bipartite.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ 异构图已保存至: {save_path}")

        plt.close()

    def plot_network_with_communities(self, save_path=None, figsize=(20, 16)):
        """
        绘制带社区检测的异构图 - 二分图布局

        使用 Louvain 算法检测网络中的社区结构
        """
        if self.G.number_of_nodes() == 0:
            self.build_heterogeneous_network()

        print("\n🎨 正在绘制带社区检测的异构图（二分图布局）...")

        # 社区检测
        communities = community.greedy_modularity_communities(self.G)

        # 为每个社区分配颜色
        community_colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        node_community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_community_map[node] = i

        fig, ax = plt.subplots(figsize=figsize)

        # 按节点类型分组
        drug_nodes = [n for n, attr in self.G.nodes(data=True)
                      if attr.get('node_type') == 'drug']
        target_nodes = [n for n, attr in self.G.nodes(data=True)
                        if attr.get('node_type') == 'target']

        # 自定义二分图布局
        pos = {}

        # 药物节点均匀分布在左侧
        if drug_nodes:
            y_positions = np.linspace(1, 0, len(drug_nodes))
            for i, node in enumerate(drug_nodes):
                jitter = np.random.uniform(-0.05, 0.05)
                pos[node] = (0.25 + jitter, y_positions[i])

        # 靶标节点均匀分布在右侧
        if target_nodes:
            y_positions = np.linspace(1, 0, len(target_nodes))
            for i, node in enumerate(target_nodes):
                jitter = np.random.uniform(-0.05, 0.05)
                pos[node] = (0.75 + jitter, y_positions[i])

        # 力导向微调
        pos = nx.spring_layout(self.G, pos=pos, fixed=list(pos.keys()),
                               k=0.5, iterations=50, seed=42)

        # 按社区绘制药物节点（圆形）
        for i, comm in enumerate(communities):
            comm_drug_nodes = [n for n in comm
                               if self.G.nodes[n].get('node_type') == 'drug']
            if comm_drug_nodes:
                color = community_colors[i]
                nx.draw_networkx_nodes(self.G, pos,
                                       nodelist=comm_drug_nodes,
                                       node_color=[color],
                                       node_size=600,
                                       alpha=0.85,
                                       edgecolors='black',
                                       linewidths=2.0,
                                       ax=ax)

        # 按社区绘制靶标节点（方形）
        for i, comm in enumerate(communities):
            comm_target_nodes = [n for n in comm
                                 if self.G.nodes[n].get('node_type') == 'target']
            if comm_target_nodes:
                color = community_colors[i]
                target_pos = {n: pos[n] for n in comm_target_nodes}
                x_coords = [target_pos[n][0] for n in comm_target_nodes]
                y_coords = [target_pos[n][1] for n in comm_target_nodes]
                ax.scatter(x_coords, y_coords,
                           s=500,
                           c=[color],
                           marker='s',
                           alpha=0.85,
                           edgecolors='black',
                           linewidths=2.0,
                           zorder=2)

        # 绘制边
        nx.draw_networkx_edges(self.G, pos,
                               width=1.5,
                               alpha=0.4,
                               edge_color='gray',
                               ax=ax)

        # 添加标签
        degrees = dict(self.G.degree())
        avg_degree = np.mean(list(degrees.values()))
        labeled_nodes = [n for n, d in degrees.items() if d >= avg_degree * 0.3]
        labels = {n: self.G.nodes[n].get('label', n[:8]) for n in labeled_nodes}

        nx.draw_networkx_labels(self.G, pos,
                                labels=labels,
                                font_size=7,
                                font_weight='bold',
                                ax=ax)

        # 创建图例
        legend_elements = [
            Patch(facecolor=community_colors[0], label='Community'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=12, label='Drug (Circle)', linestyle='None'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                       markersize=10, label='Target (Square)', linestyle='None')
        ]

        ax.legend(handles=legend_elements,
                  loc='upper center',
                  fontsize=11,
                  framealpha=0.9,
                  bbox_to_anchor=(0.5, 0.98))

        ax.set_title('Heterogeneous Network with Community Detection (Bipartite Layout)',
                     fontsize=20, fontweight='bold', pad=30)
        ax.axis('off')

        # 添加左右标签
        ax.text(0.25, 1.05, 'DRUGS', fontsize=16, fontweight='bold',
                ha='center', va='bottom', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.75, 1.05, 'TARGETS', fontsize=16, fontweight='bold',
                ha='center', va='bottom', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ 社区检测异构图已保存至: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'visualizations',
                                     'heterogeneous_network_communities_bipartite.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ 社区检测异构图已保存至: {save_path}")

        plt.close()

    def generate_all_networks(self):
        """生成所有网络图"""
        print("=" * 80)
        print("开始生成异构图可视化")
        print("=" * 80)

        # 加载数据
        self.load_data()

        # 构建网络
        self.build_heterogeneous_network(use_positive_only=True, max_nodes=80)

        output_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(output_dir, exist_ok=True)

        # 1. 基础异构图
        print("\n📊 1. 生成基础异构图...")
        self.plot_heterogeneous_network(
            save_path=os.path.join(output_dir, '06_heterogeneous_network.png')
        )

        # 2. 带社区检测的异构图
        print("\n📊 2. 生成带社区检测的异构图...")
        self.plot_network_with_communities(
            save_path=os.path.join(output_dir, '07_heterogeneous_network_communities.png')
        )

        print("\n" + "=" * 80)
        print("✅ 所有异构图生成完成！")
        print("=" * 80)


if __name__ == '__main__':
    viz = HeterogeneousDTINetwork(save_dir='save/bindingdB')
    viz.generate_all_networks()

    print("\n💡 提示:")
    print("   Heterogeneous Graph Contains:")
    print("   - Drug Nodes (Red Circles): DrugBank ID")
    print("   - Target Nodes (Green Squares): UniProt ID")
    print("   - Interaction Edges: Drug-Target binding relationships")
    print("   - Community Structure: Reveals potential functional modules")
