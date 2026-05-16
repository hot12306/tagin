import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GlobalAttentionPooling(nn.Module):
    """
    全局自注意力池化模块：为每个节点分配一个权重，做加权和，获得全局特征。
    """
    def __init__(self, in_channels, hidden_channels):
        super(GlobalAttentionPooling, self).__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, batch):
        batch = batch.to(x.device)
        att_score = self.att_mlp(x)  # [N, 1]
        att_score = torch.exp(att_score - att_score.max())
        att_sum = torch.zeros(batch.max()+1, 1, device=x.device).scatter_add_(0, batch.unsqueeze(-1), att_score)
        att_score = att_score / (att_sum[batch] + 1e-8)
        global_feat = torch.zeros(batch.max()+1, x.size(1), device=x.device).scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(1)), att_score * x)
        return global_feat  # [batch_size, F]

class GlobalLocalFusionNetGate(nn.Module):
    def __init__(self, args, num_features):
        super(GlobalLocalFusionNetGate, self).__init__()
        self.use_global = getattr(args, 'use_global', True)
        self.use_local = getattr(args, 'use_local', True)
        hidden_channels = getattr(args, 'hidden_dim', 64)
        out_channels = getattr(args, 'out_dim', 1)
        num_layers = getattr(args, 'dm_layers', 2)
        heads = getattr(args, 'dm_heads', 2)
        self.gat_layers = nn.ModuleList()
        self.gat_output_dim = hidden_channels * heads
        if self.use_local:
            self.gat_layers.append(GATConv(num_features, hidden_channels, heads=heads, concat=True))
            for _ in range(num_layers-1):
                self.gat_layers.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True))
        if self.use_global:
            self.global_pool = GlobalAttentionPooling(num_features, hidden_channels)
        self.local_dim = hidden_channels * heads if self.use_local else 0
        self.global_dim = self.local_dim if self.use_global else 0
        self.fusion_dim = self.local_dim + self.global_dim
        self.global_align = None
        if self.use_local and self.use_global:
            self.gate = nn.Sequential(
                nn.Linear(self.fusion_dim, 1),
                nn.Sigmoid()
            )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.gat_output_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.gat_output_dim)
        )
        self._fusion_mlp_built = False
        self._hidden_channels = hidden_channels
        self._out_channels = out_channels
        self._gat_output_dim = self.gat_output_dim

    def forward(self, x, edge_index, batch):
        # 获取模型设备
        device = next(self.parameters()).device
        
        # 确保所有输入都在同一设备上
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        feats = []
        local_feat = None
        global_feat_expanded = None
        if self.use_local:
            h = x
            for gat in self.gat_layers:
                h = F.elu(gat(h, edge_index))
            local_feat = h
            feats.append(local_feat)
        if self.use_global:
            global_feat = self.global_pool(x, batch)
            global_feat_expanded = global_feat[batch]
            feats.append(global_feat_expanded)
        # 门控融合
        if self.use_local and self.use_global:
            # 新增：自动对齐特征维度
            if local_feat.shape[1] != global_feat_expanded.shape[1]:
                if self.global_align is None or self.global_align.in_features != global_feat_expanded.shape[1] or self.global_align.out_features != local_feat.shape[1]:
                    self.global_align = nn.Linear(global_feat_expanded.shape[1], local_feat.shape[1]).to(device)
                global_feat_expanded = self.global_align(global_feat_expanded)
            fusion = torch.cat([local_feat, global_feat_expanded], dim=1)
            # 动态适配gate输入维度
            if not hasattr(self, '_gate_built') or self.gate[0].in_features != fusion.shape[1]:
                self.gate = nn.Sequential(
                    nn.Linear(fusion.shape[1], 1).to(device),
                    nn.Sigmoid()
                )
                self._gate_built = True
            gate_weight = self.gate(fusion)  # [N, 1]
            fusion = gate_weight * local_feat + (1 - gate_weight) * global_feat_expanded
        else:
            fusion = feats[0]
        # 动态适配fusion_mlp输入维度
        if not self._fusion_mlp_built or self.fusion_mlp[0].in_features != fusion.shape[1]:
            in_dim = fusion.shape[1]
            self.fusion_mlp = nn.Sequential(
                nn.Linear(in_dim, self._hidden_channels).to(device),
                nn.ReLU(),
                nn.Linear(self._hidden_channels, self.gat_output_dim).to(device)
            )
            self._fusion_mlp_built = True
        out = self.fusion_mlp(fusion)
        return out

class GlobalLocalFusionTaskModelGate(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        self.use_global = getattr(args, 'use_global', True)
        self.use_local = getattr(args, 'use_local', True)
        self.hid_dim = getattr(args, 'hidden_dim', 256)
        self.sf = getattr(args, 'sf', False)
        self.sf_dim = getattr(args, 's_dim', 128)
        heads = getattr(args, 'dm_heads', 2)
        hidden_channels = getattr(args, 'hidden_dim', 64)
        self.gat_output_dim = hidden_channels * heads
        self.fusion = GlobalLocalFusionNetGate(args, num_features)
        self.nf_lin1 = nn.Linear(self.gat_output_dim, self.hid_dim)
        self.nf_lin2 = nn.Linear(2 * self.hid_dim, self.hid_dim // 2)
        self.bn2 = nn.BatchNorm1d(self.hid_dim // 2)
        if self.sf:
            self.sg_lin = nn.Linear(getattr(args, 'hops', 3) * (getattr(args, 'hops', 3) + 2), self.sf_dim)
            self.bn1 = nn.BatchNorm1d(self.sf_dim)
            self.output = nn.Linear(self.sf_dim + self.hid_dim // 2, 1)
        else:
            # 为了兼容sf=True时的预训练权重，使用相同的输出层维度
            # 当sf=False时，在forward中只使用前hid_dim//2个特征
            self.output = nn.Linear(self.sf_dim + self.hid_dim // 2, 1)

    def feature_forward(self, x):
        x = self.nf_lin1(x)
        x = torch.cat([x[:, 0, :], x[:, 1, :]], 1)
        x = self.nf_lin2(x)
        x = F.dropout(F.relu(self.bn2(x)), p=0.5)
        return x

    def forward(self, data, sample_indices, links, indices):
        # 获取模型设备
        device = next(self.parameters()).device
        
        # 确保所有输入都在同一设备上
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch.to(device)
        else:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        
        # 确保数据在正确的设备上
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        
        # 确保索引在正确的设备上
        sample_indices = sample_indices.to(device)
        links = links.to(device)
        indices = indices.to(device)
        
        x = self.fusion(data.x, data.edge_index, batch)
        nf = x[links[indices]]
        nf = self.feature_forward(nf).to(torch.float)
        
        if self.sf:
            # 确保subgraph_features在正确的设备上
            if hasattr(data, 'subgraph_features'):
                data.subgraph_features = data.subgraph_features.to(device)
                sf = data.subgraph_features[sample_indices[indices]].to(torch.float32)
                sf = F.dropout(F.relu(self.bn1(self.sg_lin(sf))), p=0.5)
                x = torch.cat([sf, nf], dim=1)
            else:
                # 如果没有subgraph_features，使用零填充
                zeros = torch.zeros(nf.size(0), self.sf_dim, device=device)
                x = torch.cat([zeros, nf], dim=1)
        else:
            # 当sf=False时，创建一个与输出层维度匹配的张量
            # 前sf_dim维度用0填充，后hid_dim//2维度使用nf
            zeros = torch.zeros(nf.size(0), self.sf_dim, device=device)
            x = torch.cat([zeros, nf], dim=1)
        out = torch.sigmoid(self.output(x))
        return out 