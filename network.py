import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch
from KANLinear import KANLinear
import matplotlib.pyplot as plt
class Encoder(nn.Module):
    def __init__(self,input_dim,feature_dim):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
            KANLinear(input_dim, 500),
            KANLinear(500, 500),
            KANLinear(500, 2000),
            KANLinear(2000, feature_dim),
        )
    def forward(self,x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self,input_dim,feature_dim):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential(
            KANLinear(feature_dim, 2000),
            KANLinear(2000, 500),
            KANLinear(500, 500),
            KANLinear(500, input_dim),

        )
    def forward(self,z):
        return self.decoder(z)

class Fusion(nn.Module):
    def __init__(self, feature_dim):
        super(Fusion, self).__init__()
        self.G1 = nn.Linear(feature_dim * 8, feature_dim)
        self.G2 = nn.Linear(feature_dim * 8, feature_dim)
        self.G3 = nn.Linear(feature_dim * 8, feature_dim)
        self.G = nn.Linear(feature_dim * 3, feature_dim)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(feature_dim)
        self.bn2 = nn.BatchNorm1d(feature_dim)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.bn_final = nn.BatchNorm1d(feature_dim)

        # 权重初始化
        for layer in [self.G1, self.G2, self.G3, self.G]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def fusion(self, z, z_new):
        # 检查输入是否包含 NaN 值
        if torch.isnan(z).any() or torch.isnan(z_new).any():
            print("Warning: Input contains NaN values!")
            return torch.zeros_like(z)  # 返回与 z 相同形状的零张量

        z1 = self.G1(torch.cat([z, z_new], dim=-1))
        z1 = self.bn1(z1)  # Batch Normalization
        z1 = F.leaky_relu(z1, negative_slope=0.01)  # 使用 Leaky ReLU 激活

        z2 = self.G2(torch.cat([z, z - z_new], dim=-1))
        z2 = self.bn2(z2)
        z2 = F.leaky_relu(z2, negative_slope=0.01)

        z3 = self.G3(torch.cat([z, z * z_new], dim=-1))
        z3 = self.bn3(z3)
        z3 = F.leaky_relu(z3, negative_slope=0.01)

        z_s = self.G(torch.cat([z1, z2, z3], dim=-1))
        z_s = self.bn_final(z_s)  # 最后的 Batch Normalization
        z_s = F.dropout(z_s, p=0.2, training=self.training)  # 仅在训练期间应用 Dropout

        # 检查输出是否包含 NaN 值
        if torch.isnan(z_s).any():
            print("Warning: z_s contains NaN values!")

        return z_s

    def forward(self, z, z_new):
        x = self.fusion(z, z_new)
        return x
class Network(nn.Module):
    def __init__(self,view,input_dim,feature_dim,high_feature_dim,device):
        super(Network,self).__init__()
        self.view = view
        self.fusion = Fusion(high_feature_dim)
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_dim[v],feature_dim).to(device))
            self.decoders.append(Decoder(input_dim[v],feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim,feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim,high_feature_dim),
        )
    def forward(self,xs):
        xrs = []
        zs = []
        rs = []
        x1 = xs[0]
        z1 = self.encoders[0](x1)
        xr1 = self.decoders[0](z1)
        zs.append(z1)
        xrs.append(xr1)
        x2 = xs[1]
        z2 = self.encoders[1](x2)
        xr2 = self.decoders[1](z2)
        zs.append(z2)
        xrs.append(xr2)
        z_fusion = self.fusion(z1,z2)
        r1 = self.feature_contrastive_module(z1)
        r1 = normalize(r1,dim=1)
        rs.append(r1)
        r2 = self.feature_contrastive_module(z2)
        r2 = normalize(r2,dim=1)
        rs.append(r2)
        H = z_fusion
        return xrs,zs,rs,H


"""
mlp-fusion
class Network(nn.Module):
    def __init__(self,view,input_dim,feature_dim,high_feature_dim,device):
        super(Network,self).__init__()
        self.view = view
        self.fusion = Fusion(high_feature_dim)
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_dim[v],feature_dim).to(device))
            self.decoders.append(Decoder(input_dim[v],feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim,feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim,high_feature_dim),
        )
    def forward(self,xs):
        xrs = []
        zs = []
        rs = []
        x1 = xs[0]
        z1 = self.encoders[0](x1)
        xr1 = self.decoders[0](z1)
        zs.append(z1)
        xrs.append(xr1)
        x2 = xs[1]
        z2 = self.encoders[1](x2)
        xr2 = self.decoders[1](z2)
        zs.append(z2)
        xrs.append(xr2)
        z_fusion = self.fusion(z1,z2)
        r1 = self.feature_contrastive_module(z1)
        r1 = normalize(r1,dim=1)
        rs.append(r1)
        r2 = self.feature_contrastive_module(z2)
        r2 = normalize(r2,dim=1)
        rs.append(r2)
        H = z_fusion
        return xrs,zs,rs,H
"""