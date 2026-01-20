from torch_geometric.data import Data, Dataset as GeoDataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        layer: int = 3,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model = nn.ModuleList()
        self.model.append(GCNConv(in_channels=in_channels, out_channels=out_channels))
        for _ in range(layer - 1):
            self.model.append(
                GCNConv(in_channels=out_channels, out_channels=out_channels)
            )
        self.dropout = dropout
        self.linear_1 = nn.Linear(in_features=out_channels, out_features=256)
        self.linear_2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.model:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)  # [B, hidden]
        x = F.relu(self.linear_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear_2(x)
