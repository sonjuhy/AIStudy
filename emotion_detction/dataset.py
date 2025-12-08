from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset as GeoDataset, DataLoader
from collections import Counter
from utils import normalize_landmarks

import os
import json
import torch
import numpy as np


# ====== NLP Dataset ======
class FaceMeshDataset(Dataset):
    def __init__(self, X, y, use_pca=False, pca_dim=256, fit_pca_on=None):
        self.X = np.stack([normalize_landmarks(s) for s in X])  # (N,468,3)
        self.y = y.astype(np.int64)
        self.flat = self.X.reshape(len(self.X), -1)  # (N,1404)

        self.pca = None
        if use_pca:
            self.pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
            fit_data = (
                self.flat
                if fit_pca_on is None
                else fit_pca_on.reshape(len(fit_pca_on), -1)
            )
            self.pca.fit(fit_data)
            self.flat = self.pca.transform(self.flat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.flat[i]).float(), torch.tensor(self.y[i])


# ====== GCN Dataset ======
def build_knn_edges(coords, k=8):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    dist, idx = nn.kneighbors(coords)
    # 자기 자신 제외
    idx = idx[:, 1:]
    src = np.repeat(np.arange(coords.shape[0]), k)
    dst = idx.reshape(-1)
    edge = np.vstack([src, dst])
    # 무방향화
    edge_rev = np.vstack([edge[1], edge[0]])
    edge_all = np.concatenate([edge, edge_rev], axis=1)
    # 중복 제거
    edge_all = np.unique(edge_all, axis=1)
    return edge_all


class FaceMeshGraphDataset(GeoDataset):
    def __init__(self, X, y, fixed_edge_index=None, knn_k=8):
        super().__init__()
        self.X = np.stack([normalize_landmarks(s) for s in X])  # (N,468,3)
        self.y = y.astype(np.int64)
        self.fixed_edge_index = None
        if fixed_edge_index is not None:
            e = np.array(fixed_edge_index, dtype=np.int64)
            assert e.shape[0] == 2
            self.fixed_edge_index = torch.from_numpy(e)
        self.knn_k = knn_k

    def len(self):
        return len(self.X)

    def get(self, idx):
        coords = self.X[idx]  # (468,3)
        x = torch.from_numpy(coords).float()  # node features: [x,y,z]
        y = torch.tensor(self.y[idx]).long()

        if self.fixed_edge_index is not None:
            edge_index = self.fixed_edge_index
        else:
            edge_np = build_knn_edges(coords, k=self.knn_k)  # (2,E)
            edge_index = torch.from_numpy(edge_np).long()

        return Data(x=x, edge_index=edge_index, y=y)


def dataset_valid():
    ROOT = os.path.join(
        os.sep,
        "media",
        "edint",
        "64d115f7-57cc-417b-acf0-7738ac091615",
        "Ivern",
        "DataSets",
        "FaceLandmark",
    )
    path = os.path.join(ROOT, "train_landmarks.json")
    with open(path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    tok2 = Counter()  # 세 번째 토큰(인덱스 2)
    tok3 = Counter()  # 네 번째 토큰(인덱스 3)

    for r in recs:
        stem = os.path.splitext(os.path.basename(r["image_name"]))[0]
        parts = stem.split("_")
        if len(parts) > 2:
            tok2[parts[2]] += 1
        if len(parts) > 3:
            tok3[parts[3]] += 1

    print("3rd token (idx 2) distribution:", tok2.most_common(10))
    print("4th token (idx 3) distribution:", tok3.most_common(10))


if __name__ == "__main__":
    dataset_valid()
