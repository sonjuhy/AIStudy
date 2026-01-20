from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import os
import pandas as pd
import numpy as np


class FER2013Dataset(Dataset):
    """
    FER2013 CSV 데이터셋
    - columns: ['emotion', 'pixels', 'Usage']
    - pixels: 공백으로 구분된 48x48 (그레이스케일)
    - Usage: 'Training', 'PublicTest', 'PrivateTest'
    """

    def __init__(self, csv_path, usage="Training", transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["Usage"] == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ").reshape(48, 48)

        # 그레이스케일(1채널) → 3채널 복제 (MobileNet 호환)
        img = np.repeat(img[..., np.newaxis], 3, axis=2)

        if self.transform:
            img = self.transform(img)

        label = int(row["emotion"])
        return img, label


def download_dataset():
    # 데이터 변환 정의
    transform_train = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # CIFAR-10 데이터셋 (10 클래스)
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    return {"train": trainloader, "test": testloader}


def download_fer2013_dataset(
    csv_path="./data/fer2013.csv", batch_size=64, num_workers=4
):
    """
    FER2013 데이터셋 로드 함수
    - FER2013 CSV 파일은 https://www.kaggle.com/datasets/deadskull7/fer2013 에서 다운로드 가능
    """
    if not os.path.exists(csv_path):
        print(csv_path)
        raise FileNotFoundError(
            f"'{csv_path}' not found. Please download fer2013.csv first."
        )

    # 이미지 전처리 정의
    transform_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # FER2013 CSV 분할 로드
    trainset = FER2013Dataset(csv_path, usage="Training", transform=transform_train)
    valset = FER2013Dataset(csv_path, usage="PublicTest", transform=transform_test)
    testset = FER2013Dataset(csv_path, usage="PrivateTest", transform=transform_test)

    # 데이터로더 정의
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(
        f"Loaded FER2013 dataset: train={len(trainset)} val={len(valset)} test={len(testset)}"
    )

    return {"train": trainloader, "val": valloader, "test": testloader}


class EarlyStopping:
    def __init__(self, patience=7, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, current):
        if self.best is None:
            self.best = current
            return False

        improved = (
            (current - self.best) > self.min_delta
            if self.mode == "max"
            else (self.best - current) > self.min_delta
        )
        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.should_stop


def compute_weights_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    y = df[df["Usage"] == "Training"]["emotion"].astype(int).values
    classes = np.arange(7)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        gts.extend(y.cpu().numpy())
    loss = loss_sum / max(1, n)
    acc = accuracy_score(gts, preds)
    f1m = f1_score(gts, preds, average="macro")
    return loss, acc, f1m, np.array(gts), np.array(preds)
