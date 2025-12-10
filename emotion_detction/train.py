from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from collections import Counter
from utils import load_landmarks_json
from dataset import FaceMeshDataset, FaceMeshGraphDataset

from model import MLP, GCN

import os
import json
import joblib
import torch
import torch.nn.functional as F
import numpy as np

EMO_MAP = {
    "01": "joy",
    "02": "neutral",
    "03": "surprise",
    "04": "disgust",
    "05": "sadness",
    "06": "anger",
    "07": "fear",
}


def train_mlp_from_json(
    train_jsons: list[str],
    valid_jsons: list[str] | None = None,
    out_dir: str = "./outputs",
    use_pca: bool = True,
    pca_dim: int = 256,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 80,
    patience: int = 10,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 데이터 로드
    X_train_all, y_train_all, label2id, id2label = load_landmarks_json(
        train_jsons, code2emotion=EMO_MAP
    )
    print(f"Train raw: {X_train_all.shape}, classes={len(label2id)}")

    if valid_jsons:
        X_valid_all, y_valid_all, _, _ = load_landmarks_json(
            valid_jsons, code2emotion=EMO_MAP
        )
        print(f"Valid raw: {X_valid_all.shape}")
        # train/valid을 이미 나눠서 주신 경우 그대로 사용
        has_external_valid = True
    else:
        # 내부에서 train/val 분할
        has_external_valid = False
        X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(
            X_train_all,
            y_train_all,
            test_size=0.1,
            stratify=y_train_all,
            random_state=seed,
        )
        print(f"Split → Train: {X_train_all.shape}, Valid: {X_valid_all.shape}")

    # ---- Dataset & Dataloader (PCA는 train에만 fit)
    ds_tr = FaceMeshDataset(
        X_train_all,
        y_train_all,
        use_pca=use_pca,
        pca_dim=pca_dim,
        fit_pca_on=X_train_all,
    )
    ds_va = FaceMeshDataset(
        X_valid_all,
        y_valid_all,
        use_pca=use_pca,
        pca_dim=pca_dim,
        fit_pca_on=X_train_all,
    )

    in_dim = ds_tr.flat.shape[1]
    num_classes = len(label2id)

    tr_loader = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim, num_classes).to(device)

    # 클래스 분포/가중치
    cnt = Counter(ds_tr.y.tolist())
    print("Train class counts:", dict(cnt))
    weights = np.zeros(num_classes, dtype=np.float32)
    total = sum(cnt.values())
    for c in range(num_classes):
        weights[c] = total / (cnt.get(c, 1) + 1e-6)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_val, best_state, patience_left = 0.0, None, patience
    for epoch in range(1, max_epochs + 1):
        # ---- train
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # ---- valid
        model.eval()
        correct = total = 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                all_pred.extend(pred.cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())
        val_acc = correct / max(total, 1)
        print(f"[{epoch}/{max_epochs}] val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val, best_state, patience_left = (
                val_acc,
                {k: v.cpu() for k, v in model.state_dict().items()},
                patience,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ---- 평가 리포트
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_true.extend(yb.numpy().tolist())

    print("\nValidation classification report:")
    target_names = [id2label[i] for i in range(num_classes)]
    print(
        classification_report(all_true, all_pred, target_names=target_names, digits=4)
    )
    print("Confusion matrix:\n", confusion_matrix(all_true, all_pred))

    # ---- 저장
    model_path = os.path.join(out_dir, "mlp_facemesh.pt")
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if use_pca and ds_tr.pca is not None:
        joblib.dump(ds_tr.pca, os.path.join(out_dir, "pca.joblib"))

    print(
        f"\n✅ Saved:\n- {model_path}\n- {os.path.join(out_dir, 'labels.json')}\n- {os.path.join(out_dir, 'pca.joblib') if use_pca else '(no PCA)'}"
    )

    # ---- Iris Option(True) ----
    # Loaded OK: 38274 / Total: 38274
    # Classes detected: 7 → ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    # Train raw: (38274, 468, 3), classes=7
    # Skip reasons: {}
    # Loaded OK: 4790 / Total: 4790
    # Classes detected: 7 → ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    # Valid raw: (4790, 468, 3)
    # Train class counts: {0: 5410, 5: 5351, 6: 5625, 3: 5448, 2: 5506, 1: 5430, 4: 5504}
    # [1/80] val_acc=0.6142
    # [2/80] val_acc=0.6802
    # [3/80] val_acc=0.6931
    # [4/80] val_acc=0.7010
    # [5/80] val_acc=0.7077
    # [6/80] val_acc=0.7098
    # [7/80] val_acc=0.7154
    # [8/80] val_acc=0.7167
    # [9/80] val_acc=0.7244
    # [10/80] val_acc=0.7188
    # [11/80] val_acc=0.7236
    # [12/80] val_acc=0.7267
    # [13/80] val_acc=0.7317
    # [14/80] val_acc=0.7223
    # [15/80] val_acc=0.7240
    # [16/80] val_acc=0.7261
    # [17/80] val_acc=0.7286
    # [18/80] val_acc=0.7228
    # [19/80] val_acc=0.7271
    # [20/80] val_acc=0.7284
    # [21/80] val_acc=0.7292
    # [22/80] val_acc=0.7315
    # [23/80] val_acc=0.7261
    # Early stopping.

    # Validation classification report:
    #               precision    recall  f1-score   support

    #        anger     0.5786    0.7458    0.6517       661
    #      disgust     0.6345    0.5880    0.6104       682
    #         fear     0.7319    0.6525    0.6899       682
    #          joy     0.7702    0.8136    0.7913       692
    #      neutral     0.8100    0.7237    0.7644       695
    #      sadness     0.7271    0.7455    0.7362       672
    #     surprise     0.9117    0.8484    0.8789       706

    #     accuracy                         0.7317      4790
    #    macro avg     0.7377    0.7311    0.7318      4790
    # weighted avg     0.7396    0.7317    0.7331      4790

    # Confusion matrix:
    #  [[493  65   8  21  12  61   1]
    #  [105 401  79  54  17  23   3]
    #  [ 68  75 445  26   8  28  32]
    #  [ 36  51   8 563  24   5   5]
    #  [ 60  11  14  31 503  61  15]
    #  [ 82  24  14   8  41 501   2]
    #  [  8   5  40  28  16  10 599]]

    # ---- Iris Option(False) ----

    return model, (ds_tr.pca if use_pca else None), label2id, id2label


def train_gcn_from_json(
    train_jsons: list[str],
    valid_jsons: list[str] | None = None,
    out_dir: str = "./outputs_gcn",
    fixed_edge_index=None,  # (2,E) numpy or torch.Tensor, 없으면 k-NN
    knn_k: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 80,
    patience: int = 10,
    seed: int = 42,
    code2emotion: dict | None = None,  # EMO_MAP 전달
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 데이터 로드 ----
    X_train_all, y_train_all, label2id, id2label = load_landmarks_json(
        train_jsons, code2emotion=code2emotion
    )
    print(f"Train raw: {X_train_all.shape}, classes={len(label2id)}")

    if valid_jsons:
        X_valid_all, y_valid_all, _, _ = load_landmarks_json(
            valid_jsons, code2emotion=code2emotion
        )
        print(f"Valid raw: {X_valid_all.shape}")
        has_external_valid = True
    else:
        has_external_valid = False
        X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(
            X_train_all,
            y_train_all,
            test_size=0.1,
            stratify=y_train_all,
            random_state=seed,
        )
        print(f"Split → Train: {X_train_all.shape}, Valid: {X_valid_all.shape}")

    # ---- Datasets & Loaders ----
    ds_tr = FaceMeshGraphDataset(
        X_train_all, y_train_all, fixed_edge_index=fixed_edge_index, knn_k=knn_k
    )
    ds_va = FaceMeshGraphDataset(
        X_valid_all, y_valid_all, fixed_edge_index=fixed_edge_index, knn_k=knn_k
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    tr_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin,
        persistent_workers=True,
    )
    va_loader = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
        persistent_workers=True,
    )

    num_classes = len(label2id)
    model = GCN(num_classes=num_classes, dropout=0.3).to(device)

    # 클래스 가중치
    cnt = Counter(ds_tr.y.tolist())
    print("Train class counts:", dict(cnt))
    weights = np.zeros(num_classes, dtype=np.float32)
    total = sum(cnt.values())
    for c in range(num_classes):
        weights[c] = total / (cnt.get(c, 1) + 1e-6)
    weights = torch.tensor(
        weights / max(weights.mean(), 1e-6), dtype=torch.float32, device=device
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_val, best_state, patience_left = 0.0, None, patience

    for epoch in range(1, max_epochs + 1):
        # ---- train
        model.train()
        for batch in tr_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # ---- valid
        model.eval()
        correct = total = 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                all_pred.extend(pred.detach().cpu().numpy().tolist())
                all_true.extend(batch.y.detach().cpu().numpy().tolist())
        val_acc = correct / max(total, 1)
        print(f"[{epoch}/{max_epochs}] val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val, best_state, patience_left = (
                val_acc,
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                patience,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- 평가 리포트 ----
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in va_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_true.extend(batch.y.cpu().numpy().tolist())

    print("\nValidation classification report:")
    target_names = [id2label[i] for i in range(num_classes)]
    print(
        classification_report(all_true, all_pred, target_names=target_names, digits=4)
    )
    print("Confusion matrix:\n", confusion_matrix(all_true, all_pred))

    # ---- 저장 ----
    model_path = os.path.join(out_dir, "gcn_facemesh.pt")
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n✅ Saved:\n- {model_path}\n- {os.path.join(out_dir, 'labels.json')}")
    return model, label2id, id2label


# =========================
# 4) 실행 예시
# =========================
if __name__ == "__main__":
    # 예: 앞서 만든 파일 경로로 변경하세요.
    ROOT = os.path.join(
        os.sep,
        "media",
        "edint",
        "64d115f7-57cc-417b-acf0-7738ac091615",
        "Ivern",
        "DataSets",
        "FaceLandmark",
    )
    train_jsons = [os.path.join(ROOT, "train_landmarks.json")]
    valid_jsons = [
        os.path.join(ROOT, "valid_landmarks.json")
    ]  # 없다면 빈 리스트나 None

    train_mlp_from_json(
        train_jsons=train_jsons,
        valid_jsons=valid_jsons,
        out_dir=os.path.join(ROOT, "mlp_output"),
        use_pca=True,  # 권장: 1404D → 256D
        pca_dim=256,
        batch_size=512,
        lr=3e-4,
        weight_decay=1e-4,
        max_epochs=80,
        patience=10,
        seed=42,
    )
    # model, label2id, id2label = train_gcn_from_json(
    #     train_jsons=train_jsons,
    #     valid_jsons=valid_jsons,
    #     out_dir=os.path.join(ROOT, "gcn_output"),
    #     fixed_edge_index=None,  # 있으면 np.load("facemesh_edge_index.npy")
    #     knn_k=8,
    #     batch_size=256,
    #     lr=1e-3,
    #     weight_decay=1e-4,
    #     max_epochs=80,
    #     patience=10,
    #     seed=42,
    #     code2emotion=EMO_MAP,
    # )
