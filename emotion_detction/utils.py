from collections import Counter

import os
import json
import numpy as np


def _to_xyz(v):
    if v is None:
        return None
    if isinstance(v, dict):
        if all(k in v for k in ("x", "y", "z")):
            return [v["x"], v["y"], v["z"]]
        if all(k in v for k in ("X", "Y", "Z")):
            return [v["X"], v["Y"], v["Z"]]
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return [v[0], v[1], v[2]]
    return None


def _parse_landmarks(lm) -> np.ndarray | None:
    """
    lm을 (468,3) np.float32로 변환.
    지원:
      - dict: {"0":[x,y,z], ..., "467":[x,y,z]} (키 str/ int 모두 OK)
      - list: [[x,y,z], ...] 길이 >= 468
      - dict of dicts: {"0":{"x":..,"y":..,"z":..}, ...}도 일부 지원
    """
    coords = None
    # case 1) list
    if isinstance(lm, list) and len(lm) >= 468:
        tmp = []
        for i in range(468):
            xyz = _to_xyz(lm[i])
            if xyz is None:
                return None
            tmp.append(xyz)
        coords = np.array(tmp, dtype=np.float32)

    # case 2) dict with numeric or string keys
    elif isinstance(lm, dict) and (len(lm) >= 468):
        tmp = []
        for i in range(468):
            key = i if i in lm else str(i) if str(i) in lm else None
            if key is None:
                return None
            xyz = _to_xyz(lm[key])
            if xyz is None:
                return None
            tmp.append(xyz)
        coords = np.array(tmp, dtype=np.float32)

    # sanity check
    if coords is None or coords.shape != (468, 3):
        return None
    # NaN/Inf 체크
    if not np.isfinite(coords).all():
        return None
    return coords


def emotion_from_filename(
    image_name: str, mapping: dict[str, str] | None = None
) -> str | None:
    """
    'F0029_01_1_H.png' -> 토큰['_'] 기준 세 번째(인덱스 2) = '1'
    mapping이 주어지면 숫자 코드를 감정명으로 변환
    """
    base = os.path.basename(image_name)
    stem = os.path.splitext(base)[0]  # F0029_01_1_H
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    code = parts[1]  # '1'
    if mapping:
        return mapping.get(code, code)  # 매핑 없으면 원본 코드 유지
    return code


def load_landmarks_json(
    json_paths: list[str], code2emotion: dict[str, str] | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[int, str]]:
    """
    json_paths에 있는 파일들을 모두 읽어서
    X: (N, 468, 3), y: (N,), label2id, id2label을 반환
    JSON 포맷:
    {
      "folder_name": str,
      "image_name": str,
      "landmarks": {"0":[x,y,z], ..., "467":[x,y,z]}
    } 의 리스트
    """
    records = []
    for p in json_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        records.extend(data)

    # 먼저 감정 라벨 뽑기 (파일명에서)
    raw_labels = []
    for r in records:
        img = r.get("image_name")
        lbl = emotion_from_filename(img, mapping=code2emotion) if img else None
        raw_labels.append(lbl)

    # 유효 라벨만으로 라벨 집합 구성
    unique_labels = sorted(list({lb for lb in raw_labels if lb is not None}))
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    id2label = {i: lb for lb, i in label2id.items()}

    X_list, y_list = [], []
    reason_counter = Counter()

    for r, lb in zip(records, raw_labels):
        if lb is None:
            reason_counter["label_parse_failed"] += 1
            continue
        lm = r.get("landmarks")
        if lm is None:
            reason_counter["no_landmarks"] += 1
            continue
        coords = _parse_landmarks(lm)
        if coords is None:
            reason_counter["malformed"] += 1
            continue
        X_list.append(coords)
        y_list.append(label2id[lb])

    print("Skip reasons:", dict(reason_counter))
    X = np.stack(X_list) if X_list else np.zeros((0, 468, 3), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"Loaded OK: {len(X_list)} / Total: {len(records)}")
    print(f"Classes detected: {len(label2id)} → {unique_labels}")
    return X, y, label2id, id2label


# =========================
# 2) 정규화 & Dataset/Model
# =========================
def normalize_landmarks(x: np.ndarray) -> np.ndarray:
    """얼굴 중심 정렬 + 스케일 정규화(양 눈 외곽 33, 263) 혹은 std 백업"""
    x = x.astype(np.float32)
    center = x.mean(axis=0, keepdims=True)
    x = x - center
    try:
        L, R = x[33], x[263]
        scale = np.linalg.norm(L - R)
        if scale < 1e-6:
            raise ValueError
    except Exception:
        scale = x.std() + 1e-6
    return x / scale
