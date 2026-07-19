import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 사용자 정의 모듈 (기존 파일명에 맞춰 임포트)
from model import MobileVisionNet, YOLO11_MobileNetV4_DFL
from dataset import get_coco_dataset_val_loader


# ==========================================================================
# [모듈 1] 속도 측정 엔진 (ONNX Runtime 기반)
# ==========================================================================
class LatencyBenchmark:
    @staticmethod
    def export_onnx(
        model: nn.Module, path: str, input_shape: tuple, model_type: str = "custom"
    ) -> str:
        model.eval()

        # 🔥 A. 공식 모델(Ultralytics)인 경우: 전용 export 메서드 사용
        if model_type == "official":
            print(f"📦 Using Ultralytics specialized export for {path}")

            temp_yolo = YOLO("yolo11n.pt")
            # export()는 저장된 경로를 문자열로 반환함
            onnx_path = temp_yolo.export(
                format="onnx", imgsz=input_shape[2], opset=11, simplify=True
            )
            # 반환된 경로가 요청한 path와 다를 수 있으므로 리네임하거나 경로 리턴
            return onnx_path

        # 🔥 B. 독자 모델(MobileVisionNet 등)인 경우: 기존 방식 유지
        else:
            dummy_input = torch.randn(*input_shape)
            torch.onnx.export(
                model,
                dummy_input,
                path,
                input_names=["input"],
                output_names=["output"],
                opset_version=11,
                do_constant_folding=True,
            )
            return path

    @staticmethod
    def run(onnx_path: str, input_shape: tuple, num_tests: int = 100) -> float:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warm-up
        for _ in range(10):
            sess.run(None, {input_name: dummy_input})

        start = time.time()
        for _ in range(num_tests):
            sess.run(None, {input_name: dummy_input})
        avg_ms = ((time.time() - start) / num_tests) * 1000
        return avg_ms


# ==========================================================================
# [모듈 2] 정확도 측정 엔진 (COCO 공식 API 기반)
# ==========================================================================
class AccuracyEvaluator:
    @staticmethod
    def run(model: nn.Module, val_loader, ann_file: str, device: str = "cpu") -> dict:
        model.eval().to(device)
        coco_gt = COCO(ann_file)
        cat_ids = sorted(coco_gt.getCatIds())
        trainid2catid = {i: cid for i, cid in enumerate(cat_ids)}

        results = []
        with torch.no_grad():
            for imgs, batch in tqdm(val_loader, desc="🔍 COCO Eval"):
                imgs = imgs.to(device)
                # 모델의 predict 메서드 (NMS 포함) 사용
                preds = model.predict(imgs, conf_thres=0.001, iou_thres=0.6)

                for i, pred in enumerate(preds):
                    image_id = int(batch["img_ids"][i].item())
                    for box, score, label in zip(
                        pred["boxes"], pred["scores"], pred["labels"]
                    ):
                        x1, y1, x2, y2 = box.tolist()
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": trainid2catid[int(label)],
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(score),
                            }
                        )

        if not results:
            return {"mAP50": 0.0, "mAP50_95": 0.0}

        with open("temp_preds.json", "w") as f:
            json.dump(results, f)
        coco_dt = coco_gt.loadRes("temp_preds.json")
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {"mAP50": coco_eval.stats[1], "mAP50_95": coco_eval.stats[0]}


# ==========================================================================
# [Main] 테스트 제어부
# ==========================================================================
def main(mode: str = "both"):
    """
    mode: 'latency', 'accuracy', 'both'
    """
    DEVICE = "cpu"
    INPUT_SHAPE = (1, 3, 640, 640)
    ANN_PATH = "/home/edint/Ivern_home/WorkSpace/Python/Yolo/coco_dataset/annotations/instances_val2017.json"
    VAL_LOADER = (
        get_coco_dataset_val_loader(img_size=640)
        if mode in ["accuracy", "both"]
        else None
    )

    # 1. 테스트 대상 모델 리스트 정의
    test_configs = [
        {
            "name": "Ultralytics_YOLO11n",
            "model_obj": YOLO("yolo11n.pt").model,  # 공식 가중치 사용
            "weight": None,
            "type": "official",
        },
        {
            "name": "YOLO11_MobileNetV4_DFL",
            "model_obj": YOLO11_MobileNetV4_DFL(num_classes=80),
            "weight": "yolo11_mobilenetv4_dfl_best.pt",
            "type": "custom",
        },
        {
            "name": "MobileVisionNet_R&D",
            "model_obj": MobileVisionNet(num_classes=80),
            # "weight": "mobile_vision_net_best.pt",
            "weight": os.path.join(
                "train_20260401_154129", "mobile_vision_net_ema_best.pt"
            ),
            "type": "exclusive",  # 독자 모델
        },
    ]

    final_results = {}

    for cfg in test_configs:
        name = cfg["name"]
        print(f"\n{'='*20} 🔍 Testing: {name} {'='*20}")

        model = cfg["model_obj"]
        # 가중치 파일이 존재하면 로드
        if cfg["weight"] and os.path.exists(cfg["weight"]):
            model.load_state_dict(torch.load(cfg["weight"], map_location=DEVICE))
            print(f"✅ Loaded weights: {cfg['weight']}")

        final_results[name] = {}

        # 🚀 A. 속도 테스트
        if mode in ["latency", "both"]:
            onnx_path = f"temp_{name}.onnx"
            # 🔥 cfg["type"] ("official" 또는 "custom")을 함께 전달
            final_onnx_path = LatencyBenchmark.export_onnx(
                model, onnx_path, INPUT_SHAPE, model_type=cfg["type"]
            )
            latency = LatencyBenchmark.run(final_onnx_path, INPUT_SHAPE)
            final_results[name]["latency"] = latency

        # 📊 B. 정확도 테스트
        if mode in ["accuracy", "both"]:
            mAP = AccuracyEvaluator.run(model, VAL_LOADER, ANN_PATH, DEVICE)
            final_results[name]["mAP50"] = mAP["mAP50"]
            final_results[name]["mAP50_95"] = mAP["mAP50_95"]
            print(f"📈 mAP@50: {mAP['mAP50']:.4f}")

    # ==========================================================================
    # 🏆 최종 R&D 성능 비교표 출력
    # ==========================================================================
    print("\n" + "==" * 30)
    print(f" {'모델명':<25} | {'Latency (ms)':<12} | {'mAP@50':<8}")
    print("-" * 55)
    for name, res in final_results.items():
        latency = f"{res.get('latency', 0.0):.2f}"
        map50 = f"{res.get('mAP50', 0.0):.4f}"
        print(f" {name:<25} | {latency:<12} | {map50:<8}")
    print("🥇" * 30)


if __name__ == "__main__":
    # 원하는 모드를 선택하여 호출: "latency", "accuracy", "both"
    main(mode="latency")
