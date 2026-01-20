import os
import cv2
import json
import mediapipe as mp


def extract_data():
    ROOT_PATH = os.path.join(
        os.sep,
        "media",
        "edint",
        "64d115f7-57cc-417b-acf0-7738ac091615",
        "Ivern",
        "DataSets",
        "FaceLandmark",
    )
    train_dataset_path = os.path.join(ROOT_PATH, "train", "images")
    valid_dataset_path = os.path.join(ROOT_PATH, "valid", "images")

    # 결과 저장 경로
    save_train_json = os.path.join(ROOT_PATH, "train_non_iris_landmarks.json")
    save_valid_json = os.path.join(ROOT_PATH, "valid_non_iris_landmarks.json")

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,  # 비디오 스트림 처리에 최적화
        max_num_faces=2,
        refine_landmarks=False,  # 눈, 입술 주변 랜드마크 정제하여 정확도 향상
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    train_results = []
    valid_results = []

    for folder_name in os.listdir(train_dataset_path):
        folder_path = os.path.join(train_dataset_path, folder_name)
        for image_name in os.listdir(folder_path):
            image = cv2.imread(os.path.join(folder_path, image_name))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_result = mp_face_mesh.process(image_rgb)
            if not mp_result.multi_face_landmarks:
                continue

            face_landmarks = mp_result.multi_face_landmarks[0]
            landmark_dict = {
                str(i): [lm.x, lm.y, lm.z]
                for i, lm in enumerate(face_landmarks.landmark)
            }
            train_results.append(
                {
                    "folder_name": folder_name,
                    "image_name": image_name,
                    "landmarks": landmark_dict,
                }
            )
    for folder_name in os.listdir(valid_dataset_path):
        folder_path = os.path.join(valid_dataset_path, folder_name)
        for image_name in os.listdir(folder_path):
            image = cv2.imread(os.path.join(folder_path, image_name))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_result = mp_face_mesh.process(image_rgb)
            if not mp_result.multi_face_landmarks:
                continue

            face_landmarks = mp_result.multi_face_landmarks[0]
            landmark_dict = {
                str(i): [lm.x, lm.y, lm.z]
                for i, lm in enumerate(face_landmarks.landmark)
            }
            valid_results.append(
                {
                    "folder_name": folder_name,
                    "image_name": image_name,
                    "landmarks": landmark_dict,
                }
            )

    # JSON 저장
    with open(save_train_json, "w", encoding="utf-8") as f:
        json.dump(train_results, f, ensure_ascii=False, indent=2)
    with open(save_valid_json, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Done! Train: {len(train_results)} samples, Valid: {len(valid_results)} samples."
    )
    print(f"Saved to:\n  {save_train_json}\n  {save_valid_json}")


if __name__ == "__main__":
    extract_data()
