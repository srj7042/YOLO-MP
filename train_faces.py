import os
import cv2
import pickle
import numpy as np
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# =========================
# PATHS
# =========================
DATASET_DIR = "dataset"        # dataset/Suraj/img1.jpg
ENCODINGS_DIR = "encodings"
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "faces.pkl")

os.makedirs(ENCODINGS_DIR, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
print("[INFO] Loading YOLO face detector...")
yolo = YOLO("yolov8n.pt")  # lightweight YOLO

print("[INFO] Loading FaceNet model...")
facenet = InceptionResnetV1(pretrained="vggface2").eval()

# =========================
# TRAINING
# =========================
embeddings = {}

print("[INFO] Starting face encoding...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)

    # Ignore files like .DS_Store
    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing {person}")
    embeddings[person] = []

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        # YOLO face detection
        results = yolo(image, conf=0.5, verbose=False)

        if len(results[0].boxes) == 0:
            print(f"⚠ No face detected in {img_name}")
            continue

        # Take first detected face
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Preprocess for FaceNet
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
        face = face / 255.0

        # Generate embedding
        with torch.no_grad():
            embedding = facenet(face).numpy()[0]

        embeddings[person].append(embedding)

print("[INFO] Saving embeddings...")

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Training complete. Face embeddings saved to encodings/faces.pkl")
