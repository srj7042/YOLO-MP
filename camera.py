import cv2
import pickle
import numpy as np
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from database import mark_attendance

# =========================
# LOAD MODELS
# =========================
print("[INFO] Loading YOLO model...")
yolo = YOLO("yolov8n.pt")

print("[INFO] Loading FaceNet model...")
facenet = InceptionResnetV1(pretrained="vggface2").eval()

# =========================
# LOAD FACE EMBEDDINGS
# =========================
print("[INFO] Loading face encodings...")
with open("encodings/faces.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

# Flatten embeddings
known_names = []
known_vectors = []

for name, embeds in known_embeddings.items():
    for emb in embeds:
        known_names.append(name)
        known_vectors.append(emb)

known_vectors = np.array(known_vectors)

# =========================
# CAMERA START
# =========================
cap = cv2.VideoCapture(0)
print("🎥 Camera started. Press Q to quit.")

THRESHOLD = 0.9  # lower = stricter matching

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, conf=0.5, verbose=False)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Preprocess face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
        face = face / 255.0

        with torch.no_grad():
            emb = facenet(face).numpy()

        # Compare with known embeddings
        distances = np.linalg.norm(known_vectors - emb, axis=1)
        min_dist = np.min(distances)
        best_match = known_names[np.argmin(distances)]

        if min_dist < THRESHOLD:
            name = best_match
            color = (0, 255, 0)
            mark_attendance(name)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Draw box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

    cv2.imshow("YOLO Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed")
