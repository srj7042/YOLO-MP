import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open("embeddings/faces.pkl", "rb") as f:
    known_embeddings, labels = pickle.load(f)

yolo = YOLO("yolov8n.pt")
facenet = InceptionResnetV1(pretrained="vggface2").eval()

cap = cv2.VideoCapture(0)

THRESHOLD = 0.7

while True:
    ret, frame = cap.read()
    results = yolo(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face = cv2.resize(face, (160, 160))
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
        face = (face - 127.5) / 128.0

        with torch.no_grad():
            emb = facenet(face).numpy()

        sims = cosine_similarity(emb, known_embeddings)
        best_match = np.argmax(sims)
        name = "Unknown"

        if sims[0][best_match] > THRESHOLD:
            name = labels[best_match]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
