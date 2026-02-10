import cv2
import pickle
import face_recognition
from ultralytics import YOLO
from datetime import datetime
from database import mark_attendance

model = YOLO("yolov8n.pt")

data = pickle.load(open("encodings/faces.pkl", "rb"))
known_encodings = data["encodings"]
known_names = data["names"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]

            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)

            for enc in encodings:
                matches = face_recognition.compare_faces(known_encodings, enc)
                name = "Unknown"

                if True in matches:
                    index = matches.index(True)
                    name = known_names[index]

                    now = datetime.now()
                    mark_attendance(
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")
                    )

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
