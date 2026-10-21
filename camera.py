import cv2
import database
import face_engine

print("[INFO] Starting standalone camera attendance module...")
database.init_db()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit(1)

print("🎥 Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    faces = face_engine.detect_faces(frame)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        name, confidence, dist = face_engine.match_face(face_crop)

        if name != "Unknown":
            color = (0, 255, 0)
            student = database.get_student_by_name(name)
            if student:
                success, msg = database.mark_attendance(student["id"])
                label = f"{name} ({confidence}%)"
            else:
                label = f"{name}"
        else:
            color = (0, 0, 255)
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YOLO-MP Face Attendance (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed.")
