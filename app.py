import os
import cv2
import pickle
import numpy as np
import torch
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD MODELS
# ==============================
print("[INFO] Loading YOLO...")
yolo = YOLO("yolov8n.pt")

print("[INFO] Loading FaceNet...")
facenet = InceptionResnetV1(pretrained="vggface2").eval()

# ==============================
# LOAD FACE EMBEDDINGS
# ==============================
with open("encodings/faces.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

known_names = []
known_vectors = []

for name, embeds in known_embeddings.items():
    for emb in embeds:
        known_names.append(name)
        known_vectors.append(emb)

known_vectors = np.array(known_vectors)
THRESHOLD = 0.9


# ==============================
# DATABASE SETUP
# ==============================
def init_database():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        roll_no TEXT,
        class TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date TEXT,
        time TEXT,
        status TEXT,
        FOREIGN KEY(student_id) REFERENCES students(id)
    )
    """)

    conn.commit()
    conn.close()


def add_student(name, roll_no, class_name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO students (name, roll_no, class)
    VALUES (?, ?, ?)
    """, (name, roll_no, class_name))

    conn.commit()
    conn.close()


def mark_attendance(name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")

    cursor.execute("SELECT id FROM students WHERE name=?", (name,))
    student = cursor.fetchone()

    if student is None:
        conn.close()
        return

    student_id = student[0]

    cursor.execute("""
    SELECT * FROM attendance
    WHERE student_id=? AND date=?
    """, (student_id, today))

    if cursor.fetchone() is None:
        cursor.execute("""
        INSERT INTO attendance (student_id, date, time, status)
        VALUES (?, ?, ?, ?)
        """, (student_id, today, now, "Present"))
        conn.commit()

    conn.close()


# ==============================
# FACE DETECTION + RECOGNITION
# ==============================
def detect_and_mark(image_path):
    frame = cv2.imread(image_path)
    results = yolo(frame, conf=0.5)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
        face = face / 255.0

        with torch.no_grad():
            emb = facenet(face).numpy()

        distances = np.linalg.norm(known_vectors - emb, axis=1)
        min_dist = np.min(distances)
        best_match = known_names[np.argmin(distances)]

        if min_dist < THRESHOLD:
            mark_attendance(best_match)


# ==============================
# ROUTES
# ==============================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            detect_and_mark(path)
        return redirect("/dashboard")

    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT students.name, students.roll_no, students.class,
           attendance.date, attendance.time, attendance.status
    FROM attendance
    JOIN students ON attendance.student_id = students.id
    ORDER BY attendance.id DESC
    """)

    records = cursor.fetchall()
    conn.close()

    return render_template("dashboard.html", records=records)


# ==============================
# START APP
# ==============================
if __name__ == "__main__":
    init_database()

    # Add your students here (edit as needed)
    # Updated Students
    add_student("Yuvarudra Pawar", "62", "CSE")
    add_student("Vedant Rai", "58", "CSE")
    add_student("Suraj Jaiswal", "48", "CSE")
    add_student("Sujal Kumar", "47", "CSE")


    app.run(debug=True)
