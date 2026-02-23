import os
import cv2
import pickle
import numpy as np
import torch
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, session
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

app = Flask(__name__)
app.secret_key = "supersecretkey"

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
# LOAD EMBEDDINGS
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
# DATABASE
# ==============================
def init_db():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        roll_no TEXT UNIQUE,
        class_name TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date TEXT,
        time TEXT,
        status TEXT DEFAULT 'Present',
        UNIQUE(student_id, date),
        FOREIGN KEY(student_id) REFERENCES students(id)
    )
    """)

    conn.commit()
    conn.close()


def add_student(name, roll_no, class_name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR IGNORE INTO students (name, roll_no, class_name)
    VALUES (?, ?, ?)
    """, (name, roll_no, class_name))
    conn.commit()
    conn.close()


def get_student_id(name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM students WHERE name=?", (name,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def mark_attendance(student_id):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    try:
        cursor.execute("""
        INSERT INTO attendance (student_id, date, time, status)
        VALUES (?, ?, ?, ?)
        """, (student_id, date, time, "Present"))

        conn.commit()
        print("Attendance marked")

    except sqlite3.IntegrityError:
        print("Already marked today")

    conn.close()


# ==============================
# FACE DETECTION
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
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            emb = facenet(face).numpy()

        distances = np.linalg.norm(known_vectors - emb, axis=1)
        min_dist = np.min(distances)
        best_match = known_names[np.argmin(distances)]

        if min_dist < THRESHOLD:
            student_id = get_student_id(best_match)
            if student_id:
                mark_attendance(student_id)


# ==============================
# ROUTES
# ==============================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form["role"]
        session["role"] = role
        if role == "admin":
            return redirect("/dashboard")
        else:
            return redirect("/upload_page")
    return render_template("login.html")


@app.route("/upload_page")
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file found"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    detect_and_mark(filepath)

    return redirect("/dashboard")


@app.route("/dashboard")
def dashboard():
    if session.get("role") != "admin":
        return redirect("/")

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT students.name,
           students.roll_no,
           students.class_name,
           attendance.date,
           attendance.time,
           attendance.status
    FROM attendance
    JOIN students ON attendance.student_id = students.id
    ORDER BY attendance.id DESC
    """)

    records = cursor.fetchall()
    conn.close()

    return render_template("dashboard.html", records=records)


# ==============================
# START
# ==============================
if __name__ == "__main__":
    init_db()

    add_student("Suraj Jaiswal", "101", "CSE")
    add_student("Sahil Ithape", "102", "CSE")
    add_student("Sarthak Gupta", "103", "CSE")
    add_student("Vedant Rai", "104", "CSE")
    add_student("Yuvarudra Pawar", "105", "CSE")
    add_student("Sujal Kumar", "106", "CSE")

    app.run(debug=True)