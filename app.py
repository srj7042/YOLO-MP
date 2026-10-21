import os
import io
import cv2
import pandas as pd
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    flash, Response, send_file, jsonify
)
from werkzeug.utils import secure_filename

import database
import face_engine

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "yolo-mp-super-secret-key-2026")

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Default credentials
CREDENTIALS = {
    "admin": "admin123",
    "teacher": "teacher123"
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "role" not in session:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login"))
            if role and session.get("role") != role and session.get("role") != "admin":
                flash("Access denied: insufficient permissions.", "danger")
                return redirect(url_for("dashboard"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ==============================
# AUTH ROUTES
# ==============================
@app.route("/", methods=["GET", "POST"])
def login():
    if "role" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        role = request.form.get("role", "admin").strip().lower()
        password = request.form.get("password", "").strip()

        if role in CREDENTIALS and password == CREDENTIALS[role]:
            session["role"] = role
            session["username"] = role.capitalize()
            flash(f"Welcome back, {session['username']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid role or password. Default: admin / admin123 or teacher / teacher123", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


# ==============================
# DASHBOARD
# ==============================
@app.route("/dashboard")
@login_required()
def dashboard():
    date_filter = request.args.get("date", "").strip() or None
    class_filter = request.args.get("class", "").strip() or None

    records = database.get_attendance_records(date_filter=date_filter, class_filter=class_filter)
    stats = database.get_stats()
    students = database.get_all_students()

    classes = sorted(list({s["class_name"] for s in students if s.get("class_name")}))

    return render_template(
        "dashboard.html",
        records=records,
        stats=stats,
        classes=classes,
        current_date=date_filter or stats["today_date"],
        current_class=class_filter or ""
    )


# ==============================
# LIVE WEBCAM STREAM ROUTE
# ==============================
def gen_camera_frames():
    """Video streaming generator function."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[WARN] Default webcam 0 not accessible.")
        return

    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1
        # Process face recognition every 3 frames to save CPU
        if frame_count % 3 == 0:
            frame, _ = face_engine.process_image_and_mark(frame, mark_in_db=True)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    camera.release()


@app.route("/live")
@login_required()
def live_page():
    return render_template("live.html")


@app.route("/video_feed")
@login_required()
def video_feed():
    return Response(gen_camera_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ==============================
# IMAGE UPLOAD ATTENDANCE
# ==============================
@app.route("/upload", methods=["GET", "POST"])
@login_required()
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in request.", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "warning")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Allowed file types are: PNG, JPG, JPEG, WEBP.", "warning")
            return redirect(request.url)

        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process image
        annotated_img, detections = face_engine.process_image_and_mark(filepath, mark_in_db=True)
        annotated_filename = "annotated_" + filename
        annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
        cv2.imwrite(annotated_path, annotated_img)

        return render_template(
            "upload.html",
            uploaded=True,
            original_image=url_for("static", filename=f"uploads/{filename}"),
            annotated_image=url_for("static", filename=f"uploads/{annotated_filename}"),
            detections=detections
        )

    return render_template("upload.html", uploaded=False)


# ==============================
# STUDENT MANAGEMENT
# ==============================
@app.route("/students")
@login_required(role="admin")
def students_list():
    students = database.get_all_students()
    known_faces, _, _ = face_engine.load_known_faces()
    return render_template("students.html", students=students, known_faces=known_faces)


@app.route("/students/add", methods=["POST"])
@login_required(role="admin")
def add_student():
    name = request.form.get("name", "").strip()
    roll_no = request.form.get("roll_no", "").strip()
    class_name = request.form.get("class_name", "").strip()
    photo = request.files.get("photo")

    if not name or not roll_no or not class_name:
        flash("All student fields are required.", "danger")
        return redirect(url_for("students_list"))

    student_id = database.add_student(name, roll_no, class_name)
    if not student_id:
        flash("Student could not be added (Name or Roll Number already exists).", "warning")
        return redirect(url_for("students_list"))

    # Register face if photo uploaded
    if photo and photo.filename and allowed_file(photo.filename):
        filename = secure_filename(f"student_{student_id}_{photo.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        photo.save(filepath)

        success, msg = face_engine.register_new_face(name, filepath)
        if success:
            flash(f"Student {name} registered with face biometric profile!", "success")
        else:
            flash(f"Student added, but face not detected in photo: {msg}", "warning")
    else:
        flash(f"Student {name} added without photo. You can train their face later.", "info")

    return redirect(url_for("students_list"))


@app.route("/students/delete/<int:student_id>", methods=["POST"])
@login_required(role="admin")
def delete_student(student_id):
    student = database.get_student_by_id(student_id)
    if student:
        database.delete_student(student_id)
        # Remove from faces dict
        faces_data, _, _ = face_engine.load_known_faces()
        if student["name"] in faces_data:
            del faces_data[student["name"]]
            face_engine.save_known_faces(faces_data)
        flash(f"Student {student['name']} deleted successfully.", "success")
    return redirect(url_for("students_list"))


# ==============================
# EXPORT REPORTS
# ==============================
@app.route("/export/excel")
@login_required()
def export_excel():
    records = database.get_attendance_records()
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["name", "roll_no", "class_name", "date", "time", "status"])
    else:
        df = df[["name", "roll_no", "class_name", "date", "time", "status"]]
        df.columns = ["Name", "Roll No", "Class", "Date", "Time", "Status"]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance_Report")
    output.seek(0)

    filename = f"Attendance_Report_{datetime.now().strftime('%Y_%m_%d')}.xlsx"
    return send_file(output, download_name=filename, as_attachment=True)


@app.route("/export/csv")
@login_required()
def export_csv():
    records = database.get_attendance_records()
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["name", "roll_no", "class_name", "date", "time", "status"])
    else:
        df = df[["name", "roll_no", "class_name", "date", "time", "status"]]
        df.columns = ["Name", "Roll No", "Class", "Date", "Time", "Status"]

    csv_data = df.to_csv(index=False)
    output = io.BytesIO(csv_data.encode("utf-8"))
    filename = f"Attendance_Report_{datetime.now().strftime('%Y_%m_%d')}.csv"
    return send_file(output, download_name=filename, as_attachment=True, mimetype="text/csv")


# ==============================
# SEED & STARTUP
# ==============================
def seed_initial_data():
    database.init_db()
    existing = database.get_all_students()
    if not existing:
        print("[INFO] Seeding initial student records...")
        database.add_student("Suraj Jaiswal", "101", "CSE")
        database.add_student("Sahil Ithape", "102", "CSE")
        database.add_student("Sarthak Gupta", "103", "CSE")
        database.add_student("Vedant Rai", "104", "CSE")
        database.add_student("Yuvarudra Pawar", "105", "CSE")
        database.add_student("Sujal Kumar", "106", "CSE")


if __name__ == "__main__":
    seed_initial_data()
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 YOLO-MP Attendance Server running at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)