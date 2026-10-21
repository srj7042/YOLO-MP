# YOLO-MP: AI Biometric Attendance System

A modern, fast, and robust face recognition attendance management platform built with Python, Flask, OpenCV, FaceNet, and SQLite.

---

## ⚡ Quick Start (Single Command)

To clean up legacy files, set up the virtual environment, install dependencies, initialize the database, and launch the portal, run:

```bash
./run.sh
```

### Other Run Modes:
- **Standalone Desktop Camera**:
  ```bash
  ./run.sh camera
  ```
- **Train Faces from `dataset/` directory**:
  ```bash
  ./run.sh train
  ```

---

## 🔑 Default Login Credentials

| Role | Username / Selection | Password |
| :--- | :--- | :--- |
| **Admin** | `admin` | `admin123` |
| **Faculty / Teacher** | `teacher` | `teacher123` |

---

## ✨ Features

1. **In-Browser Real-Time Video Attendance (`/live`)**:
   - Stream webcam directly to the browser with automated face detection, matching, and attendance marking.
2. **Classroom Group Photo Scanner (`/upload`)**:
   - Upload high-resolution classroom images to detect and mark multiple students simultaneously.
   - Displays annotated visual bounding boxes and confidence scores.
3. **Student Directory & Biometric Registration (`/students`)**:
   - Add students with their portrait photo to automatically generate FaceNet embeddings without running terminal scripts.
4. **Dynamic Excel & CSV Export (`/export/excel`, `/export/csv`)**:
   - Download complete or filtered attendance reports in `.xlsx` or `.csv` format.
5. **Modern Glassmorphic Dashboard (`/dashboard`)**:
   - Real-time KPI statistics: Total Enrolled, Present Today, Absent Today, Turnout Rate.
   - Filter attendance logs by Date and Class.