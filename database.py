import sqlite3
import os
from datetime import datetime

from contextlib import contextmanager

DB_NAME = os.environ.get("ATTENDANCE_DB", "attendance.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()

        # Students Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            class_name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Attendance Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            UNIQUE(student_id, date),
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        )
        """)

        conn.commit()


def add_student(name, roll_no, class_name):
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO students (name, roll_no, class_name)
            VALUES (?, ?, ?)
            """, (name.strip(), roll_no.strip(), class_name.strip()))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Already exists
            cursor.execute("SELECT id FROM students WHERE name=? OR roll_no=?", (name.strip(), roll_no.strip()))
            row = cursor.fetchone()
            return row["id"] if row else None


def delete_student(student_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
        cursor.execute("DELETE FROM students WHERE id=?", (student_id,))
        conn.commit()
        return True


def get_all_students():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students ORDER BY roll_no ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_student_by_name(name):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", (name.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_student_by_id(student_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE id=?", (student_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def mark_attendance(student_id, date=None, time=None, status="Present"):
    now = datetime.now()
    if not date:
        date = now.strftime("%Y-%m-%d")
    if not time:
        time = now.strftime("%H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO attendance (student_id, date, time, status)
            VALUES (?, ?, ?, ?)
            """, (student_id, date, time, status))
            conn.commit()
            return True, "Marked successfully"
        except sqlite3.IntegrityError:
            return False, "Already marked for today"


def get_attendance_records(date_filter=None, class_filter=None):
    with get_connection() as conn:
        cursor = conn.cursor()
        query = """
        SELECT attendance.id as attendance_id,
               students.id as student_id,
               students.name,
               students.roll_no,
               students.class_name,
               attendance.date,
               attendance.time,
               attendance.status
        FROM attendance
        JOIN students ON attendance.student_id = students.id
        WHERE 1=1
        """
        params = []
        if date_filter:
            query += " AND attendance.date = ?"
            params.append(date_filter)
        if class_filter:
            query += " AND students.class_name = ?"
            params.append(class_filter)

        query += " ORDER BY attendance.date DESC, attendance.time DESC"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM students")
        total_students = cursor.fetchone()["count"]

        cursor.execute("""
        SELECT COUNT(*) as count FROM attendance
        WHERE date = ? AND status = 'Present'
        """, (today,))
        present_today = cursor.fetchone()["count"]

        absent_today = max(0, total_students - present_today)
        percentage = round((present_today / total_students * 100), 1) if total_students > 0 else 0.0

        return {
            "total_students": total_students,
            "present_today": present_today,
            "absent_today": absent_today,
            "attendance_percentage": percentage,
            "today_date": today
        }


if __name__ == "__main__":
    init_db()
    print("✅ Database initialized successfully.")