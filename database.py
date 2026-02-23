import sqlite3
from datetime import datetime

DB_NAME = "attendance.db"


# =========================
# CONNECT DATABASE
# =========================
def connect_db():
    return sqlite3.connect(DB_NAME)


# =========================
# CREATE TABLES
# =========================
def create_tables():
    conn = connect_db()
    cursor = conn.cursor()

    # STUDENTS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    """)

    # ATTENDANCE TABLE (UPDATED)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        class_name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        UNIQUE(student_name, class_name, date)
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database tables ready")


# =========================
# ADD STUDENT
# =========================
def add_student(name):
    conn = connect_db()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO students (name) VALUES (?)", (name,))
        conn.commit()
        print(f"✅ Student added: {name}")
    except sqlite3.IntegrityError:
        print(f"⚠️ Student already exists: {name}")

    conn.close()


# =========================
# MARK ATTENDANCE
# =========================
def mark_attendance(student_name, class_name):
    conn = connect_db()
    cursor = conn.cursor()

    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    try:
        cursor.execute("""
        INSERT INTO attendance (student_name, class_name, date, time)
        VALUES (?, ?, ?, ?)
        """, (student_name, class_name, date, time))

        conn.commit()
        print(f"🟢 Attendance marked: {student_name} ({class_name})")

    except sqlite3.IntegrityError:
        print(f"⚠️ Already marked today: {student_name} ({class_name})")

    conn.close()


# =========================
# FETCH ALL ATTENDANCE
# =========================
def get_attendance():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT student_name, class_name, date, time 
    FROM attendance
    ORDER BY date DESC, class_name
    """)

    records = cursor.fetchall()
    conn.close()
    return records


# =========================
# FETCH CLASS-WISE
# =========================
def get_class_attendance(class_name):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT student_name, date, time 
    FROM attendance
    WHERE class_name = ?
    ORDER BY date DESC
    """, (class_name,))

    records = cursor.fetchall()
    conn.close()
    return records


# =========================
# RUN ONCE TO CREATE TABLES
# =========================
if __name__ == "__main__":
    create_tables()