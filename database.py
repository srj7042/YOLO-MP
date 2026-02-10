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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL
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
    except sqlite3.IntegrityError:
        pass  # student already exists

    conn.close()


# =========================
# MARK ATTENDANCE
# =========================
def mark_attendance(student_name):
    conn = connect_db()
    cursor = conn.cursor()

    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    # prevent duplicate entry on same day
    cursor.execute("""
    SELECT * FROM attendance 
    WHERE student_name = ? AND date = ?
    """, (student_name, date))

    if cursor.fetchone() is None:
        cursor.execute("""
        INSERT INTO attendance (student_name, date, time)
        VALUES (?, ?, ?)
        """, (student_name, date, time))
        conn.commit()
        print(f"🟢 Attendance marked: {student_name}")

    conn.close()


# =========================
# FETCH ATTENDANCE
# =========================
def get_attendance():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()

    conn.close()
    return records


# =========================
# MAIN (RUN ONCE)
# =========================
if __name__ == "__main__":
    create_tables()
