import unittest
import os
import io
import tempfile
import database
import face_engine

class TestYoloMP(unittest.TestCase):

    def setUp(self):
        # Set up isolated test database
        self.db_fd, self.db_path = tempfile.mkstemp()
        database.DB_NAME = self.db_path
        database.init_db()

        # Import app and configure testing
        import app
        self.app = app.app
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.client = self.app.test_client()

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    # 1. DATABASE TESTS
    def test_database_crud(self):
        # Add student
        sid = database.add_student("Test Student", "999", "IT")
        self.assertIsNotNone(sid)

        # Retrieve student
        student = database.get_student_by_name("Test Student")
        self.assertIsNotNone(student)
        self.assertEqual(student["roll_no"], "999")

        # Duplicate student should return existing ID without crashing
        sid_dup = database.add_student("Test Student", "999", "IT")
        self.assertEqual(sid, sid_dup)

        # Mark attendance
        success, msg = database.mark_attendance(sid)
        self.assertTrue(success)

        # Marking again same day should be detected as already marked
        success2, msg2 = database.mark_attendance(sid)
        self.assertFalse(success2)
        self.assertIn("Already marked", msg2)

        # Check records
        records = database.get_attendance_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["name"], "Test Student")

        # Stats check
        stats = database.get_stats()
        self.assertEqual(stats["total_students"], 1)
        self.assertEqual(stats["present_today"], 1)
        self.assertEqual(stats["absent_today"], 0)
        self.assertEqual(stats["attendance_percentage"], 100.0)

        # Delete student
        deleted = database.delete_student(sid)
        self.assertTrue(deleted)
        self.assertIsNone(database.get_student_by_id(sid))
        self.assertEqual(len(database.get_attendance_records()), 0)

    # 2. FACE ENGINE TESTS
    def test_face_engine_empty_state(self):
        # Should gracefully return empty when no faces exist
        data, names, vectors = face_engine.load_known_faces()
        self.assertIsInstance(data, dict)
        self.assertIsInstance(names, list)

    # 3. AUTHENTICATION & ROUTE TESTS
    def test_auth_and_routes(self):
        # Unauthenticated request to /dashboard should redirect to /
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/", resp.headers["Location"])

        # Invalid login
        resp = self.client.post("/", data={"role": "admin", "password": "wrongpassword"}, follow_redirects=True)
        self.assertIn(b"Invalid role or password", resp.data)

        # Valid login as Admin
        resp = self.client.post("/", data={"role": "admin", "password": "admin123"}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Dashboard", resp.data)

        # Dashboard loads with 200
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Attendance Logs", resp.data)

        # Live feed page loads with 200
        resp = self.client.get("/live")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Live Camera Attendance", resp.data)

        # Upload page loads with 200
        resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Process Classroom Group Photo", resp.data)

        # Students directory loads with 200
        resp = self.client.get("/students")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Student Directory", resp.data)

        # Export CSV returns 200 and text/csv
        resp = self.client.get("/export/csv")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["Content-Type"], "text/csv; charset=utf-8")

        # Export Excel returns 200
        resp = self.client.get("/export/excel")
        self.assertEqual(resp.status_code, 200)

        # Logout works
        resp = self.client.get("/logout", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Sign In", resp.data)

if __name__ == "__main__":
    unittest.main()
