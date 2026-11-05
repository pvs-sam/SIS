"""Role-Based Access Control (RBAC) Matrix & Tampering Test Suite.

Tests: Role × Endpoint × (Own Dept vs Other Dept) asserting 200 / 403 responses
and zero cross-department data leakage.
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import app
from data_store import get_store


class RbacMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = get_store()
        # Find CSE student and ECE student for URL tampering tests
        cse_stu = cls.store.students[cls.store.students["Department"] == "CSE"].iloc[0]["Enrollment_No"]
        ece_stu = cls.store.students[cls.store.students["Department"] == "ECE"].iloc[0]["Enrollment_No"]
        cls.cse_student_id = cse_stu
        cls.ece_student_id = ece_stu

    def test_admin_has_unrestricted_access(self):
        client = app.test_client()
        with client:
            client.post("/login", data={"username": "admin", "password": "admin123"})
            # Access dashboard, students, both CSE and ECE profiles, and fees
            self.assertEqual(client.get("/dashboard").status_code, 200)
            self.assertEqual(client.get(f"/student/{self.cse_student_id}").status_code, 200)
            self.assertEqual(client.get(f"/student/{self.ece_student_id}").status_code, 200)
            self.assertEqual(client.get("/fees").status_code, 200)
            self.assertEqual(client.get("/risk").status_code, 200)

    def test_hod_department_scoping_and_url_tampering(self):
        client = app.test_client()
        with client:
            # Login as HOD of CSE
            client.post("/login", data={"username": "hod_cse", "password": "hod123"})
            
            # 1. HOD accessing own department student profile -> 200 OK
            res_own = client.get(f"/student/{self.cse_student_id}")
            self.assertEqual(res_own.status_code, 200)

            # 2. HOD attempting to tamper URL and view another department student profile -> 403 Forbidden!
            res_other = client.get(f"/student/{self.ece_student_id}")
            self.assertEqual(res_other.status_code, 403)

            # 3. HOD students list should contain ZERO cross-department students
            students_res = client.get("/api/students")
            self.assertEqual(students_res.status_code, 200)
            items = students_res.get_json()["items"]
            for item in items:
                self.assertEqual(item["Department"], "CSE", "Data leak! HOD received cross-department student data")

    def test_faculty_scoping_and_fee_isolation(self):
        client = app.test_client()
        with client:
            # Login as Faculty (fac_001 assigned to CSE-A / CSE-B)
            client.post("/login", data={"username": "fac_001", "password": "faculty123"})

            # 1. Faculty accessing assigned student profile -> 200 OK
            res_assigned = client.get(f"/student/{self.cse_student_id}")
            self.assertEqual(res_assigned.status_code, 200)

            # 2. Faculty attempting to access fees page -> 403 Forbidden!
            res_fees = client.get("/fees")
            self.assertEqual(res_fees.status_code, 403)

            # 3. Faculty attempting to export fee reports -> 403 Forbidden!
            res_fee_export = client.get("/reports/export?report=fees")
            self.assertEqual(res_fee_export.status_code, 403)

            # 4. Faculty attempting to access cross-department student -> 403 Forbidden!
            res_other = client.get(f"/student/{self.ece_student_id}")
            self.assertEqual(res_other.status_code, 403)


if __name__ == "__main__":
    unittest.main()

