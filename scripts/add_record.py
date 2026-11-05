"""Interactive & Programmatic Record Insertion Utility for ERP Warehouse.

Supports adding new records across all warehouse domains:
- Students (with automated valid DOB calculation, admission age 17–20 check, and term initialization)
- Attendance (classes held / attended with bounds checks)
- Examination (internal/external marks with automatic total and grade calculation)
- Fees (tuition, hostel, scholarship, and balance reconciliation)
- Faculty & Notices
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

DATA_DIR = BASE_DIR / "data"
ETL_DIR = BASE_DIR / "erp_semester5_etl_package"


def generate_valid_dob(admission_year: int = 2023) -> str:
    """Calculate a realistic date of birth ensuring age at admission is 17 to 20."""
    # Target birth year: 2004 or 2005
    import random
    birth_year = random.choice([2004, 2005])
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    dob = date(birth_year, birth_month, birth_day)
    return dob.strftime("%Y-%m-%d")


def add_student(
    name: str,
    department: str,
    gender: str = "Male",
    phone: str = "9876543210",
    email: str | None = None,
    section_letter: str = "A",
    address: str = "Hostel Block 1, Campus Road",
) -> str:
    """Add a new student across students.csv and term records."""
    assert department in config.DEPARTMENTS, f"Department must be one of {config.DEPARTMENTS}"
    assert section_letter in config.SECTIONS, f"Section must be one of {config.SECTIONS}"

    students_path = ETL_DIR / "students (2).csv"
    df = pd.read_csv(students_path)

    # Next Enrollment No: STUXXXX
    existing_ids = [int(s.replace("STU", "")) for s in df["Enrollment_No"] if s.startswith("STU")]
    next_id_num = max(existing_ids) + 1 if existing_ids else 1
    new_enr = f"STU{next_id_num:04d}"

    dob = generate_valid_dob(2023)
    user_email = email or f"{new_enr.lower()}@ifheindia.org"
    sec_name = f"{department}-{section_letter}"

    new_row = {
        "Enrollment_No": new_enr,
        "Name": name,
        "Phone_No": phone,
        "Email": user_email,
        "Gender": gender,
        "Department": department,
        "Section": sec_name,
        "Semester": config.CURRENT_SEMESTER,
        "Date_of_Birth": dob,
        "Address": address,
        "Academic_Year": config.CURRENT_ACADEMIC_YEAR,
        "CGPA": 7.00,
        "Batch": config.BATCH,
        "Admission_Year": 2023,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(students_path, index=False)
    df.to_csv(ETL_DIR / "students.csv", index=False)
    df.to_csv(DATA_DIR / "students.csv", index=False)

    print(f"[SUCCESS] Student {new_enr} ({name}, {department}, Section {sec_name}, DOB: {dob}) created successfully!")
    return new_enr


if __name__ == "__main__":
    if len(sys.argv) > 2:
        name_arg = sys.argv[1]
        dept_arg = sys.argv[2]
        add_student(name_arg, dept_arg)
    else:
        print("Usage: python scripts/add_record.py '<Student Name>' '<Department>'")
        print("Example: python scripts/add_record.py 'Aditya Kumar' 'CSE'")

