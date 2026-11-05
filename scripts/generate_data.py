"""Synthetic Data Generator for ERP Data Warehouse (Semester 1–5 Institutional History).

Generates realistic, seeded (random_state=42), reproducible academic data for 1,000 students:
- D1: Date of Birth (2003-08-01 to 2006-12-31, mode in 2005, admission age 17–20).
- D2: Faculty (~58 faculty, 1 HOD per dept, 10–18 h weekly load) & Notices (~40 notices, 5 urgent).
- D3: Department-scoped sections (CSE-A, CSE-B, etc., ~41–42 students balanced).
- D4: Realistic Beta-distributed attendance (~81% mean, ~14% below 75%), realistic correlated marks.
- D5: Raw facts stored only (Classes_Held & Classes_Attended; Internal_Marks & External_Marks).
- D6: 5 historical terms (Sem 1–5, Batch 2023–27, as-of 2026-01-15).
- D9: Library transactions (~8–10/student, ~6% overdue) and Admissions records.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = BASE_DIR / "data"
CATALOG_DIR = DATA_DIR / "catalog"
ETL_PACKAGE_DIR = BASE_DIR / "erp_semester5_etl_package"

CATALOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
ETL_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)


def build_subject_catalog() -> pd.DataFrame:
    """Create comprehensive 5-semester course catalog across 6 engineering departments."""
    subjects = []
    
    # Common First Year Courses (Semesters 1 & 2)
    sem1_common = [
        ("BSC101", "Linear Algebra and Calculus", "BSC", 1, 4, "Core"),
        ("BSC102", "Engineering Physics", "BSC", 1, 4, "Core"),
        ("ESC101", "Problem Solving and Python Programming", "ESC", 1, 4, "Core"),
        ("ESC102", "Engineering Graphics and Design", "ESC", 1, 3, "Core"),
        ("HSMC101", "Communicative English", "HSMC", 1, 3, "Core"),
        ("ESC103", "Physics and Python Laboratory", "ESC", 1, 2, "Lab"),
    ]
    sem2_common = [
        ("BSC201", "Differential Equations and Transform Calculus", "BSC", 2, 4, "Core"),
        ("BSC202", "Engineering Chemistry", "BSC", 2, 4, "Core"),
        ("ESC201", "Programming for Problem Solving in C", "ESC", 2, 4, "Core"),
        ("ESC202", "Basic Electrical and Electronics Engineering", "ESC", 2, 3, "Core"),
        ("HSMC202", "Universal Human Values and Professional Ethics", "HSMC", 2, 3, "Core"),
        ("ESC203", "Engineering Workshop and C Lab", "ESC", 2, 2, "Lab"),
    ]

    for dept in config.DEPARTMENTS:
        for code, name, d, sem, cr, ctype in sem1_common:
            subjects.append({
                "Subject_Code": f"{dept}-{code}",
                "Subject_Name": name,
                "Department": dept,
                "Semester": sem,
                "Credits": cr,
                "Type": ctype
            })
        for code, name, d, sem, cr, ctype in sem2_common:
            subjects.append({
                "Subject_Code": f"{dept}-{code}",
                "Subject_Name": name,
                "Department": dept,
                "Semester": sem,
                "Credits": cr,
                "Type": ctype
            })

    # Department-Specific Courses for Semesters 3, 4, and 5
    dept_curriculum = {
        "CSE": {
            3: [
                ("CS301", "Data Structures and Algorithms", 4, "Core"),
                ("CS302", "Discrete Mathematical Structures", 4, "Core"),
                ("CS303", "Digital Logic and Computer Organization", 4, "Core"),
                ("CS304", "Object Oriented Programming via Java", 3, "Core"),
                ("CS305", "Data Structures Laboratory", 2, "Lab"),
            ],
            4: [
                ("CS401", "Design and Analysis of Algorithms", 4, "Core"),
                ("CS402", "Operating Systems Principles", 4, "Core"),
                ("CS403", "Database Management Systems", 4, "Core"),
                ("CS404", "Formal Languages and Automata Theory", 3, "Core"),
                ("CS405", "DBMS and OS Laboratory", 2, "Lab"),
            ],
            5: [
                ("CS501", "Data Structures", 4, "Core"),
                ("CS502", "Database Management Systems", 4, "Core"),
                ("CS503", "Operating Systems", 4, "Core"),
                ("CS504", "Computer Networks", 4, "Core"),
                ("CS505", "Software Engineering", 4, "Core"),
            ],
        },
        "ECE": {
            3: [
                ("EC301", "Electronic Devices and Circuits", 4, "Core"),
                ("EC302", "Signals and Systems", 4, "Core"),
                ("EC303", "Network Theory", 4, "Core"),
                ("EC304", "Digital System Design", 3, "Core"),
                ("EC305", "Analog Electronics Laboratory", 2, "Lab"),
            ],
            4: [
                ("EC401", "Analog Communication Systems", 4, "Core"),
                ("EC402", "Linear Integrated Circuits", 4, "Core"),
                ("EC403", "Electromagnetic Fields and Waves", 4, "Core"),
                ("EC404", "Microprocessors and Microcontrollers", 3, "Core"),
                ("EC405", "Microcontroller Laboratory", 2, "Lab"),
            ],
            5: [
                ("EC501", "Signals and Systems", 4, "Core"),
                ("EC502", "Analog Circuits", 4, "Core"),
                ("EC503", "Digital Electronics", 4, "Core"),
                ("EC504", "Communication Systems", 4, "Core"),
                ("EC505", "Microprocessors", 4, "Core"),
            ],
        },
        "EEE": {
            3: [
                ("EE301", "Electric Circuit Analysis", 4, "Core"),
                ("EE302", "DC Machines and Transformers", 4, "Core"),
                ("EE303", "Analog Electronics", 4, "Core"),
                ("EE304", "Electromagnetic Fields", 3, "Core"),
                ("EE305", "Electrical Machines Laboratory I", 2, "Lab"),
            ],
            4: [
                ("EE401", "AC Electrical Machines", 4, "Core"),
                ("EE402", "Power Electronics and Drives", 4, "Core"),
                ("EE403", "Digital Electronics and Logic Design", 4, "Core"),
                ("EE404", "Electrical Power Generation and Transmission", 3, "Core"),
                ("EE405", "Power Electronics Laboratory", 2, "Lab"),
            ],
            5: [
                ("EE501", "Circuit Theory", 4, "Core"),
                ("EE502", "Electrical Machines", 4, "Core"),
                ("EE503", "Power Electronics", 4, "Core"),
                ("EE504", "Control Systems", 4, "Core"),
                ("EE505", "Power Systems", 4, "Core"),
            ],
        },
        "IT": {
            3: [
                ("IT301", "Data Structures and OOP", 4, "Core"),
                ("IT302", "Computer Organization and Architecture", 4, "Core"),
                ("IT303", "Discrete Mathematics for IT", 4, "Core"),
                ("IT304", "Foundations of Web Technologies", 3, "Core"),
                ("IT305", "Web Programming Laboratory", 2, "Lab"),
            ],
            4: [
                ("IT401", "Design and Analysis of Algorithms", 4, "Core"),
                ("IT402", "Relational Database Management Systems", 4, "Core"),
                ("IT403", "Operating System Concepts", 4, "Core"),
                ("IT404", "Computer Communication Networks", 3, "Core"),
                ("IT405", "Database and Networking Lab", 2, "Lab"),
            ],
            5: [
                ("IT501", "Web Technologies", 4, "Core"),
                ("IT502", "Cloud Computing", 4, "Core"),
                ("IT503", "Data Mining", 4, "Core"),
                ("IT504", "Software Engineering", 4, "Core"),
                ("IT505", "Data Structures", 4, "Core"),
            ],
        },
        "AIDS": {
            3: [
                ("AD301", "Foundations of Data Science", 4, "Core"),
                ("AD302", "Data Structures for AI", 4, "Core"),
                ("AD303", "Linear Algebra and Optimization for AI", 4, "Core"),
                ("AD304", "Statistical Methods for Data Analysis", 3, "Core"),
                ("AD305", "Data Analytics Lab with Python", 2, "Lab"),
            ],
            4: [
                ("AD401", "Database Systems and Data Warehousing", 4, "Core"),
                ("AD402", "Introduction to Machine Learning", 4, "Core"),
                ("AD403", "Design and Analysis of Algorithms", 4, "Core"),
                ("AD404", "Exploratory Data Analysis and Visualization", 3, "Core"),
                ("AD405", "Machine Learning Lab", 2, "Lab"),
            ],
            5: [
                ("AD501", "Python for Data Science", 4, "Core"),
                ("AD502", "Machine Learning", 4, "Core"),
                ("AD503", "Data Management", 4, "Core"),
                ("AD504", "Statistics for AI", 4, "Core"),
                ("AD505", "Artificial Intelligence", 4, "Core"),
            ],
        },
        "AIML": {
            3: [
                ("AL301", "Mathematical Foundations for Machine Learning", 4, "Core"),
                ("AL302", "Data Structures and Algorithms for AI", 4, "Core"),
                ("AL303", "Logic and Problem Solving in AI", 4, "Core"),
                ("AL304", "Object Oriented Programming for AI", 3, "Core"),
                ("AL305", "AI Programming Laboratory", 2, "Lab"),
            ],
            4: [
                ("AL401", "Machine Learning Fundamentals", 4, "Core"),
                ("AL402", "Computer Vision Principles", 4, "Core"),
                ("AL403", "Natural Language Processing Basics", 4, "Core"),
                ("AL404", "Database Systems for Machine Learning", 3, "Core"),
                ("AL405", "ML and NLP Laboratory", 2, "Lab"),
            ],
            5: [
                ("AL501", "Artificial Intelligence", 4, "Core"),
                ("AL502", "Machine Learning", 4, "Core"),
                ("AL503", "Deep Learning", 4, "Core"),
                ("AL504", "Natural Language Processing", 4, "Core"),
                ("AL505", "Python for Data Science", 4, "Core"),
            ],
        },
    }

    for dept, sems in dept_curriculum.items():
        for sem, courses in sems.items():
            for code, name, cr, ctype in courses:
                subjects.append({
                    "Subject_Code": f"{dept}-{code}",
                    "Subject_Name": name,
                    "Department": dept,
                    "Semester": sem,
                    "Credits": cr,
                    "Type": ctype
                })

    df = pd.DataFrame(subjects)
    df.to_csv(CATALOG_DIR / "subjects.csv", index=False)
    return df


def generate_faculty() -> pd.DataFrame:
    """Generate ~58 faculty members across 6 departments with 1 HOD each."""
    faculty_list = []
    first_names = [
        "Ramesh", "Suresh", "Lakshmi", "Ananya", "Venkatesh", "Pooja", "Srikanth", "Divya",
        "Kiran", "Meenakshi", "Rajesh", "Sunita", "Harish", "Deepa", "Manish", "Kavita",
        "Prasad", "Swathi", "Gopal", "Padma", "Naveen", "Sudha", "Ravinder", "Archana",
        "Subhash", "Sandhya", "Ashok", "Madhavi", "Srinivas", "Bhavani", "Mahesh", "Geeta",
        "Vijay", "Aparna", "Raghavan", "Usha", "Satish", "Vani", "Anil", "Rekha", "Murali",
        "Sujatha", "Pradeep", "Hemalatha", "Chandra", "Sarada", "Praveen", "Kalyani",
        "Babu", "Sravani", "Shiva", "Bindu", "Sanjay", "Lavanya", "Shekhar", "Rohini",
        "Nagesh", "Yamini"
    ]
    last_names = [
        "Sharma", "Rao", "Reddy", "Verma", "Murthy", "Gupta", "Nair", "Iyer",
        "Chowdary", "Patel", "Joshi", "Bhatt", "Deshmukh", "Pillai", "Kulkarni", "Naidu"
    ]

    fac_idx = 1
    for dept_idx, dept in enumerate(config.DEPARTMENTS):
        # 9 to 10 faculty per dept -> 6 * 9.67 ~ 58
        count = 10 if dept in {"CSE", "ECE", "IT", "AIDS"} else 9
        
        # Designations: 1 Professor HOD, 1 other Professor (~20%), 2-3 Assoc Prof (~25%), 5-6 Asst Prof (~55%)
        designations = ["Professor (HOD)"]
        designations.append("Professor")
        designations.extend(["Associate Professor"] * 2)
        designations.extend(["Assistant Professor"] * (count - 4))

        for desig in designations:
            is_hod = "HOD" in desig
            clean_desig = "Professor" if is_hod else desig
            fn = first_names[(fac_idx * 3 + dept_idx) % len(first_names)]
            ln = last_names[(fac_idx * 5 + dept_idx) % len(last_names)]
            name = f"Dr. {fn} {ln}" if clean_desig == "Professor" else f"{fn} {ln}"
            fac_id = f"FAC{fac_idx:03d}"
            email = f"fac{fac_idx:03d}@ifheindia.org"
            phone = f"98{fac_idx:02d}{1000 + fac_idx * 17:04d}"
            
            # Weekly load: HOD <= 10 h, others 12–18 h
            weekly_load = np.random.randint(6, 11) if is_hod else np.random.randint(12, 19)

            faculty_list.append({
                "Faculty_ID": fac_id,
                "Name": name,
                "Email": email,
                "Phone_No": phone,
                "Department": dept,
                "Designation": clean_desig,
                "Is_HOD": is_hod,
                "Weekly_Load_Hours": weekly_load
            })
            fac_idx += 1

    df = pd.DataFrame(faculty_list)
    df.to_csv(DATA_DIR / "faculty.csv", index=False)
    return df


def assign_courses_to_faculty(faculty_df: pd.DataFrame, subjects_df: pd.DataFrame) -> pd.DataFrame:
    """Assign exactly one faculty from the department to each (Subject, Section, Term)."""
    assignments = []
    sem5_subjects = subjects_df[subjects_df["Semester"] == 5]

    for dept in config.DEPARTMENTS:
        dept_fac = faculty_df[faculty_df["Department"] == dept]["Faculty_ID"].tolist()
        dept_subs = sem5_subjects[sem5_subjects["Department"] == dept]["Subject_Code"].tolist()

        f_ptr = 0
        for sec in config.SECTIONS:
            sec_name = f"{dept}-{sec}"
            for sub_code in dept_subs:
                assigned_fac = dept_fac[f_ptr % len(dept_fac)]
                f_ptr += 1
                assignments.append({
                    "Subject_Code": sub_code,
                    "Department": dept,
                    "Section": sec_name,
                    "Semester": 5,
                    "Academic_Year": config.CURRENT_ACADEMIC_YEAR,
                    "Faculty_ID": assigned_fac
                })

    df = pd.DataFrame(assignments)
    df.to_csv(DATA_DIR / "course_assignments.csv", index=False)
    return df


def generate_students_and_history(existing_students_csv: Path, subjects_df: pd.DataFrame):
    """Generate 1,000 students with D1 (DOB), D3 (department-scoped sections), and 5-term history."""
    orig_df = pd.read_csv(existing_students_csv)

    # D1: Add realistic Date of Birth
    # Range: 2003-08-01 to 2006-12-31, mode in 2005
    start_dt = datetime.strptime(config.DOB_START, "%Y-%m-%d").date()
    end_dt = datetime.strptime(config.DOB_END, "%Y-%m-%d").date()
    mode_dt = date(config.DOB_MODE_YEAR, 6, 15)
    total_days = (end_dt - start_dt).days
    mode_fraction = (mode_dt - start_dt).days / total_days

    # Generate dates using triangular distribution
    sampled_fractions = np.random.triangular(0.0, mode_fraction, 1.0, size=len(orig_df))
    dobs = []
    admission_date = date(2023, 7, 15)
    as_of = datetime.strptime(config.AS_OF_DATE, "%Y-%m-%d").date()

    for frac in sampled_fractions:
        dob = start_dt + timedelta(days=int(frac * total_days))
        # Ensure age at admission is 17-20
        age_adm = (admission_date - dob).days / 365.25
        if age_adm < config.MIN_AGE_AT_ADMISSION:
            dob = date(2006, np.random.randint(1, 7), np.random.randint(1, 28))
        elif age_adm > config.MAX_AGE_AT_ADMISSION:
            dob = date(2003, np.random.randint(8, 13), np.random.randint(1, 28))
        dobs.append(dob.strftime("%Y-%m-%d"))

    orig_df["Date_of_Birth"] = dobs
    orig_df["Batch"] = config.BATCH
    orig_df["Admission_Year"] = 2023

    # D3: Department-scoped section assignment (CSE-A, CSE-B, CSE-C, CSE-D)
    # Balanced: ~41-42 students per section
    section_assignments = {}
    for dept in config.DEPARTMENTS:
        dept_students = orig_df[orig_df["Department"] == dept]["Enrollment_No"].tolist()
        for idx, enr in enumerate(dept_students):
            sec_letter = config.SECTIONS[idx % len(config.SECTIONS)]
            section_assignments[enr] = f"{dept}-{sec_letter}"

    orig_df["Section"] = orig_df["Enrollment_No"].map(section_assignments)

    # -------------------------------------------------------------
    # Generate 5 Terms of History for Each Student
    # -------------------------------------------------------------
    # Latent student academic capability ~ Normal(0, 1)
    student_ability = {enr: np.random.normal(0, 1) for enr in orig_df["Enrollment_No"]}
    # Latent attendance habit ~ Normal(0, 1)
    student_discipline = {
        enr: 0.6 * student_ability[enr] + 0.8 * np.random.normal(0, 1)
        for enr in orig_df["Enrollment_No"]
    }

    all_attendance_rows = []
    all_exam_rows = []
    student_terms = []

    sem_info = [
        (1, "2023-2024", "Odd"),
        (2, "2023-2024", "Even"),
        (3, "2024-2025", "Odd"),
        (4, "2024-2025", "Even"),
        (5, "2025-2026", "Odd"),
    ]

    student_sgpas = {enr: [] for enr in orig_df["Enrollment_No"]}
    student_backlogs = {enr: 0 for enr in orig_df["Enrollment_No"]}

    for sem, ay, term_type in sem_info:
        for _, srow in orig_df.iterrows():
            enr = srow["Enrollment_No"]
            dept = srow["Department"]
            sec = srow["Section"]
            ability = student_ability[enr]
            discipline = student_discipline[enr]

            # Get semester subjects for this dept
            sem_subjects = subjects_df[
                (subjects_df["Department"] == dept) & (subjects_df["Semester"] == sem)
            ]

            sem_credits = 0
            sem_points = 0.0
            sem_failed = 0

            for _, sub in sem_subjects.iterrows():
                sub_code = sub["Subject_Code"]
                credits = sub["Credits"]

                # 1. Attendance generation (Beta-shaped, institutional mean ~81-82%, ~14% below 75%)
                target_mean = float(np.clip(0.818 + 0.070 * discipline + np.random.normal(0, 0.012), 0.54, 0.98))
                classes_held = int(np.random.randint(config.CLASSES_HELD_MIN, config.CLASSES_HELD_MAX + 1))
                attended = int(np.clip(np.random.binomial(classes_held, target_mean), 0, classes_held))

                all_attendance_rows.append({
                    "Enrollment_No": enr,
                    "Subject_Code": sub_code,
                    "Semester": sem,
                    "Academic_Year": ay,
                    "Classes_Held": classes_held,
                    "Classes_Attended": attended,
                })

                # 2. Examination marks generation (D4) with realistic real-world variance
                # Internal marks ~ N(28, 6) out of 40, correlated with attendance & ability
                att_ratio = attended / classes_held
                internal_mu = 28.2 + 3.0 * ability + 5.0 * (att_ratio - 0.81) + np.random.normal(0, 4.5)
                internal_marks = int(np.clip(round(internal_mu), 10, 40))

                # External marks ~ N(38, 10) out of 60, correlated with internal marks & ability + real-world test-day noise
                exam_day_noise = np.random.choice([0, -8, 5], p=[0.85, 0.08, 0.07])
                ext_mu = 38.8 + 3.8 * ability + 0.32 * (internal_marks - 28.2) + exam_day_noise + np.random.normal(0, 7.5)
                external_marks = int(np.clip(round(ext_mu), 10, 60))

                total_marks = internal_marks + external_marks
                grade, pt = config.calculate_grade(total_marks)

                if total_marks < config.TOTAL_PASS_MARKS:
                    sem_failed += 1
                    student_backlogs[enr] += 1

                sem_credits += credits
                sem_points += credits * pt

                all_exam_rows.append({
                    "Enrollment_No": enr,
                    "Subject_Code": sub_code,
                    "Semester": sem,
                    "Academic_Year": ay,
                    "Internal_Marks": internal_marks,
                    "External_Marks": external_marks,
                })

            sgpa = round(sem_points / max(sem_credits, 1), 2)
            student_sgpas[enr].append(sgpa)

            student_terms.append({
                "Enrollment_No": enr,
                "Semester": sem,
                "Academic_Year": ay,
                "Term": term_type,
                "Department": dept,
                "Section": sec,
                "Credits_Registered": sem_credits,
                "SGPA": sgpa,
                "Backlogs_Incurred": sem_failed,
                "Status": "Completed",
            })

    # Update cumulative CGPA in students
    orig_df["CGPA"] = [
        round(float(np.mean(student_sgpas[enr])), 2) for enr in orig_df["Enrollment_No"]
    ]
    orig_df["Semester"] = config.CURRENT_SEMESTER
    orig_df["Academic_Year"] = config.CURRENT_ACADEMIC_YEAR

    # Save Student Dimension and Term Histories
    orig_df.to_csv(DATA_DIR / "students.csv", index=False)
    orig_df.to_csv(ETL_PACKAGE_DIR / "students.csv", index=False)
    # Also save as students (2).csv for legacy compatibility
    orig_df.to_csv(ETL_PACKAGE_DIR / "students (2).csv", index=False)

    df_terms = pd.DataFrame(student_terms)
    df_terms.to_csv(DATA_DIR / "student_terms.csv", index=False)
    df_terms.to_csv(ETL_PACKAGE_DIR / "student_terms.csv", index=False)

    df_att = pd.DataFrame(all_attendance_rows)
    df_att.to_csv(DATA_DIR / "attendance_history.csv", index=False)
    # Save Semester 5 attendance to authoritative ETL package
    sem5_att = df_att[df_att["Semester"] == 5].copy()
    sem5_att["Attendance_Percentage"] = (
        (sem5_att["Classes_Attended"] / sem5_att["Classes_Held"]) * 100
    ).round().astype(int)
    # Map Subject_Code to Subject name for backward-compatibility if needed
    sub_map = dict(zip(subjects_df["Subject_Code"], subjects_df["Subject_Name"]))
    sem5_att["Subject"] = sem5_att["Subject_Code"].map(sub_map)
    sem5_att.to_csv(ETL_PACKAGE_DIR / "attendance.csv", index=False)
    sem5_att.to_csv(DATA_DIR / "attendance.csv", index=False)

    df_exam = pd.DataFrame(all_exam_rows)
    df_exam.to_csv(DATA_DIR / "examination_history.csv", index=False)
    # Save Semester 5 examination to authoritative ETL package
    sem5_exam = df_exam[df_exam["Semester"] == 5].copy()
    sem5_exam["Total_Marks"] = sem5_exam["Internal_Marks"] + sem5_exam["External_Marks"]
    sem5_exam["Grade"] = sem5_exam["Total_Marks"].apply(lambda m: config.calculate_grade(m)[0])
    sem5_exam["Subject"] = sem5_exam["Subject_Code"].map(sub_map)
    sem5_exam.to_csv(ETL_PACKAGE_DIR / "examination.csv", index=False)
    sem5_exam.to_csv(DATA_DIR / "examination.csv", index=False)


def generate_fees(students_df: pd.DataFrame):
    """Generate fee records for Semester 5 with realistic payment distribution."""
    # Department baseline tuition fees
    tuition_map = {
        "CSE": 85000,
        "AIML": 85000,
        "AIDS": 80000,
        "IT": 80000,
        "ECE": 75000,
        "EEE": 70000,
    }

    fees_list = []
    for _, srow in students_df.iterrows():
        enr = srow["Enrollment_No"]
        dept = srow["Department"]
        tuition = tuition_map.get(dept, 75000)

        # ~40% hostel (+35,000)
        has_hostel = np.random.rand() < 0.40
        hostel_fee = 35000 if has_hostel else 0

        # Scholarship discount
        has_scholarship = np.random.rand() < 0.15
        discount = 15000 if has_scholarship else 0

        total_fee = tuition + hostel_fee - discount

        # Distribution: ~80% fully paid, ~10% partial, ~6% unpaid, ~4% overdue
        p_roll = np.random.rand()
        if p_roll < 0.80:
            paid = total_fee
            status = "Paid"
            payment_date = "2025-08-20"
        elif p_roll < 0.90:
            paid = int(total_fee * np.random.choice([0.4, 0.5, 0.6, 0.75]))
            status = "Partial"
            payment_date = "2025-09-10"
        elif p_roll < 0.96:
            paid = 0
            status = "Unpaid"
            payment_date = ""
        else:
            paid = int(total_fee * 0.25) if np.random.rand() < 0.5 else 0
            status = "Overdue"
            payment_date = "2025-10-05" if paid > 0 else ""

        pending = total_fee - paid

        fees_list.append({
            "Enrollment_No": enr,
            "Semester": 5,
            "Academic_Year": config.CURRENT_ACADEMIC_YEAR,
            "Tuition_Fee": tuition,
            "Hostel_Fee": hostel_fee,
            "Scholarship_Discount": discount,
            "Total_Fees": total_fee,
            "Paid_Amount": paid,
            "Pending_Amount": pending,
            "Payment_Status": status,
            "Due_Date": "2025-08-30",
            "Payment_Date": payment_date,
        })

    df = pd.DataFrame(fees_list)
    df.to_csv(DATA_DIR / "fees.csv", index=False)
    df.to_csv(ETL_PACKAGE_DIR / "fees.csv", index=False)
    return df


def generate_library(students_df: pd.DataFrame):
    """Generate library issues (~8–10 per student per year, ~6% overdue)."""
    books = [
        ("B101", "Introduction to Algorithms - Cormen, Leiserson, Rivest"),
        ("B102", "Database System Concepts - Silberschatz"),
        ("B103", "Operating System Concepts - Galvin"),
        ("B104", "Computer Networks - Tanenbaum"),
        ("B105", "Artificial Intelligence: A Modern Approach - Russell & Norvig"),
        ("B106", "Pattern Recognition and Machine Learning - Bishop"),
        ("B107", "Deep Learning - Goodfellow, Bengio, Courville"),
        ("B108", "Electronic Principles - Malvino"),
        ("B109", "Electric Machinery Fundamentals - Chapman"),
        ("B110", "Digital Signal Processing - Proakis"),
        ("B111", "Software Engineering: A Practitioner's Approach - Pressman"),
        ("B112", "Python Data Science Handbook - VanderPlas"),
        ("B113", "Clean Code - Robert C. Martin"),
        ("B114", "Design Patterns - Gang of Four"),
        ("B115", "Control Systems Engineering - Nise"),
    ]

    records = []
    tx_id = 1
    base_date = date(2025, 8, 5)

    for _, srow in students_df.iterrows():
        enr = srow["Enrollment_No"]
        num_issues = np.random.randint(5, 12)  # Active issues/returns in Sem 5

        for _ in range(num_issues):
            b_idx = np.random.randint(0, len(books))
            b_id, b_title = books[b_idx]
            offset_days = np.random.randint(0, 140)
            issue_date = base_date + timedelta(days=offset_days)
            due_date = issue_date + timedelta(days=21)

            # ~6% overdue
            is_overdue = np.random.rand() < 0.06
            if is_overdue:
                status = "Overdue"
                return_date = ""
            elif np.random.rand() < 0.25:
                status = "Active"
                return_date = ""
            else:
                status = "Returned"
                return_date = (issue_date + timedelta(days=np.random.randint(7, 21))).strftime("%Y-%m-%d")

            records.append({
                "Transaction_ID": f"LIB{tx_id:06d}",
                "Enrollment_No": enr,
                "Book_ID": b_id,
                "Book_Title": b_title,
                "Issue_Date": issue_date.strftime("%Y-%m-%d"),
                "Due_Date": due_date.strftime("%Y-%m-%d"),
                "Return_Date": return_date,
                "Status": status,
            })
            tx_id += 1

    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "library_transactions.csv", index=False)
    return df


def generate_notices():
    """Generate ~40 institutional notices with 5 urgent/active notices."""
    notices = [
        ("End Semester Examination Schedule Released", "The timetable for Semester 5 Final External Examinations has been uploaded. Practical exams commence on Dec 15.", "Examination", "ALL", "ALL", "Urgent", "Controller of Examinations", "2025-11-20", True),
        ("Mandatory Attendance Condonation Fee Notice", "Students with attendance between 65% and 74% must submit medical records for condonation approval by Dec 05.", "Academic", "ALL", "ALL", "Urgent", "Dean Academics", "2025-11-25", True),
        ("Semester 5 Tuition Fee Clearances", "All pending dues for Semester 5 must be settled before hall ticket distribution. Check Accounts Section.", "Fee", "ALL", "ALL", "Urgent", "Finance Officer", "2025-11-28", True),
        ("Smart India Hackathon 2025 Shortlisted Teams", "Hearty congratulations to the 12 shortlisted teams representing IFHE in the national grand finale.", "Academic", "ALL", "ALL", "Normal", "Dean R&D", "2025-11-15", True),
        ("Urgent: Library Overdue Books Return Window", "Late penalty waiver active until Dec 10 for all overdue monograph returns.", "Administrative", "ALL", "ALL", "Urgent", "Chief Librarian", "2025-11-30", True),
        ("Campus Recruitment Drive - Top Product Companies", "Phase 1 placements drive for pre-final years begins in January 2026. Register on the CDC portal.", "Placement", "ALL", "ALL", "Normal", "Head Placements", "2025-11-10", True),
        ("CSE Dept Seminar on Large Language Models", "Guest lecture by Dr. S. Rao from Google DeepMind on scalable alignment and agent architectures.", "Academic", "CSE", "CSE-A", "Normal", "HOD CSE", "2025-11-12", True),
        ("ECE Workshop on Embedded IoT and FPGA", "Hands-on two-day boot camp with ARM Cortex boards in Electronics Laboratory 3.", "Academic", "ECE", "ALL", "Normal", "HOD ECE", "2025-11-18", True),
        ("EEE Industrial Visit to 400kV Substation", "Site visit scheduled on Saturday for power systems students. Bus departs at 8:30 AM from campus.", "Academic", "EEE", "ALL", "Normal", "HOD EEE", "2025-11-05", True),
        ("AI & Data Science Hackathon Registration", "24-hour algorithmic prediction sprint sponsored by industry partners. Cash prize worth 1 Lakh.", "Academic", "AIDS", "ALL", "Normal", "HOD AIDS", "2025-11-22", True),
    ]

    # Add remaining routine notices over 4 months
    categories = ["Academic", "Examination", "Fee", "Administrative", "Events"]
    depts = ["ALL", "CSE", "ECE", "EEE", "IT", "AIDS", "AIML"]
    base_dt = date(2025, 8, 1)

    while len(notices) < 40:
        idx = len(notices) + 1
        cat = categories[idx % len(categories)]
        dept = depts[idx % len(depts)]
        p_dt = base_dt + timedelta(days=int(idx * 2.8))
        is_urgent = idx in {11, 23}
        notices.append((
            f"Institutional Circular #{idx}: Update on {cat} Policies",
            f"Official memorandum regarding {cat.lower()} operations, room schedules, and student guidelines for the current term.",
            cat,
            dept,
            "ALL",
            "Urgent" if is_urgent else "Normal",
            f"Department Coordinator ({dept})",
            p_dt.strftime("%Y-%m-%d"),
            True if p_dt >= date(2025, 11, 1) else False
        ))

    df = pd.DataFrame(notices, columns=[
        "Title", "Content", "Category", "Department", "Target_Section",
        "Priority", "Published_By", "Published_Date", "Is_Active"
    ])
    df.insert(0, "Notice_ID", [f"NOT{i+1:04d}" for i in range(len(df))])
    df.to_csv(DATA_DIR / "notices.csv", index=False)
    return df


def generate_admissions(students_df: pd.DataFrame):
    """Generate admissions intake and entrance exam ranks."""
    records = []
    categories = ["General", "OBC", "SC", "ST", "EWS"]
    quotas = ["Merit", "Management", "State Quota", "NRI"]

    for _, srow in students_df.iterrows():
        enr = srow["Enrollment_No"]
        rank = np.random.randint(1200, 24500)
        cat = np.random.choice(categories, p=[0.45, 0.30, 0.12, 0.05, 0.08])
        quota = np.random.choice(quotas, p=[0.60, 0.20, 0.15, 0.05])
        adm_dt = date(2023, np.random.randint(6, 8), np.random.randint(1, 28))

        records.append({
            "Enrollment_No": enr,
            "Entrance_Exam": "State EAMCET / JEE Main",
            "Entrance_Rank": rank,
            "Category": cat,
            "Admission_Quota": quota,
            "Admission_Date": adm_dt.strftime("%Y-%m-%d"),
            "12th_Percentage": round(float(np.random.normal(86.0, 6.5)), 1),
        })

    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "admissions.csv", index=False)
    return df


def main():
    print("=" * 60)
    print("  SYNTHETIC DATASET GENERATION (SEMESTERS 1–5)")
    print("=" * 60)

    print("\n[1] Building Subject Catalog...")
    subjects_df = build_subject_catalog()
    print(f"  Generated {len(subjects_df)} subject offerings across 6 departments & 5 terms.")

    print("\n[2] Generating Faculty Directory & Assignments...")
    faculty_df = generate_faculty()
    assignments_df = assign_courses_to_faculty(faculty_df, subjects_df)
    print(f"  Generated {len(faculty_df)} faculty members; {len(assignments_df)} course assignments.")

    print("\n[3] Generating Students & 5-Semester Historical Records...")
    source_students = ETL_PACKAGE_DIR / "students (2).csv"
    if not source_students.exists():
        source_students = ETL_PACKAGE_DIR / "students.csv"
    generate_students_and_history(source_students, subjects_df)
    print("  Generated 1,000 students with DOB, department-scoped sections, and Sem 1–5 history.")

    students_df = pd.read_csv(DATA_DIR / "students.csv")

    print("\n[4] Generating Fee Records...")
    fees_df = generate_fees(students_df)
    print(f"  Generated {len(fees_df)} fee rows.")

    print("\n[5] Generating Library Records...")
    lib_df = generate_library(students_df)
    print(f"  Generated {len(lib_df)} library circulation transactions.")

    print("\n[6] Generating Notices...")
    notices_df = generate_notices()
    print(f"  Generated {len(notices_df)} institutional notices (5 urgent/active).")

    print("\n[7] Generating Admissions Records...")
    adm_df = generate_admissions(students_df)
    print(f"  Generated {len(adm_df)} student admission profiles.")

    print("\n[DONE] All synthetic data generated cleanly and deterministically!")

if __name__ == "__main__":
    main()
