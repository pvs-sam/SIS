"""ERP Data Warehouse & Institutional Decision Support System (Flask Application).

Features:
- Multi-dataset integration across Students, Terms, Attendance, Examination, Fees,
  Faculty, Notices, Library, Admissions, and Machine Learning Early-Warning Risk Models.
- Strict Role-Based Access Control (RBAC): ADMIN, HOD (own department), FACULTY (assigned sections).
- Server-side query layer scoping & direct URL tampering protection.
- In-memory login rate limiting and CSRF token protection.
- Input validation enforcing type integrity.
"""

from __future__ import annotations
import os
import re
import json
from io import BytesIO
from functools import wraps
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    flash,
    session,
    send_file,
    abort,
)

import config
from db_config import DATA_MODE, get_db_config, MYSQL_DATABASE, MYSQL_HOST, MYSQL_PORT, MYSQL_USER
from data_store import get_store
from auth import (
    authenticate_user,
    login_required,
    roles_required,
    check_student_scope,
    get_csrf_token,
    validate_csrf,
    is_rate_limited,
    record_failed_attempt,
    clear_failed_attempts,
    USERS,
)
from ml.predict import RiskPredictorService

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    mysql = None
    Error = Exception

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = config.SESSION_COOKIE_HTTPONLY
app.config["SESSION_COOKIE_SAMESITE"] = config.SESSION_COOKIE_SAMESITE


@app.after_request
def set_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.context_processor
def inject_security_context():
    return {
        "csrf_token": get_csrf_token(),
        "current_user": session.get("user"),
        "config": config,
    }


# ==========================================
# INPUT VALIDATION UTILITIES
# ==========================================
def validate_alpha_only(text: str) -> bool:
    """Validate that text contains only alphabet letters, spaces, hyphens, and periods (no digits)."""
    if not text:
        return True
    return bool(re.match(r"^[a-zA-Z\s\.\-']+$", text))


def validate_numeric_only(text: str) -> bool:
    """Validate that text contains only numeric digits."""
    if not text:
        return True
    return bool(re.match(r"^\d+$", text))


# ==========================================
# DATABASE & DATA STORE ACCESS
# ==========================================
def get_db_connection():
    if DATA_MODE != "mysql":
        raise RuntimeError("MySQL access requested while DATA_MODE is not 'mysql'")
    if mysql is None:
        raise RuntimeError("mysql-connector-python is required when DATA_MODE=mysql")
    try:
        return mysql.connector.connect(**get_db_config(include_db=True))
    except Error as e:
        print(f"[FATAL] MySQL Connection Error: {e}")
        raise


def get_kpis(user: dict | None = None):
    if DATA_MODE == "csv":
        data = get_store()
        kpis = data.kpis(user=user)
        kpis.update({
            "departments_count": len(kpis["departments"]),
            "dept_counts": kpis["department_counts"],
            "avg_cgpa": round(float(data.students["CGPA"].mean()), 2),
            "total_fees": f"{kpis['total_fees']:,}",
            "total_paid": f"{kpis['total_paid']:,}",
            "total_pending": f"{kpis['total_pending']:,}",
            "subjects": sorted(data.attendance["Subject"].unique().tolist()),
        })
        return kpis

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) AS total FROM `students`;")
    total_students = cursor.fetchone()["total"]
    cursor.execute("SELECT `Department`, COUNT(*) AS `count` FROM `students` GROUP BY `Department`;")
    dept_counts = {row["Department"]: row["count"] for row in cursor.fetchall()}
    cursor.execute("SELECT AVG(`Attendance_Percentage`) AS `avg_att` FROM `attendance`;")
    avg_att = round(float(cursor.fetchone()["avg_att"] or 0), 1)
    cursor.execute("SELECT AVG(`Total_Marks`) AS `avg_marks` FROM `examination`;")
    avg_marks = round(float(cursor.fetchone()["avg_marks"] or 0), 1)
    cursor.execute("SELECT AVG(`CGPA`) AS `avg_cgpa` FROM `students`;")
    avg_cgpa = round(float(cursor.fetchone()["avg_cgpa"] or 0), 2)
    cursor.execute("SELECT SUM(`Total_Fees`) AS `total`, SUM(`Paid_Amount`) AS `paid`, SUM(`Pending_Amount`) AS `pending` FROM `fees`;")
    f_totals = cursor.fetchone()
    cursor.close()
    conn.close()

    return {
        "total_students": total_students,
        "departments_count": len(dept_counts),
        "dept_counts": dept_counts,
        "avg_attendance": avg_att,
        "avg_marks": avg_marks,
        "avg_cgpa": avg_cgpa,
        "total_fees": f"{int(f_totals['total'] or 0):,}",
        "total_paid": f"{int(f_totals['paid'] or 0):,}",
        "total_pending": f"{int(f_totals['pending'] or 0):,}",
        "departments": sorted(dept_counts.keys()),
        "sections": ["A", "B", "C", "D"],
        "semesters": [config.CURRENT_SEMESTER],
        "subjects": [],
        "students_requiring_attention": 0,
        "at_risk_students_count": 0,
        "high_risk_count": 0,
    }


def get_integrated_student(enrollment_no: str, user: dict | None = None):
    if DATA_MODE == "csv":
        return get_store().student(enrollment_no, user=user)

    key = enrollment_no.strip().upper()
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM `students` WHERE UPPER(`Enrollment_No`) = %s;", (key,))
    student_row = cursor.fetchone()
    if not student_row:
        cursor.close()
        conn.close()
        return None

    if user:
        role = user.get("role")
        if role == "HOD" and student_row["Department"] != user.get("department"):
            cursor.close()
            conn.close()
            return None
        if role == "FACULTY":
            if student_row["Department"] != user.get("department"):
                cursor.close()
                conn.close()
                return None
            if student_row.get("Section") not in user.get("assigned_sections", []):
                cursor.close()
                conn.close()
                return None

    cursor.execute("SELECT `Subject`, `Attendance_Percentage` FROM `attendance` WHERE UPPER(`Enrollment_No`) = %s;", (key,))
    att_recs = cursor.fetchall()
    avg_att = round(sum(r["Attendance_Percentage"] for r in att_recs) / len(att_recs), 1) if att_recs else 0.0

    cursor.execute("SELECT `Subject`, `Internal_Marks`, `External_Marks`, `Total_Marks`, `Grade` FROM `examination` WHERE UPPER(`Enrollment_No`) = %s;", (key,))
    exam_recs = cursor.fetchall()
    avg_marks = round(sum(r["Total_Marks"] for r in exam_recs) / len(exam_recs), 1) if exam_recs else 0.0

    fee_row = None
    if not user or user.get("role") != "FACULTY":
        cursor.execute("SELECT * FROM `fees` WHERE UPPER(`Enrollment_No`) = %s;", (key,))
        fee_row = cursor.fetchone()

    cursor.close()
    conn.close()

    risk_service = RiskPredictorService.get_instance()
    risk_profile = risk_service.get_student_risk(key)

    return {
        "personal": student_row,
        "attendance": {"records": att_recs, "average": avg_att, "count": len(att_recs)},
        "examination": {"records": exam_recs, "average": avg_marks, "count": len(exam_recs)},
        "fees": fee_row,
        "terms_history": [],
        "library_history": [],
        "admissions": None,
        "ml_risk": risk_profile,
        "source_datasets": [
            {"name": "Dataset 1: Student Information", "file": "MySQL Table: students", "status": "Found"},
            {"name": "Dataset 2: Attendance Records", "file": "MySQL Table: attendance", "status": "Found"},
            {"name": "Dataset 3: Examination Results", "file": "MySQL Table: examination", "status": "Found"},
            {"name": "Dataset 4: Institutional Fees", "file": "MySQL Table: fees", "status": "Found"},
        ],
    }


# ==========================================
# AUTHENTICATION & LOGIN ROUTES
# ==========================================
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        ip_addr = request.remote_addr or "127.0.0.1"

        if is_rate_limited(ip_addr) or is_rate_limited(username):
            flash("Too many failed login attempts. Please wait 5 minutes before retrying.", "danger")
            return render_template("login.html"), 429

        user = authenticate_user(username, password)
        if user:
            clear_failed_attempts(ip_addr)
            clear_failed_attempts(username)
            session.clear()
            session["user"] = {
                "username": user["username"],
                "role": user["role"],
                "name": user["name"],
                "department": user["department"],
                "title": user["title"],
                "faculty_id": user.get("faculty_id"),
                "assigned_sections": user.get("assigned_sections", []),
            }
            session["csrf_token"] = get_csrf_token()
            flash(f"Welcome back, {user['name']} ({user['role']}).", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("dashboard"))
        else:
            record_failed_attempt(ip_addr)
            record_failed_attempt(username)
            flash("Invalid credentials. Please verify your username and password.", "danger")

    if "user" in session:
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


# ==========================================
# DASHBOARD
# ==========================================
def get_dashboard_summary(user: dict | None = None):
    if DATA_MODE == "csv":
        data = get_store()
        students = data.students.copy()
        attendance = data.attendance.copy()
        examination = data.examination.copy()
        fees = data.fees.copy()

        # Scoping based on role
        if user:
            role = user.get("role")
            if role == "HOD":
                students = students[students["Department"] == user.get("department")]
                s_ids = set(students["Enrollment_No"])
                attendance = attendance[attendance["Enrollment_No"].isin(s_ids)]
                examination = examination[examination["Enrollment_No"].isin(s_ids)]
                fees = fees[fees["Enrollment_No"].isin(s_ids)]
            elif role == "FACULTY":
                students = students[students["Section"].isin(user.get("assigned_sections", []))]
                s_ids = set(students["Enrollment_No"])
                attendance = attendance[attendance["Enrollment_No"].isin(s_ids)]
                examination = examination[examination["Enrollment_No"].isin(s_ids)]
                fees = fees.iloc[0:0]

        student_att = attendance.groupby("Enrollment_No")["Attendance_Percentage"].mean()
        grades = examination["Grade"].value_counts()
        departments = []

        for dept in sorted(students["Department"].unique()):
            dept_st = students[students["Department"] == dept]
            dept_fees = fees[fees["Enrollment_No"].isin(dept_st["Enrollment_No"])]
            dept_att = attendance[attendance["Enrollment_No"].isin(dept_st["Enrollment_No"])]
            departments.append({
                "code": dept,
                "name": dept,
                "section": "A–D",
                "students_count": len(dept_st),
                "avg_attendance": round(float(dept_att["Attendance_Percentage"].mean()), 1) if not dept_att.empty else 0.0,
                "semester": int(dept_st["Semester"].dropna().iloc[0]) if not dept_st.empty else 5,
                "total_fees": int(dept_fees["Total_Fees"].sum()),
                "pending_fees": int(dept_fees["Pending_Amount"].sum()),
            })

        # Risk metrics
        risk_service = RiskPredictorService.get_instance()
        risk_df = risk_service.predictions_df
        if risk_df is not None and not risk_df.empty:
            scoped_ids = set(students["Enrollment_No"])
            scoped_risk = risk_df[risk_df["Enrollment_No"].isin(scoped_ids)]
            at_risk_count = int(scoped_risk["Predicted_At_Risk"].sum())
            high_risk_count = int((scoped_risk["Risk_Band"] == "High").sum())
        else:
            at_risk_count = int((student_att < config.ATTENDANCE_WARNING_THRESHOLD).sum())
            high_risk_count = 0

        return {
            "student": {
                "total": len(students),
                "cse": int((students["Department"] == "CSE").sum()),
                "ece": int((students["Department"] == "ECE").sum()),
                "eee": int((students["Department"] == "EEE").sum()),
            },
            "attendance": {
                "overall_avg": round(float(attendance["Attendance_Percentage"].mean()), 1) if not attendance.empty else 0.0,
                "good_count": int((student_att >= config.ATTENDANCE_WARNING_THRESHOLD).sum()),
                "low_count": int((student_att < config.ATTENDANCE_WARNING_THRESHOLD).sum()),
            },
            "examination": {
                "avg_marks": round(float(examination["Total_Marks"].mean()), 1) if not examination.empty else 0.0,
                "highest_marks": int(examination["Total_Marks"].max()) if not examination.empty else 0,
                "lowest_marks": int(examination["Total_Marks"].min()) if not examination.empty else 0,
                "excellent_count": int(grades.get("O", 0) + grades.get("A+", 0)),
                "good_count": int(grades.get("A", 0) + grades.get("B+", 0)),
                "average_count": int(grades.get("B", 0) + grades.get("C", 0) + grades.get("D", 0) + grades.get("F", 0)),
            },
            "fees": {
                "total": f"{int(fees['Total_Fees'].sum()):,}",
                "paid": f"{int(fees['Paid_Amount'].sum()):,}",
                "pending": f"{int(fees['Pending_Amount'].sum()):,}",
                "pending_students": int((fees["Pending_Amount"] > 0).sum()) if not fees.empty else 0,
            },
            "departments": departments,
            "at_risk_count": at_risk_count,
            "high_risk_count": high_risk_count,
        }

    return {
        "student": {"total": 0, "cse": 0, "ece": 0, "eee": 0},
        "attendance": {"overall_avg": 0, "good_count": 0, "low_count": 0},
        "examination": {"avg_marks": 0, "highest_marks": 0, "lowest_marks": 0},
        "fees": {"total": "0", "paid": "0", "pending": "0", "pending_students": 0},
        "departments": [],
        "at_risk_count": 0,
        "high_risk_count": 0,
    }


@app.route("/dashboard")
@login_required
def dashboard():
    user = session.get("user")
    kpis = get_kpis(user=user)
    summary = get_dashboard_summary(user=user)
    return render_template("dashboard.html", kpis=kpis, summary=summary, active_page="dashboard")


# ==========================================
# STUDENT DIRECTORY & PROFILES
# ==========================================
def _csv_module_data(kind: str, section_filter: str = "", subject_filter: str = "", status_filter: str = "", search_q: str = "", user: dict | None = None):
    data = get_store()
    students = data.students.copy()

    # Scope filtering
    if user:
        role = user.get("role")
        if role == "HOD":
            students = students[students["Department"] == user.get("department")]
        elif role == "FACULTY":
            students = students[students["Section"].isin(user.get("assigned_sections", []))]
            if kind == "fees":
                abort(403)

    if search_q:
        q = search_q.lower()
        students = students[
            students["Enrollment_No"].str.lower().str.contains(q, na=False)
            | students["Name"].str.lower().str.contains(q, na=False)
        ]
    if section_filter:
        students = students[students["Section"].str.upper() == section_filter.upper()]

    ids = set(students["Enrollment_No"])
    frame = getattr(data, kind).copy()
    frame = frame[frame["Enrollment_No"].isin(ids)]

    if subject_filter and "Subject" in frame:
        frame = frame[frame["Subject"] == subject_filter]
    if status_filter and "Payment_Status" in frame:
        frame = frame[frame["Payment_Status"] == status_filter]
        if kind == "fees":
            students = students[students["Enrollment_No"].isin(frame["Enrollment_No"])]

    sections = []
    # Department-scoped grouping (D3)
    for section, sec_students in students.groupby("Section", dropna=False, sort=True):
        items = []
        for enr, srow in sec_students.sort_values("Enrollment_No").set_index("Enrollment_No").iterrows():
            recs = frame[frame["Enrollment_No"] == enr]
            s_dict = srow.to_dict()
            s_dict["Enrollment_No"] = enr
            item = {
                "student": s_dict,
                "section_letter": section or "Unknown",
                "semester": int(srow["Semester"]) if pd.notna(srow["Semester"]) else 5,
                "records": recs.to_dict("records"),
            }
            if kind == "attendance":
                item["average"] = round(float(recs["Attendance_Percentage"].mean()), 1) if not recs.empty else 0.0
            elif kind == "examination":
                item["average"] = round(float(recs["Total_Marks"].mean()), 1) if not recs.empty else 0.0
            else:
                item["fee"] = recs.iloc[0].to_dict() if not recs.empty else None
            items.append(item)

        if items:
            dept_code = items[0]["student"].get("Department", "ALL")
            sections.append({
                "letter": section or "Unknown",
                "dept_code": dept_code,
                "dept_name": f"{dept_code} — Section {section}",
                "students": items,
                "student_count": len(items),
            })

    dept_order = {d: i for i, d in enumerate(config.DEPARTMENTS)}
    def sec_key(s_dict):
        parts = str(s_dict.get("letter", "")).split("-")
        dept = parts[0] if parts else ""
        letter = parts[1] if len(parts) > 1 else ""
        return (dept_order.get(dept, 99), letter)

    sections.sort(key=sec_key)
    return students, frame, sections


def _paginate_sections(sections, page: int, page_size: int):
    all_items = [(section, item) for section in sections for item in section["students"]]
    total = len(all_items)
    start = (page - 1) * page_size
    selected = all_items[start:start + page_size]
    grouped: dict[str, dict] = {}
    for section, item in selected:
        grouped.setdefault(section["letter"], {**section, "students": []})
        grouped[section["letter"]]["students"].append(item)
    for section in grouped.values():
        section["student_count"] = len(section["students"])
    return list(grouped.values()), total


@app.route("/search")
@login_required
def search():
    enrollment_no = request.args.get("enrollment_no", "").strip()
    if not enrollment_no:
        flash("Please enter an Enrollment Number to search.", "warning")
        return redirect(url_for("search_student_page"))

    current_user = session.get("user")
    student = get_integrated_student(enrollment_no, user=current_user)
    if not student:
        flash(f"No authorized record found for Enrollment Number '{enrollment_no}'.", "danger")
        return redirect(url_for("search_student_page"))

    return redirect(url_for("student_profile", enrollment_no=student["personal"]["Enrollment_No"]))


@app.route("/search-student")
@login_required
def search_student_page():
    user = session.get("user")
    data = get_store()
    st = data.students.copy()
    if user:
        if user.get("role") == "HOD":
            st = st[st["Department"] == user.get("department")]
        elif user.get("role") == "FACULTY":
            st = st[st["Section"].isin(user.get("assigned_sections", []))]
    examples = st.head(4).to_dict("records")
    return render_template("search_student.html", example_students=examples, active_page="search_student")


@app.route("/student/")
@login_required
def student_profile_index():
    return render_template(
        "student_error.html",
        error_title="Enrollment number required",
        error_message="A student profile URL must include an Enrollment_No.",
        error_code=400,
        active_page="students",
    ), 400


@app.route("/student/<enrollment_no>")
@login_required
def student_profile(enrollment_no):
    current_user = session.get("user")
    student_data = get_integrated_student(enrollment_no, user=current_user)

    if not student_data:
        # Check if student exists in the database but outside user's role scope -> 403 Forbidden!
        store = get_store()
        key = enrollment_no.strip().upper()
        raw_match = store.students[store.students["Enrollment_No"].str.upper() == key]
        if not raw_match.empty:
            abort(403)
        return render_template(
            "student_error.html",
            error_title="Student not found",
            error_message=f"No student record exists for Enrollment_No '{enrollment_no}'.",
            error_code=404,
            active_page="students",
        ), 404

    return render_template("profile.html", data=student_data, active_page="students")


@app.route("/students")
@login_required
def students_page():
    user = session.get("user")
    kpis = get_kpis(user=user)

    dept_filter = request.args.get("department", "").strip()
    sec_filter = request.args.get("section", "").strip()
    sem_filter = request.args.get("semester", "").strip()
    search_q = request.args.get("q", "").strip()
    page = max(request.args.get("page", 1, type=int), 1)
    page_size = min(max(request.args.get("page_size", 50, type=int), 1), 200)

    # Input validation
    if dept_filter and not validate_alpha_only(dept_filter):
        dept_filter = ""
    if sem_filter and not validate_numeric_only(sem_filter):
        sem_filter = ""

    students = get_store().students.copy()

    # Scope filtering
    if user:
        if user.get("role") == "HOD":
            students = students[students["Department"] == user.get("department")]
        elif user.get("role") == "FACULTY":
            students = students[students["Section"].isin(user.get("assigned_sections", []))]

    if dept_filter:
        students = students[students["Department"] == dept_filter]
    if sec_filter:
        students = students[students["Section"] == sec_filter]
    if sem_filter:
        students = students[students["Semester"] == int(sem_filter)]
    if search_q:
        q = search_q.lower()
        students = students[
            students["Enrollment_No"].str.lower().str.contains(q, na=False)
            | students["Name"].str.lower().str.contains(q, na=False)
        ]

    students = students.sort_values("Enrollment_No")
    total_students = len(students)
    student_list = students.iloc[(page - 1) * page_size:page * page_size].to_dict("records")

    return render_template(
        "students.html",
        students=student_list,
        kpis=kpis,
        selected_dept=dept_filter,
        selected_sec=sec_filter,
        selected_sem=sem_filter,
        search_query=search_q,
        page=page,
        page_size=page_size,
        total_students=total_students,
        active_page="students",
    )


# ==========================================
# ACADEMIC MODULES: ATTENDANCE, EXAMS, FEES
# ==========================================
@app.route("/attendance")
@login_required
def attendance_page():
    user = session.get("user")
    kpis = get_kpis(user=user)
    sec_filter = request.args.get("section", "").strip().upper()
    subj_filter = request.args.get("subject", "").strip()
    search_q = request.args.get("q", "").strip()

    students_list, flat_records, sections_data = _csv_module_data(
        "attendance", sec_filter, subj_filter, search_q=search_q, user=user
    )
    page = max(request.args.get("page", 1, type=int), 1)
    page_size = min(max(request.args.get("page_size", 10, type=int), 1), 100)
    sections_data, total_students = _paginate_sections(sections_data, page, page_size)
    values = flat_records["Attendance_Percentage"] if not flat_records.empty else pd.Series(dtype=float)

    # FIX: low_attendance_count must count UNIQUE STUDENTS whose per-student
    # average attendance is below threshold — NOT individual subject rows.
    if not flat_records.empty:
        per_student_avg = flat_records.groupby("Enrollment_No")["Attendance_Percentage"].mean()
        low_attendance_count = int((per_student_avg < config.ATTENDANCE_WARNING_THRESHOLD).sum())
    else:
        low_attendance_count = 0

    return render_template(
        "attendance.html",
        sections_data=sections_data,
        summary_stats={
            "overall_avg": round(float(values.mean()), 1) if not values.empty else 0.0,
            "low_attendance_count": low_attendance_count,
            "subjects_count": int(flat_records["Subject"].nunique()) if not flat_records.empty else 0,
            "total_students": total_students,
        },
        records=flat_records.to_dict("records"),
        kpis=kpis,
        selected_sec=sec_filter,
        selected_subj=subj_filter,
        search_query=search_q,
        page=page,
        page_size=page_size,
        total_students=total_students,
        active_page="attendance",
    )


@app.route("/examination")
@login_required
def examination_page():
    user = session.get("user")
    kpis = get_kpis(user=user)
    sec_filter = request.args.get("section", "").strip().upper()
    subj_filter = request.args.get("subject", "").strip()
    search_q = request.args.get("q", "").strip()

    students_list, flat_records, sections_data = _csv_module_data(
        "examination", sec_filter, subj_filter, search_q=search_q, user=user
    )
    page = max(request.args.get("page", 1, type=int), 1)
    page_size = min(max(request.args.get("page_size", 10, type=int), 1), 100)
    sections_data, total_students = _paginate_sections(sections_data, page, page_size)
    values = flat_records["Total_Marks"] if not flat_records.empty else pd.Series(dtype=float)

    return render_template(
        "examination.html",
        sections_data=sections_data,
        summary_stats={
            "overall_avg": round(float(values.mean()), 1) if not values.empty else 0.0,
            "highest_marks": int(values.max()) if not values.empty else 0,
            "subjects_count": int(flat_records["Subject"].nunique()) if not flat_records.empty else 0,
            "total_students": total_students,
        },
        records=flat_records.to_dict("records"),
        kpis=kpis,
        selected_sec=sec_filter,
        selected_subj=subj_filter,
        search_query=search_q,
        page=page,
        page_size=page_size,
        total_students=total_students,
        active_page="examination",
    )


@app.route("/fees")
@login_required
@roles_required("ADMIN", "HOD")
def fees_page():
    user = session.get("user")
    kpis = get_kpis(user=user)
    sec_filter = request.args.get("section", "").strip().upper()
    status_filter = request.args.get("status", "").strip()
    search_q = request.args.get("q", "").strip()

    students_list, flat_records, sections_data = _csv_module_data(
        "fees", section_filter=sec_filter, status_filter=status_filter, search_q=search_q, user=user
    )
    page = max(request.args.get("page", 1, type=int), 1)
    page_size = min(max(request.args.get("page_size", 10, type=int), 1), 100)
    sections_data, total_students = _paginate_sections(sections_data, page, page_size)

    return render_template(
        "fees.html",
        sections_data=sections_data,
        summary_stats={
            "total_fees": f"{int(flat_records['Total_Fees'].sum()):,}" if not flat_records.empty else "0",
            "total_collected": f"{int(flat_records['Paid_Amount'].sum()):,}" if not flat_records.empty else "0",
            "total_pending": f"{int(flat_records['Pending_Amount'].sum()):,}" if not flat_records.empty else "0",
            "pending_students": int((flat_records["Pending_Amount"] > 0).sum()) if not flat_records.empty else 0,
            "total_students": total_students,
        },
        records=flat_records.to_dict("records"),
        kpis=kpis,
        selected_sec=sec_filter,
        selected_status=status_filter,
        search_query=search_q,
        page=page,
        page_size=page_size,
        total_students=total_students,
        active_page="fees",
    )


# ==========================================
# INSTITUTIONAL MODULES: FACULTY, NOTICES, LIBRARY, ADMISSIONS
# ==========================================
@app.route("/faculty")
@login_required
def faculty_page():
    user = session.get("user")
    data = get_store()
    fac_df = data.faculty.copy() if data.faculty is not None else pd.DataFrame()

    q = request.args.get("q", "").strip()
    dept = request.args.get("department", "").strip()

    if dept and not validate_alpha_only(dept):
        dept = ""

    # Scope filtering
    if user and user.get("role") == "HOD":
        fac_df = fac_df[fac_df["Department"] == user.get("department")]
    elif dept:
        fac_df = fac_df[fac_df["Department"] == dept]

    if q and not fac_df.empty:
        q_low = q.lower()
        fac_df = fac_df[
            fac_df["Name"].str.lower().str.contains(q_low, na=False)
            | fac_df["Email"].str.lower().str.contains(q_low, na=False)
            | fac_df["Faculty_ID"].str.lower().str.contains(q_low, na=False)
        ]

    faculty_list = fac_df.to_dict("records") if not fac_df.empty else []
    return render_template(
        "faculty.html",
        faculty_list=faculty_list,
        departments=config.DEPARTMENTS,
        selected_dept=dept,
        query=q,
        active_page="faculty",
    )


@app.route("/notices")
@login_required
def notices_page():
    user = session.get("user")
    data = get_store()
    notices_df = data.notices.copy() if data.notices is not None else pd.DataFrame()

    q = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()
    priority = request.args.get("priority", "").strip()

    # Scope: Faculty & HOD see notices for ALL and their dept
    if user and user.get("role") in {"HOD", "FACULTY"}:
        u_dept = user.get("department")
        notices_df = notices_df[notices_df["Department"].isin(["ALL", u_dept])]

    if category and not notices_df.empty:
        notices_df = notices_df[notices_df["Category"] == category]
    if priority and not notices_df.empty:
        notices_df = notices_df[notices_df["Priority"] == priority]
    if q and not notices_df.empty:
        q_low = q.lower()
        notices_df = notices_df[
            notices_df["Title"].str.lower().str.contains(q_low, na=False)
            | notices_df["Content"].str.lower().str.contains(q_low, na=False)
        ]

    urgent_notices = []
    if not notices_df.empty:
        urgent_df = notices_df[(notices_df["Priority"] == "Urgent") & (notices_df["Is_Active"] == True)]
        urgent_notices = urgent_df.head(5).to_dict("records")

    items = notices_df.to_dict("records") if not notices_df.empty else []
    return render_template(
        "notices.html",
        notices_list=items,
        urgent_notices=urgent_notices,
        selected_cat=category,
        selected_priority=priority,
        query=q,
        active_page="notices",
    )


@app.route("/notices/post", methods=["POST"])
@login_required
def post_notice():
    validate_csrf()
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()
    category = request.form.get("category", "Academic").strip()
    priority = request.form.get("priority", "Normal").strip()
    user = session.get("user", {})

    if not title or not content:
        flash("Circular title and content are required.", "danger")
        return redirect(url_for("notices_page"))

    new_row = {
        "Notice_ID": f"NOT{pd.Timestamp.now().strftime('%m%d%H%M')}",
        "Title": title,
        "Content": content,
        "Category": category,
        "Department": user.get("department", "ALL"),
        "Target_Section": "ALL",
        "Priority": priority,
        "Published_By": f"{user.get('name', 'Admin')} ({user.get('role')})",
        "Published_Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "Is_Active": True,
    }

    data = get_store()
    if data.notices is not None:
        data.notices = pd.concat([pd.DataFrame([new_row]), data.notices], ignore_index=True)

    flash("Circular notice posted successfully.", "success")
    return redirect(url_for("notices_page"))


@app.route("/library")
@login_required
def library_page():
    user = session.get("user")
    data = get_store()
    lib_df = data.library.copy() if data.library is not None else pd.DataFrame()

    q = request.args.get("q", "").strip()
    status = request.args.get("status", "").strip()

    # Scope filtering
    if user and user.get("role") == "HOD":
        dept_students = set(data.students[data.students["Department"] == user.get("department")]["Enrollment_No"])
        lib_df = lib_df[lib_df["Enrollment_No"].isin(dept_students)]
    elif user and user.get("role") == "FACULTY":
        sec_students = set(data.students[data.students["Section"].isin(user.get("assigned_sections", []))]["Enrollment_No"])
        lib_df = lib_df[lib_df["Enrollment_No"].isin(sec_students)]

    if status and not lib_df.empty:
        lib_df = lib_df[lib_df["Status"] == status]
    if q and not lib_df.empty:
        q_low = q.lower()
        lib_df = lib_df[
            lib_df["Enrollment_No"].str.lower().str.contains(q_low, na=False)
            | lib_df["Book_Title"].str.lower().str.contains(q_low, na=False)
        ]

    total_tx = len(lib_df)
    overdue_count = int((lib_df["Status"] == "Overdue").sum()) if not lib_df.empty else 0
    overdue_rate = round((overdue_count / total_tx) * 100.0, 1) if total_tx > 0 else 0.0
    items = lib_df.head(100).to_dict("records") if not lib_df.empty else []

    return render_template(
        "library.html",
        items=items,
        total_tx=total_tx,
        overdue_rate=overdue_rate,
        selected_status=status,
        query=q,
        active_page="library",
    )


@app.route("/admissions")
@login_required
@roles_required("ADMIN", "HOD")
def admissions_page():
    user = session.get("user")
    data = get_store()
    adm_df = data.admissions.copy() if data.admissions is not None else pd.DataFrame()

    if user and user.get("role") == "HOD":
        dept_enrs = set(data.students[data.students["Department"] == user.get("department")]["Enrollment_No"])
        adm_df = adm_df[adm_df["Enrollment_No"].isin(dept_enrs)]

    items = adm_df.head(150).to_dict("records") if not adm_df.empty else []
    return render_template("admissions.html", admissions_list=items, active_page="admissions")


# ==========================================
# PHASE 6: MACHINE LEARNING & RISK DECISION SUPPORT
# ==========================================
@app.route("/risk")
@login_required
def risk_page():
    user = session.get("user")
    data = get_store()
    risk_service = RiskPredictorService.get_instance()
    pred_df = risk_service.predictions_df.copy() if risk_service.predictions_df is not None else pd.DataFrame()

    # Scope filtering
    if user and user.get("role") == "HOD":
        pred_df = pred_df[pred_df["Department"] == user.get("department")]
    elif user and user.get("role") == "FACULTY":
        sec_enrs = set(data.students[data.students["Section"].isin(user.get("assigned_sections", []))]["Enrollment_No"])
        pred_df = pred_df[pred_df["Enrollment_No"].isin(sec_enrs)]

    q = request.args.get("q", "").strip()
    dept = request.args.get("department", "").strip()
    band = request.args.get("band", "").strip()

    if dept and not pred_df.empty:
        pred_df = pred_df[pred_df["Department"] == dept]
    if band and not pred_df.empty:
        pred_df = pred_df[pred_df["Risk_Band"] == band]
    if q and not pred_df.empty:
        q_low = q.lower()
        pred_df = pred_df[
            pred_df["Enrollment_No"].str.lower().str.contains(q_low, na=False)
            | pred_df["Name"].str.lower().str.contains(q_low, na=False)
        ]

    # Department counts
    dept_risk_counts = pred_df[pred_df["Predicted_At_Risk"] == 1]["Department"].value_counts().to_dict() if not pred_df.empty else {}
    for d in config.DEPARTMENTS:
        dept_risk_counts.setdefault(d, 0)

    risk_summary = {
        "high_count": int((pred_df["Risk_Band"] == "High").sum()) if not pred_df.empty else 0,
        "med_count": int((pred_df["Risk_Band"] == "Medium").sum()) if not pred_df.empty else 0,
        "low_count": int((pred_df["Risk_Band"] == "Low").sum()) if not pred_df.empty else 0,
        "predicted_at_risk": int(pred_df["Predicted_At_Risk"].sum()) if not pred_df.empty else 0,
    }

    import ast
    items = []
    if not pred_df.empty:
        for _, row in pred_df.sort_values("Risk_Probability", ascending=False).iterrows():
            r_dict = row.to_dict()
            for col in ("Top_Reasons", "Interventions"):
                if isinstance(r_dict.get(col), str) and r_dict[col].startswith("["):
                    try:
                        r_dict[col] = ast.literal_eval(r_dict[col])
                    except Exception:
                        pass
            items.append(r_dict)

    return render_template(
        "risk.html",
        items=items,
        risk_summary=risk_summary,
        dept_risk_counts=dept_risk_counts,
        departments=config.DEPARTMENTS,
        selected_dept=dept,
        selected_band=band,
        query=q,
        active_page="risk",
    )


@app.route("/ml/performance")
@login_required
@roles_required("ADMIN", "HOD")
def ml_performance_page():
    metrics_path = config.BASE_DIR / "models" / "metrics_summary.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    risk_service = RiskPredictorService.get_instance()
    anomalies_list = []
    if risk_service.predictions_df is not None and "Is_Anomaly" in risk_service.predictions_df.columns:
        anom_df = risk_service.predictions_df[risk_service.predictions_df["Is_Anomaly"] == 1]
        anomalies_list = anom_df.to_dict("records")

    return render_template("ml_performance.html", metrics=metrics, anomalies_list=anomalies_list, active_page="ml_performance")


@app.route("/api/ml/what-if", methods=["POST"])
@login_required
def api_what_if():
    validate_csrf()
    body = request.get_json(silent=True) or {}
    att_delta = float(body.get("attendance_delta", 0.0))
    int_delta = float(body.get("internal_delta", 0.0))

    base_features = {
        # Baseline: STU0604 (Praveen Khan) — a genuine High Risk student (97.8% risk)
        # Drag sliders UP to see risk drop from High → Medium → Low
        "overall_attendance_pct": 62.5,
        "subjects_below_75": 5,
        "min_subject_attendance": 53.33,
        "internal_marks_avg_pct": 55.0,
        "internal_marks_min_pct": 42.5,
        "prior_sgpa_mean": 5.36,
        "prior_sgpa_last": 5.29,
        "prior_sgpa_trend": 0.14,
        "prior_backlogs": 1,
        "prior_sem_attendance": 61.64,
        "fee_pending_ratio": 0.0,
        "fee_overdue_flag": 0,
        "library_issues_count": 9,
        "credits_registered": 20,
        "dept_CSE": 0,
        "dept_ECE": 0,
        "dept_EEE": 0,
        "dept_IT": 1,
        "dept_AIDS": 0,
        "dept_AIML": 0,
        "original_prob": 0.9775,
    }

    service = RiskPredictorService.get_instance()
    result = service.simulate_what_if(base_features, att_delta, int_delta)
    return jsonify(result)


# ==========================================
# REPORTS & CSV EXPORT
# ==========================================
@app.route("/reports")
@login_required
def reports_page():
    user = session.get("user")
    kpis = get_kpis(user=user)
    data = get_store()
    students = data.students.copy()
    attendance = data.attendance.copy()
    examination = data.examination.copy()
    fees = data.fees.copy()

    if user and user.get("role") == "HOD":
        dept = user.get("department")
        students = students[students["Department"] == dept]
        s_ids = set(students["Enrollment_No"])
        attendance = attendance[attendance["Enrollment_No"].isin(s_ids)]
        examination = examination[examination["Enrollment_No"].isin(s_ids)]
        fees = fees[fees["Enrollment_No"].isin(s_ids)]
    elif user and user.get("role") == "FACULTY":
        students = students[students["Section"].isin(user.get("assigned_sections", []))]
        s_ids = set(students["Enrollment_No"])
        attendance = attendance[attendance["Enrollment_No"].isin(s_ids)]
        examination = examination[examination["Enrollment_No"].isin(s_ids)]
        fees = fees.iloc[0:0]

    dept_stats = []
    dept_order = {d: i for i, d in enumerate(config.DEPARTMENTS)}
    for dept in sorted(students["Department"].unique(), key=lambda d: dept_order.get(d, 99)):
        dept_st = students[students["Department"] == dept]
        d_ids = set(dept_st["Enrollment_No"])
        d_att = attendance[attendance["Enrollment_No"].isin(d_ids)]
        d_fees = fees[fees["Enrollment_No"].isin(d_ids)]
        dept_stats.append({
            "Department": dept,
            "Students": len(dept_st),
            "Average_Attendance": round(float(d_att["Attendance_Percentage"].mean()), 1) if not d_att.empty else 0.0,
            "Average_CGPA": round(float(dept_st["CGPA"].mean()), 2) if not dept_st.empty else 0.0,
            "Total_Fees": int(d_fees["Total_Fees"].sum()),
            "Paid_Amount": int(d_fees["Paid_Amount"].sum()),
            "Pending_Amount": int(d_fees["Pending_Amount"].sum()),
        })

    exam_vals = examination["Total_Marks"] if not examination.empty else pd.Series(dtype=float)
    exam_summary = {
        "avg_marks": round(float(exam_vals.mean()), 1) if not exam_vals.empty else 0.0,
        "highest_marks": int(exam_vals.max()) if not exam_vals.empty else 0,
        "lowest_marks": int(exam_vals.min()) if not exam_vals.empty else 0,
        "students_appeared": len(examination["Enrollment_No"].unique()) if not examination.empty else 0,
    }

    return render_template(
        "reports.html",
        kpis=kpis,
        dept_stats=dept_stats,
        exam_summary=exam_summary,
        active_page="reports",
    )


@app.route("/reports/export")
@login_required
def reports_export():
    report = request.args.get("report", "").strip().lower()
    user = session.get("user", {})
    role = user.get("role")

    # Faculty is strictly forbidden from accessing or exporting fees
    if report == "fees" and role == "FACULTY":
        abort(403)

    if DATA_MODE != "csv":
        return jsonify({"error": "Report export is supported in CSV warehouse mode"}), 501

    data = get_store()
    if report in {"students", "attendance", "examination", "fees"}:
        frame = getattr(data, report).copy()
        # Scope export by user department
        if role == "HOD":
            dept_students = set(data.students[data.students["Department"] == user.get("department")]["Enrollment_No"])
            frame = frame[frame["Enrollment_No"].isin(dept_students)]
        elif role == "FACULTY":
            sec_students = set(data.students[data.students["Section"].isin(user.get("assigned_sections", []))]["Enrollment_No"])
            frame = frame[frame["Enrollment_No"].isin(sec_students)]
        filename = f"ifhe_{report}_semester5.csv"

    elif report == "departments":
        students = data.students
        if role == "HOD":
            students = students[students["Department"] == user.get("department")]
        attendance = data.attendance.merge(students[["Enrollment_No", "Department"]], on="Enrollment_No", how="inner")
        rows = []
        for department, group in students.groupby("Department", sort=True):
            ids = set(group["Enrollment_No"])
            dept_att = attendance[attendance["Enrollment_No"].isin(ids)]
            dept_fees = data.fees[data.fees["Enrollment_No"].isin(ids)]
            rows.append({
                "Department": department,
                "Students": len(group),
                "Average_Attendance": round(float(dept_att["Attendance_Percentage"].mean()), 1),
                "Average_CGPA": round(float(group["CGPA"].mean()), 2),
                "Total_Fees": int(dept_fees["Total_Fees"].sum()),
                "Paid_Amount": int(dept_fees["Paid_Amount"].sum()),
                "Pending_Amount": int(dept_fees["Pending_Amount"].sum()),
            })
        frame = pd.DataFrame(rows)
        filename = "ifhe_department_summary_semester5.csv"

    else:
        return jsonify({
            "error": "Unsupported report",
            "available_reports": ["students", "attendance", "examination", "fees", "departments"],
        }), 400

    payload = frame.to_csv(index=False).encode("utf-8")
    return send_file(
        BytesIO(payload),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/settings")
@login_required
def settings_page():
    return render_template("settings.html", user=session.get("user"), active_page="settings")


# ==========================================
# REST API ENDPOINTS
# ==========================================
@app.route("/api/student/<enrollment_no>")
@login_required
def api_student(enrollment_no):
    user = session.get("user")
    student_data = get_integrated_student(enrollment_no, user=user)
    if not student_data:
        store = get_store()
        key = enrollment_no.strip().upper()
        raw = store.students[store.students["Enrollment_No"].str.upper() == key]
        if not raw.empty:
            abort(403)
        return jsonify({"error": "Student not found", "enrollment_no": enrollment_no}), 404
    return jsonify(student_data)


@app.route("/api/kpis")
@login_required
def api_kpis():
    user = session.get("user")
    return jsonify(get_kpis(user=user))


@app.route("/api/students")
@login_required
def api_students():
    data = get_store() if DATA_MODE == "csv" else None
    if data is None:
        return jsonify({"error": "Student collection API is available in CSV mode"}), 501

    user = session.get("user")
    students = data.students.copy()

    # Scope enforcement
    if user:
        role = user.get("role")
        if role == "HOD":
            students = students[students["Department"] == user.get("department")]
        elif role == "FACULTY":
            students = students[students["Section"].isin(user.get("assigned_sections", []))]

    query = request.args.get("q", "").strip().lower()
    dept = request.args.get("department", "").strip()
    sec = request.args.get("section", "").strip()
    sem = request.args.get("semester", type=int)
    page = max(request.args.get("page", 1, type=int), 1)
    page_size = min(max(request.args.get("page_size", 50, type=int), 1), 200)

    if query:
        students = students[
            students["Enrollment_No"].str.lower().str.contains(query, na=False)
            | students["Name"].str.lower().str.contains(query, na=False)
        ]
    if dept:
        students = students[students["Department"] == dept]
    if sec:
        students = students[students["Section"] == sec]
    if sem is not None:
        students = students[students["Semester"] == sem]

    students = students.sort_values("Enrollment_No")
    total = len(students)
    rows = students.iloc[(page - 1) * page_size:page * page_size].to_dict("records")
    return jsonify({"items": rows, "page": page, "page_size": page_size, "total": total})


@app.route("/api/data-quality")
@login_required
def api_data_quality():
    return jsonify(get_store().quality.as_dict() if DATA_MODE == "csv" else {
        "mode": "mysql",
        "message": "Quality summary is supplied by MySQL warehouse",
    })


# ==========================================
# ADMIN — ADD STUDENT
# ==========================================
import csv as _csv
from datetime import date as _date, datetime as _datetime

DEPARTMENTS = ["AIDS", "AIML", "CSE", "ECE", "EEE", "IT"]
SECTION_LETTERS = ["A", "B", "C", "D"]


@app.route("/students/add", methods=["GET", "POST"])
@login_required
@roles_required("ADMIN")
def add_student_page():
    """ADMIN-only page: Add a new student record to the system."""
    user = session.get("user")
    kpis = get_kpis(user=user)
    errors: list[str] = []

    if request.method == "POST":
        validate_csrf()

        # --- Collect fields ---
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "").strip().upper()
        section_letter = request.form.get("section_letter", "").strip().upper()
        gender = request.form.get("gender", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip().lower()
        dob_str = request.form.get("date_of_birth", "").strip()
        address = request.form.get("address", "").strip()

        # --- Validate ---
        if not name:
            errors.append("Name is required.")
        elif not validate_alpha_only(name):
            errors.append("Name must contain only letters, spaces, hyphens, or periods.")

        if department not in DEPARTMENTS:
            errors.append(f"Department must be one of: {', '.join(DEPARTMENTS)}.")

        if section_letter not in SECTION_LETTERS:
            errors.append(f"Section must be A, B, C, or D.")

        if gender not in ("Male", "Female", "Other"):
            errors.append("Gender must be Male, Female, or Other.")

        if not phone:
            errors.append("Phone number is required.")
        elif not validate_numeric_only(phone) or len(phone) != 10:
            errors.append("Phone must be exactly 10 digits.")

        if not email or "@" not in email:
            errors.append("A valid email address is required.")

        if not dob_str:
            errors.append("Date of Birth is required.")
        else:
            try:
                dob = _datetime.strptime(dob_str, "%Y-%m-%d").date()
                admission_year = dob.year + 17
                if dob.year < 2000 or dob.year > 2008:
                    errors.append("Date of Birth must yield an admission age of 17–20 (DOB year 2000–2008).")
            except ValueError:
                errors.append("Date of Birth must be in YYYY-MM-DD format.")

        if not address:
            errors.append("Address is required.")

        if not errors:
            # --- Duplicate prevention: same phone OR same email ---
            data = get_store()
            dup_phone = data.students[data.students["Phone_No"].astype(str).str.strip() == phone]
            dup_email = data.students[data.students["Email"].astype(str).str.strip().str.lower() == email.lower()]
            if not dup_phone.empty:
                errors.append(
                    f"A student with phone {phone} already exists "
                    f"({dup_phone.iloc[0]['Enrollment_No']} – {dup_phone.iloc[0]['Name']}). "
                    "Cannot add duplicate student."
                )
            if not dup_email.empty:
                errors.append(
                    f"A student with email {email} already exists "
                    f"({dup_email.iloc[0]['Enrollment_No']} – {dup_email.iloc[0]['Name']}). "
                    "Cannot add duplicate student."
                )

        if not errors:
            # --- Generate new Enrollment_No ---
            data = get_store()
            existing_ids = data.students["Enrollment_No"].str.extract(r"STU(\d+)")[0].dropna().astype(int)
            next_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1001
            enrollment_no = f"STU{next_id:04d}"

            section = f"{department}-{section_letter}"
            new_row = {
                "Enrollment_No": enrollment_no,
                "Name": name,
                "Phone_No": phone,
                "Email": email,
                "Gender": gender,
                "Department": department,
                "Section": section,
                "Semester": config.CURRENT_SEMESTER,
                "Address": address,
                "Academic_Year": "2025-2026",
                "CGPA": 0.0,
                "Date_of_Birth": dob_str,
                "Batch": "2023-27",
                "Admission_Year": admission_year,
            }

            # --- Append to both CSV locations ---
            csv_paths = [
                config.BASE_DIR / "erp_semester5_etl_package" / "students (2).csv",
                config.BASE_DIR / "data" / "students.csv",
            ]
            for path in csv_paths:
                if path.exists():
                    with open(path, "a", newline="", encoding="utf-8") as f:
                        writer = _csv.DictWriter(f, fieldnames=list(new_row.keys()))
                        writer.writerow(new_row)

            # Invalidate store cache so new student appears immediately
            import data_store as _ds
            _ds._STORE = None

            flash(f"Student {name} ({enrollment_no}) added successfully!", "success")
            return redirect(url_for("students_page"))

    return render_template(
        "add_student.html",
        errors=errors,
        departments=DEPARTMENTS,
        sections=SECTION_LETTERS,
        kpis=kpis,
        active_page="add_student",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="127.0.0.1", port=port)
