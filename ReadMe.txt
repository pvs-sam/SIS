================================================================================
  ERP DATA WAREHOUSE & INSTITUTIONAL DECISION SUPPORT SYSTEM
  ICFAI Foundation for Higher Education (IFHE), Hyderabad
================================================================================

Project Title  : ERP Data Warehouse Design with ML-Based Student Risk Analytics
Institution    : ICFAI Foundation for Higher Education (IFHE), Hyderabad
Batch          : 2023-2027  |  Semester 5  |  Academic Year 2025-2026
Dataset Size   : 1,000 Students  |  6 Departments  |  5 Semesters of History
Tech Stack     : Python 3.11 + Flask + Pandas + scikit-learn + Bootstrap 5


================================================================================
TABLE OF CONTENTS
================================================================================

  1.  Project Overview
  2.  Key Features
  3.  Tech Stack & Architecture
  4.  Project Structure
  5.  Dataset Description
  6.  Installation & Setup
  7.  How to Run
  8.  Login Credentials
  9.  Module Descriptions
  10. Machine Learning (Phase 6)
  11. Screenshots & Demo Flow
  12. API Endpoints
  13. Testing
  14. Known Limitations
  15. Future Enhancements


================================================================================
1. PROJECT OVERVIEW
================================================================================

This project implements a full-stack ERP Data Warehouse and Decision Support
System for an engineering institution. It integrates data across 10 institutional
domains (Students, Attendance, Examination, Fees, Faculty, Library, Notices,
Admissions, Terms, and ML Predictions) into a unified analytics platform.

The system features:
- Multi-module institutional ERP with 16+ pages
- Role-Based Access Control (RBAC) for Admin, HOD, and Faculty
- Phase 6 Machine Learning layer for student risk prediction
- Real-time What-If Simulator for intervention planning
- Student Segmentation, Anomaly Detection, and SGPA Prediction

The ML models are trained on synthetic but statistically realistic data,
demonstrating how Random Forest classifiers can identify at-risk students
at mid-semester — before final exams — enabling timely academic intervention.


================================================================================
2. KEY FEATURES
================================================================================

CORE ERP MODULES:
  [x] Dashboard with KPI cards and department performance overview
  [x] Student Management (All Students, Search, Add New Student)
  [x] Attendance module with section-wise breakdown and low-attendance alerts
  [x] Examination module with grade distribution and subject analytics
  [x] Fees & Payments with collection tracking and outstanding dues
  [x] Faculty Directory with designation distribution and workload data
  [x] Library Circulation tracking with overdue detection
  [x] Circulars & Notices with priority-based display
  [x] Admissions Intake with admission statistics
  [x] Institutional Reports with department-wise summary and CSV export

SECURITY:
  [x] Role-Based Access Control (ADMIN / HOD / FACULTY)
  [x] Server-side scope verification — URL tampering blocked
  [x] CSRF token protection on all POST forms
  [x] Login rate limiting (5 attempts per 15 minutes)
  [x] Session-based authentication with HTTPOnly cookies

MACHINE LEARNING (Phase 6):
  [x] At-Risk Student Prediction using Random Forest (300 trees)
  [x] Model comparison: RF vs Logistic Regression vs Decision Tree vs GB
  [x] SGPA Predictor using RF Regression
  [x] Student Segmentation using KMeans (4 personas)
  [x] Anomaly Detection using Isolation Forest
  [x] What-If Simulator for real-time intervention modelling
  [x] Per-student reason codes and risk bands
  [x] Leakage-safe feature engineering with test assertions

ADMIN FEATURES:
  [x] Add New Student form with full validation
  [x] Institutional Reports with CSV export
  [x] Settings page with system configuration view


================================================================================
3. TECH STACK & ARCHITECTURE
================================================================================

BACKEND:
  - Python 3.11           : Core language for backend and ML
  - Flask 3.x             : Lightweight WSGI web framework
  - Pandas 2.x            : DataFrame-based data wrangling and aggregation
  - NumPy                  : Numerical computing for ML pipelines
  - scikit-learn 1.3+     : Machine learning models and evaluation
  - joblib                 : Model serialisation/persistence
  - Werkzeug               : Password hashing (PBKDF2-SHA256)

FRONTEND:
  - Bootstrap 5.3.3       : Responsive CSS framework
  - Bootstrap Icons 1.11  : Icon library
  - Jinja2                 : Server-side HTML templating
  - Vanilla JavaScript    : Interactive form controls and sliders

DATA STORAGE:
  - CSV files (demo mode)        : Portable, no database installation needed
  - MySQL (supported)            : Production-ready database support
  NOTE: The system supports both CSV and MySQL as data backends.
        For this submission, CSV is used for portability — the project
        runs out of the box without any database installation.

ARCHITECTURE DIAGRAM:

  CSV Files (data/*.csv)
       |
       v
  data_store.py   ---->  Schema validation & referential integrity
       |
       v
  app.py          ---->  Flask routes, RBAC, KPI computation
       |
       v
  templates/*.html ---->  Jinja2 server-side rendering
       |
       v
  Browser (Bootstrap 5 UI)

  ml/features.py  ---->  Leakage-safe feature extraction
       |
       v
  ml/train.py     ---->  Model training (RF, LR, DT, GB, KMeans, IF)
       |
       v
  models/*.joblib  ---->  Persisted models loaded by ml/predict.py


================================================================================
4. PROJECT STRUCTURE
================================================================================

erp-datawarehouse-design/
|
|-- app.py                      # Main Flask application (all routes)
|-- auth.py                     # Authentication, RBAC decorators, CSRF
|-- config.py                   # Central configuration (thresholds, settings)
|-- data_store.py               # CSV data loader with validation
|-- db_config.py                # MySQL database configuration
|-- database_setup.py           # MySQL schema creation script
|-- requirements.txt            # Python dependencies
|-- README.txt                  # This file
|
|-- data/                       # Primary data directory
|   |-- students.csv            # 1,000 student records
|   |-- attendance.csv          # 5,000 attendance records (5 subjects/student)
|   |-- examination.csv         # 5,000 exam records
|   |-- fees.csv                # 1,000 fee records
|   |-- faculty.csv             # ~58 faculty records
|   |-- library_transactions.csv # Library borrowing records
|   |-- notices.csv             # ~40 institutional notices
|   |-- admissions.csv          # Admission records
|   |-- student_terms.csv       # Historical term-level data (Sem 1-5)
|   |-- attendance_history.csv  # Multi-semester attendance history
|   |-- examination_history.csv # Multi-semester exam history
|   `-- catalog/                # Subject catalog
|
|-- erp_semester5_etl_package/  # ETL-validated data (mirrors data/)
|   |-- students (2).csv       # Primary student data source
|   |-- attendance.csv
|   |-- examination.csv
|   |-- fees.csv
|   |-- student_terms.csv
|   |-- schema.sql              # Database schema definition
|   `-- etl_quality_report.json # ETL validation results
|
|-- ml/                         # Machine Learning module
|   |-- features.py             # Feature extraction & FEATURE_COLUMNS list
|   |-- train.py                # Model training pipeline
|   `-- predict.py              # Runtime inference service
|
|-- models/                     # Trained model artefacts
|   |-- risk_model_rf.joblib    # Random Forest classifier (300 trees)
|   |-- scaler.joblib           # StandardScaler for Logistic Regression
|   |-- sgpa_predictor.joblib   # RF Regressor for SGPA prediction
|   |-- kmeans_personas.joblib  # KMeans clustering model (4 personas)
|   |-- isolation_forest.joblib # Isolation Forest anomaly detector
|   |-- ml_predictions.csv      # Full-cohort predictions (1,000 rows)
|   |-- metrics_summary.json    # All model metrics and comparisons
|   |-- model_card.md           # Model card (data, features, limits)
|   `-- decision_tree_rules.txt # Human-readable DT rules
|
|-- templates/                  # Jinja2 HTML templates
|   |-- base.html               # Layout with sidebar navigation
|   |-- login.html              # Authentication page
|   |-- dashboard.html          # Main dashboard with KPIs
|   |-- students.html           # All Students listing
|   |-- search_student.html     # Student search
|   |-- add_student.html        # Admin: Add new student form
|   |-- profile.html            # Individual student profile + AI risk
|   |-- attendance.html         # Attendance module
|   |-- examination.html        # Examination module
|   |-- fees.html               # Fees & payments module
|   |-- faculty.html            # Faculty directory
|   |-- library.html            # Library circulation
|   |-- notices.html            # Circulars & notices
|   |-- admissions.html         # Admissions intake
|   |-- risk.html               # At-Risk Radar (ML predictions)
|   |-- ml_performance.html     # ML Diagnostics, What-If, anomalies
|   |-- reports.html            # Institutional reports
|   `-- settings.html           # System settings
|
|-- static/
|   |-- css/style.css           # Custom stylesheet
|   |-- js/                     # Client-side JavaScript
|   `-- images/                 # Static images
|
|-- scripts/
|   |-- generate_data.py        # Synthetic data generator (seeded)
|   |-- add_record.py           # CLI utility to add student records
|   `-- uniformity_check.py     # Threshold consistency checker
|
|-- tests/
|   |-- test_ml_no_leakage.py   # ML leakage safety assertions
|   `-- test_rbac_matrix.py     # RBAC permission tests
|
`-- test_*.py                   # Integration tests (attendance, fees, etc.)


================================================================================
5. DATASET DESCRIPTION
================================================================================

All data is SYNTHETIC — generated with fixed random seeds for reproducibility.
Distributions are statistically realistic to mimic a real institution.

DATASET LOCATION:
  - Local path   : data/              (included in project submission)
  - ETL package  : erp_semester5_etl_package/
  - Dataset link : https://github.com/pvssamhita/erp-datawarehouse-design/tree/main/data

STUDENTS (1,000 records):
  - 6 Departments: CSE (167), ECE (167), EEE (167), IT (167), AIDS (166), AIML (166)
  - 4 Sections per department (A-D), ~41-42 students each
  - Batch 2023-27, currently in Semester 5, AY 2025-26
  - Date of Birth between 2003-08-01 and 2006-12-31
  - Fields: Enrollment_No, Name, Phone, Email, Gender, Department, Section,
            Semester, Address, Academic_Year, CGPA, Date_of_Birth, Batch

ATTENDANCE (5,000 records — 5 subjects per student):
  - Classes held: 36-48 per subject
  - Distribution: Beta-shaped, mean ~82.1%, ~155 students below 75%
  - Fields: Enrollment_No, Subject_Code, Semester, Academic_Year,
            Classes_Held, Classes_Attended, Attendance_Percentage, Subject

EXAMINATION (5,000 records):
  - Internal marks: ~N(28, 6) out of 40
  - External marks: ~N(38, 10) out of 60, correlated with internal
  - Total out of 100; Average: 67.2; Range: 25-100
  - Per-subject pass rate: ~88-93%
  - Fields: Enrollment_No, Subject, Internal_Marks, External_Marks,
            Total_Marks, Grade, Subject_Code, Semester

FEES (1,000 records):
  - ~80% fully paid, ~10% partial, ~6% unpaid, rest overdue
  - Total Fees: Rs 9,05,75,000 | Collected: Rs 7,81,76,750 | Pending: Rs 1,23,98,250
  - Weak negative correlation with attendance

FACULTY (~58 records):
  - ~1:18 student-faculty ratio; 1 HOD per department
  - Designations: 55% Asst. Professor / 25% Assoc. Professor / 20% Professor
  - Weekly teaching load: 10-18 hours (HOD <= 10 hours)

LIBRARY TRANSACTIONS:
  - ~8-10 issues per active student per year
  - ~6% overdue returns

NOTICES (~40 records):
  - Mixed priority and audience over 4 months
  - 5 currently Urgent/Active


================================================================================
6. INSTALLATION & SETUP
================================================================================

PREREQUISITES:
  - Python 3.9 or higher (tested with 3.11)
  - pip (Python package manager)
  - Git (optional, for cloning)

STEP 1: Clone or extract the project

  git clone <repository-url>
  cd erp-datawarehouse-design

  OR extract the submitted ZIP file and navigate to the project folder.

STEP 2: Create a virtual environment

  python3 -m venv .venv

STEP 3: Activate the virtual environment

  macOS / Linux:
    source .venv/bin/activate

  Windows:
    .venv\Scripts\activate

STEP 4: Install dependencies

  pip install -r requirements.txt

  Required packages:
    - flask>=3.0.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scikit-learn>=1.3.0
    - joblib>=1.3.0
    - mysql-connector-python>=8.0.0  (optional — only for MySQL mode)

STEP 5: Verify installation

  python3 -c "import flask, pandas, sklearn; print('All dependencies OK')"


================================================================================
7. HOW TO RUN
================================================================================

OPTION A: Direct run (recommended)

  python3 app.py

  The server starts at: http://127.0.0.1:5001
  Open this URL in your web browser.

OPTION B: If port 5001 is occupied (macOS AirPlay Receiver)

  python3 -c "from app import app; app.run(debug=True, port=5002)"

  Then open: http://127.0.0.1:5002

OPTION C: Retrain ML models (optional — pre-trained models included)

  python3 ml/train.py

  This regenerates all model files in the models/ directory.

STOPPING THE SERVER:
  Press Ctrl+C in the terminal.

NOTE: No database installation is required for the demo. The system supports
both CSV and MySQL backends. CSV mode is used in this submission for
portability — the project runs out of the box without any database setup.


================================================================================
8. LOGIN CREDENTIALS
================================================================================

  +----------+----------+-----------+---------------------------------+
  | Username | Password | Role      | Access Scope                    |
  +----------+----------+-----------+---------------------------------+
  | admin    | admin123 | ADMIN     | Full system access              |
  | hod_cse  | hod123   | HOD       | CSE department only             |
  | hod_ece  | hod123   | HOD       | ECE department only             |
  | hod_eee  | hod123   | HOD       | EEE department only             |
  | hod_it   | hod123   | HOD       | IT department only              |
  | hod_aids | hod123   | HOD       | AIDS department only            |
  | hod_aiml | hod123   | HOD       | AIML department only            |
  | fac_001  | fac123   | FACULTY   | Assigned sections only, no fees |
  +----------+----------+-----------+---------------------------------+

  RBAC Enforcement:
  - ADMIN: Can access all modules, add students, view reports
  - HOD: Can access only their own department's data
  - FACULTY: Can access only assigned sections; CANNOT view fees data


================================================================================
9. MODULE DESCRIPTIONS
================================================================================

DASHBOARD (/)
  - Institutional KPIs: total students, avg attendance, avg CGPA, total fees
  - Department performance overview with progress bars
  - At-risk student count (ML-predicted)
  - Examination distribution by grade band
  - Finance snapshot with collection rate
  - Attention indicators (below-threshold alerts)

ALL STUDENTS (/students)
  - Paginated student directory with search and filter
  - Filter by Department, Section, Semester
  - Links to individual student profiles

ADD NEW STUDENT (/students/add) — ADMIN ONLY
  - Complete form with input validation
  - Name: letters only (no numbers)
  - Phone: exactly 10 digits (no letters)
  - DOB: between 2000-2008 (admission age 17-20)
  - Auto-generates Enrollment ID (STU1001, STU1002, etc.)
  - Saves to CSV files and refreshes data store immediately

STUDENT PROFILE (/profile/<enrollment_no>)
  - Personal information, academic history, attendance breakdown
  - AI Risk Card: risk probability, risk band, top-3 reason codes
  - SGPA Prediction and predicted grade
  - Anomaly detection warning (if flagged)
  - Semester-wise SGPA history chart
  - Fee payment status and library activity

ATTENDANCE (/attendance)
  - Section-wise breakdown with per-student per-subject attendance
  - Overall average: 82.1% | Low attendance students: 155
  - Subject-wise filter and search
  - Status badges: Good (>=85%), Average (75-84%), Low (<75%)

EXAMINATION (/examination)
  - Subject-wise marks (Internal + External = Total out of 100)
  - Grade distribution: O, A+, A, B+, B, C, D, F
  - Average marks: 67.2 | Highest: 100 | Lowest: 25

FEES & PAYMENTS (/fees) — ADMIN & HOD ONLY
  - Payment status tracking: Paid / Partial / Unpaid / Overdue
  - Total: Rs 9,05,75,000 | Collected: Rs 7,81,76,750
  - Outstanding: Rs 1,23,98,250 | Collection rate: 86.3%

FACULTY DIRECTORY (/faculty)
  - 58 faculty members across 6 departments
  - Designation distribution and department counts

LIBRARY CIRCULATION (/library)
  - Book issue/return tracking
  - Overdue detection and alerts

CIRCULARS & NOTICES (/notices)
  - Priority-based display (Urgent, High, Normal, Low)
  - Audience filtering (All Students, Department-specific, Faculty)

AT-RISK RADAR (/risk)
  - ML-predicted at-risk students with risk bands (High/Medium/Low)
  - Per-student reason codes and recommended interventions
  - Sortable and filterable table

ML DIAGNOSTICS & WHAT-IF (/ml/performance) — ADMIN & HOD ONLY
  - What-If Simulator: adjust attendance/marks sliders to see risk change
  - Real-time inference using the trained Random Forest model

REPORTS (/reports)
  - Department-wise summary: enrolled count, avg attendance, avg CGPA, fees
  - Institution-wide highlights
  - Examination summary statistics
  - CSV export for all modules


================================================================================
10. MACHINE LEARNING (Phase 6)
================================================================================

PROBLEM STATEMENT:
  Identify students at risk of failing Semester 5 at MID-SEMESTER
  (after internal exams, BEFORE final/external exams) to enable
  timely academic intervention.

LABEL DEFINITION:
  at_risk = 1  if  (any Sem-5 subject failed OR Sem-5 SGPA < 6.5)
  at_risk = 0  otherwise

FEATURES (20 total — all available at prediction time, leakage-safe):
  Attendance:   overall_attendance_pct, subjects_below_75,
                min_subject_attendance, prior_sem_attendance
  Marks:        internal_marks_avg_pct, internal_marks_min_pct
  SGPA History: prior_sgpa_mean, prior_sgpa_last, prior_sgpa_trend
  Backlogs:     prior_backlogs
  Finance:      fee_pending_ratio, fee_overdue_flag
  Engagement:   library_issues_count, credits_registered
  Department:   dept_CSE, dept_ECE, dept_EEE, dept_IT, dept_AIDS, dept_AIML

EXCLUDED (leakage prevention):
  Name, Gender, DOB, Address, Email, Enrollment No,
  Sem-5 External Marks, Sem-5 Final SGPA, Sem-5 Final Grades

MODELS TRAINED:
  +--------------------+---------+--------+--------+---------+-------+--------+
  | Model              | ROC-AUC | PR-AUC | Thresh | Recall  | Prec  | Brier  |
  +--------------------+---------+--------+--------+---------+-------+--------+
  | Random Forest *    | 0.9251  | 0.7310 | 0.29   | 81.4%   | 56.5% | 0.0676 |
  | Logistic Regression| 0.9294  | 0.7661 | 0.37   | 81.4%   | 55.6% | 0.0773 |
  | Gradient Boosting  | 0.9202  | 0.7539 | 0.50   | 51.2%   | 78.6% | 0.0706 |
  | Decision Tree (d4) | 0.8252  | 0.5419 | 0.50   | 67.4%   | 46.0% | 0.1019 |
  +--------------------+---------+--------+--------+---------+-------+--------+
  * = Selected core model

WHY RANDOM FOREST:
  1. Best Brier Score (0.0676) — most calibrated probability estimates
  2. 81.4% Recall — catches 35 out of 43 at-risk students in test set
  3. Handles non-linear feature interactions without manual engineering
  4. class_weight='balanced_subsample' addresses 78/22 class imbalance
  5. Provides global feature importance and per-student reason codes

TOP FEATURE IMPORTANCES (Gini):
  1. prior_sgpa_mean        : 25.9%
  2. internal_marks_avg_pct : 25.2%
  3. internal_marks_min_pct : 13.1%
  4. prior_sgpa_last        :  9.9%
  5. prior_backlogs         :  6.7%
  6. prior_sem_attendance   :  5.0%

ADDITIONAL ML FEATURES:
  - SGPA Predictor: RF Regressor predicts final SGPA before results
  - Student Segmentation: KMeans (k=4) creates named personas
      -> Consistent High Performers
      -> Steady Average Learners
      -> High Marks, Moderate Attendance
      -> Struggling & Academically Disengaged
  - Anomaly Detection: Isolation Forest flags data quality issues
      -> Proxy attendance, marks entry errors, irregular fee records
  - What-If Simulator: real-time risk recalculation via sliders

VALIDATION:
  - Stratified 70/30 train-test split (random_state=42)
  - 5-fold Stratified Cross-Validation
  - Leakage test: tests/test_ml_no_leakage.py asserts features ∩ labels = ∅
  - Calibration curves verify probability reliability

IMPORTANT DISCLAIMER:
  All data is synthetic. Model metrics demonstrate the system recovers
  the simulated statistical relationships. They do NOT claim real-world
  predictive accuracy. All predictions are advisory and require human review.


================================================================================
11. DEMO FLOW (For Presentation)
================================================================================

  1. Open http://127.0.0.1:5001 -> Login as admin / admin123
  2. Dashboard: Show 1,000 students, 82.1% attendance, At-Risk count
  3. All Students: Browse, filter by department/section
  4. Add Student: Sidebar -> "Add New Student" -> Fill form -> Submit
  5. Attendance: Show corrected low-attendance count (155 students)
  6. Student Profile: Search STU0001 -> View AI Risk Card + SGPA prediction
  7. At-Risk Radar: View ML predictions with risk bands and reason codes
  8. What-If Simulator: Adjust sliders to demo real-time risk recalculation
  9. Reports: Department-wise summary, export CSV
  10. Login as hod_cse/hod123: Verify only CSE data is visible


================================================================================
12. API ENDPOINTS
================================================================================

  GET  /api/students         — Paginated student list (JSON)
  GET  /api/what_if          — What-If risk recalculation
  GET  /api/data-quality     — Data quality report (JSON)
  GET  /reports/export/<type>— CSV export (departments/students/attendance/
                                examination/fees)

All API endpoints require authentication via session cookie.


================================================================================
13. TESTING
================================================================================

Run all tests:

  python3 -m pytest tests/ -v
  python3 -m pytest test_*.py -v

Key test files:
  - tests/test_ml_no_leakage.py  : Verifies ML feature set has no label leakage
  - tests/test_rbac_matrix.py    : Verifies role-based access permissions
  - test_csv_mode.py             : End-to-end CSV mode integration tests
  - test_attendance_fix.py       : Verifies low-attendance count fix


================================================================================
14. KNOWN LIMITATIONS
================================================================================

  - Data is synthetic — ML metrics reflect simulated relationships only
  - CSV mode stores data in files (not production-grade concurrency)
  - Single-server Flask development server (not for production deployment)
  - No email/SMS notification system for at-risk alerts
  - No edit/deactivate student UI (CLI script available)
  - Student photos not supported


================================================================================
15. FUTURE ENHANCEMENTS
================================================================================

  - Integration with real university database systems
  - Email/SMS alerts for at-risk students to academic advisers
  - Student edit and deactivate functionality via admin UI
  - LMS integration for assignment submission tracking
  - Real-time attendance via RFID/biometric integration
  - SHAP-based per-student explainability (replaces rule-based reasons)
  - Temporal backtesting across multiple academic years
  - Mobile-responsive progressive web app (PWA)
  - Production deployment with Gunicorn + Nginx + PostgreSQL


================================================================================
END OF README
================================================================================

