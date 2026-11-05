"""Central configuration for ERP Data Warehouse and Decision Support System.

Defines all institutional thresholds, academic calendar parameters, grading bands,
risk model cutoffs, and role-based permissions in one authoritative location (D8).
"""

from __future__ import annotations
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# ACADEMIC CALENDAR & INSTITUTIONAL SETTINGS
# ==========================================
INSTITUTION_NAME = "ICFAI Foundation for Higher Education"
INSTITUTION_SHORT = "IFHE"
CAMPUS_LOCATION = "Hyderabad"

BATCH = "2023-2027"
CURRENT_SEMESTER = 5
CURRENT_ACADEMIC_YEAR = "2025-2026"
CURRENT_TERM = "2025-26 Odd"
AS_OF_DATE = "2026-01-15"

DEPARTMENTS = ["CSE", "ECE", "EEE", "IT", "AIDS", "AIML"]
SECTIONS = ["A", "B", "C", "D"]

# Age boundaries at admission (Year 2023)
MIN_AGE_AT_ADMISSION = 17
MAX_AGE_AT_ADMISSION = 20
DOB_START = "2003-08-01"
DOB_END = "2006-12-31"
DOB_MODE_YEAR = 2005

# ==========================================
# THRESHOLDS & UNIFORMITY RULES (D8)
# ==========================================
ATTENDANCE_WARNING_THRESHOLD = 75.0      # Mandatory minimum attendance %
ATTENDANCE_EXCELLENT_THRESHOLD = 85.0    # Commendable attendance %

CLASSES_HELD_MIN = 36
CLASSES_HELD_MAX = 48

# Examination marks allocation
INTERNAL_MAX_MARKS = 40.0
EXTERNAL_MAX_MARKS = 60.0
TOTAL_MAX_MARKS = 100.0

INTERNAL_PASS_MARKS = 16.0  # 40% of 40
EXTERNAL_PASS_MARKS = 24.0  # 40% of 60
TOTAL_PASS_MARKS = 40.0     # 40% of 100

# Standard university letter grade boundaries and grade points
GRADE_BANDS = [
    ("O",  90.0, 100.0, 10.0),
    ("A+", 80.0, 89.99,  9.0),
    ("A",  70.0, 79.99,  8.0),
    ("B+", 60.0, 69.99,  7.0),
    ("B",  55.0, 59.99,  6.0),
    ("C",  50.0, 54.99,  5.0),
    ("D",  40.0, 49.99,  4.0),
    ("F",   0.0, 39.99,  0.0),
]

def calculate_grade(total_marks: float) -> tuple[str, float]:
    """Return (Grade, Grade_Point) for a given total mark."""
    m = round(float(total_marks), 2)
    for grade, low, high, point in GRADE_BANDS:
        if low <= m <= high:
            return grade, point
    return "F", 0.0

# ==========================================
# RISK PREDICTION & ML THRESHOLDS (PHASE 6)
# ==========================================
RISK_SGPA_CUTOFF = 6.0
RISK_TARGET_RECALL = 0.80

RISK_BAND_LOW_MAX = 0.33
RISK_BAND_MED_MAX = 0.66

def get_risk_band(prob: float) -> str:
    """Classify probability into Low, Medium, or High risk."""
    if prob < RISK_BAND_LOW_MAX:
        return "Low"
    elif prob < RISK_BAND_MED_MAX:
        return "Medium"
    return "High"

# ==========================================
# SECURITY & RBAC CONFIGURATION
# ==========================================
SECRET_KEY = os.environ.get("SECRET_KEY", "erp-academic-dw-2026-production-secret-token")
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

ROLES = ["ADMIN", "HOD", "FACULTY"]

# Rate limiting settings
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300  # 5 minutes

