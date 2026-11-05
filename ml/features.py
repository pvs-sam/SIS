"""Leakage-safe Feature Extraction for ML Early-Warning Risk Prediction (Phase 6).

PREDICTION POINT: Mid Semester 5 (after internals are declared, before external exams).

Excluded from feature space:
- Student identifiers / demographics (Name, Gender, DOB, Phone, Email, Address, Enrollment_No)
- Final semester outcome / leakage sources (Sem 5 External_Marks, Total_Marks, Grade, SGPA, CGPA)

Features available at prediction time:
- Sem-5 Attendance metrics (overall %, subjects below 75, minimum subject attendance %)
- Sem-5 Internal marks metrics (average %, minimum %)
- Prior Academic performance (Sem 1–4 mean SGPA, Sem 4 SGPA, trend, prior backlogs)
- Prior Semester attendance (Sem 4 attendance %)
- Financial & Engagement signals (fee pending ratio, overdue flag, library circulation count)
- Curriculum load (credits registered, department one-hot)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

EXCLUDED_LABEL_SOURCES = {
    "External_Marks", "Total_Marks", "Grade", "SGPA", "CGPA", "Semester_Result"
}

EXCLUDED_IDENTIFIERS = {
    "Enrollment_No", "Name", "Gender", "Date_of_Birth", "Address", "Email", "Phone_No"
}

FEATURE_COLUMNS = [
    "overall_attendance_pct",
    "subjects_below_75",
    "min_subject_attendance",
    "internal_marks_avg_pct",
    "internal_marks_min_pct",
    "prior_sgpa_mean",
    "prior_sgpa_last",
    "prior_sgpa_trend",
    "prior_backlogs",
    "prior_sem_attendance",
    "fee_pending_ratio",
    "fee_overdue_flag",
    "library_issues_count",
    "credits_registered",
    "dept_CSE",
    "dept_ECE",
    "dept_EEE",
    "dept_IT",
    "dept_AIDS",
    "dept_AIML",
]


def extract_features_and_labels(
    data_dir: Path | str | None = None,
    target_sem: int = 5
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Extract features and risk label for all 1,000 students at mid-term of target_sem.
    
    Returns:
        X (pd.DataFrame): Leakage-safe feature matrix.
        y (pd.Series): Binary at-risk label (1 = at-risk, 0 = safe).
        meta (pd.DataFrame): Metadata (Enrollment_No, Department) for tracking/explainability.
    """
    base = Path(data_dir) if data_dir else (BASE_DIR / "data")
    students_df = pd.read_csv(base / "students.csv")
    att_hist = pd.read_csv(base / "attendance_history.csv")
    exam_hist = pd.read_csv(base / "examination_history.csv")
    terms_df = pd.read_csv(base / "student_terms.csv")
    fees_df = pd.read_csv(base / "fees.csv")
    lib_df = pd.read_csv(base / "library_transactions.csv")

    rows = []
    labels = []
    meta_rows = []

    for _, srow in students_df.iterrows():
        enr = srow["Enrollment_No"]
        dept = srow["Department"]

        # 1. Sem 5 Attendance Features
        att_sem = att_hist[(att_hist["Enrollment_No"] == enr) & (att_hist["Semester"] == target_sem)]
        if not att_sem.empty:
            att_pcts = (att_sem["Classes_Attended"] / att_sem["Classes_Held"]) * 100.0
            overall_att = float(att_pcts.mean())
            sub_below_75 = int((att_pcts < config.ATTENDANCE_WARNING_THRESHOLD).sum())
            min_att = float(att_pcts.min())
        else:
            overall_att = 80.0
            sub_below_75 = 0
            min_att = 80.0

        # 2. Sem 5 Internal Marks Features (Available at mid-term; internals out of 40)
        exam_sem = exam_hist[(exam_hist["Enrollment_No"] == enr) & (exam_hist["Semester"] == target_sem)]
        if not exam_sem.empty:
            int_pcts = (exam_sem["Internal_Marks"] / config.INTERNAL_MAX_MARKS) * 100.0
            internal_avg_pct = float(int_pcts.mean())
            internal_min_pct = float(int_pcts.min())
        else:
            internal_avg_pct = 70.0
            internal_min_pct = 60.0

        # 3. Prior Academic History (Semesters 1 through target_sem - 1)
        prior_terms = terms_df[(terms_df["Enrollment_No"] == enr) & (terms_df["Semester"] < target_sem)]
        if not prior_terms.empty:
            prior_sgpas = prior_terms["SGPA"].tolist()
            prior_sgpa_mean = float(np.mean(prior_sgpas))
            prior_sgpa_last = float(prior_sgpas[-1])
            prior_sgpa_trend = float(prior_sgpas[-1] - prior_sgpas[0])
            prior_backlogs = int(prior_terms["Backlogs_Incurred"].sum())
        else:
            prior_sgpa_mean = 7.0
            prior_sgpa_last = 7.0
            prior_sgpa_trend = 0.0
            prior_backlogs = 0

        # 4. Prior Semester Attendance (Semester target_sem - 1)
        prior_att_df = att_hist[(att_hist["Enrollment_No"] == enr) & (att_hist["Semester"] == (target_sem - 1))]
        if not prior_att_df.empty:
            prior_sem_att = float(((prior_att_df["Classes_Attended"] / prior_att_df["Classes_Held"]) * 100.0).mean())
        else:
            prior_sem_att = overall_att

        # 5. Financial & Fee Signals
        fee_row = fees_df[(fees_df["Enrollment_No"] == enr) & (fees_df["Semester"] == target_sem)]
        if not fee_row.empty:
            tot_f = float(fee_row.iloc[0]["Total_Fees"])
            pend_f = float(fee_row.iloc[0]["Pending_Amount"])
            fee_ratio = (pend_f / tot_f) if tot_f > 0 else 0.0
            fee_overdue = 1 if fee_row.iloc[0]["Payment_Status"] == "Overdue" else 0
        else:
            fee_ratio = 0.0
            fee_overdue = 0

        # 6. Library Engagement (Transactions in current academic term)
        lib_tx = lib_df[lib_df["Enrollment_No"] == enr]
        lib_count = len(lib_tx)

        # 7. Curriculum Load & One-hot department
        target_term_row = terms_df[(terms_df["Enrollment_No"] == enr) & (terms_df["Semester"] == target_sem)]
        credits_reg = int(target_term_row.iloc[0]["Credits_Registered"]) if not target_term_row.empty else 20

        feature_dict = {
            "overall_attendance_pct": round(overall_att, 2),
            "subjects_below_75": sub_below_75,
            "min_subject_attendance": round(min_att, 2),
            "internal_marks_avg_pct": round(internal_avg_pct, 2),
            "internal_marks_min_pct": round(internal_min_pct, 2),
            "prior_sgpa_mean": round(prior_sgpa_mean, 2),
            "prior_sgpa_last": round(prior_sgpa_last, 2),
            "prior_sgpa_trend": round(prior_sgpa_trend, 2),
            "prior_backlogs": prior_backlogs,
            "prior_sem_attendance": round(prior_sem_att, 2),
            "fee_pending_ratio": round(fee_ratio, 3),
            "fee_overdue_flag": fee_overdue,
            "library_issues_count": lib_count,
            "credits_registered": credits_reg,
            "dept_CSE": 1 if dept == "CSE" else 0,
            "dept_ECE": 1 if dept == "ECE" else 0,
            "dept_EEE": 1 if dept == "EEE" else 0,
            "dept_IT": 1 if dept == "IT" else 0,
            "dept_AIDS": 1 if dept == "AIDS" else 0,
            "dept_AIML": 1 if dept == "AIML" else 0,
        }
        rows.append(feature_dict)

        # 8. Ground Truth Label (Formed strictly from final results / external exams)
        # Label at_risk_sem5 = 1 if (>= 1 failed subject in Sem 5 OR sem-5 SGPA < RISK_SGPA_CUTOFF)
        has_failed_subject = False
        if not exam_sem.empty:
            totals = exam_sem["Internal_Marks"] + exam_sem["External_Marks"]
            if (totals < config.TOTAL_PASS_MARKS).any():
                has_failed_subject = True

        sem_sgpa = float(target_term_row.iloc[0]["SGPA"]) if not target_term_row.empty else 7.0
        is_risk = 1 if (has_failed_subject or sem_sgpa < config.RISK_SGPA_CUTOFF) else 0
        labels.append(is_risk)

        meta_rows.append({
            "Enrollment_No": enr,
            "Name": srow["Name"],
            "Department": dept,
            "Actual_SGPA": sem_sgpa,
            "Actual_Failed_Count": int((totals < config.TOTAL_PASS_MARKS).sum()) if not exam_sem.empty else 0
        })

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    y = pd.Series(labels, name="at_risk")
    meta = pd.DataFrame(meta_rows)
    return X, y, meta

