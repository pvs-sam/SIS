"""Validated data access for CSV development mode and hosted MySQL mode.

Enforces:
- Schema validation and referential integrity (D1, D5, D6).
- Unified derivation of academic and financial KPIs without raw stored drift.
- Historical terms (Semesters 1–5).
- Supplementary institutional modules: Faculty, Notices, Library, Admissions, and ML Risk Profiles.
- Server-side role-based scoping (ADMIN, HOD, FACULTY).
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from ml.predict import RiskPredictorService

DEFAULT_CSV_DIR = BASE_DIR / "erp_semester5_etl_package"
DATA_DIR = BASE_DIR / "data"

REQUIRED_COLUMNS = {
    "students": {
        "Enrollment_No", "Name", "Phone_No", "Email", "Gender",
        "Department", "Section", "Semester", "Address", "Academic_Year", "CGPA", "Date_of_Birth"
    },
    "attendance": {"Enrollment_No", "Subject", "Attendance_Percentage"},
    "examination": {
        "Enrollment_No", "Subject", "Internal_Marks", "External_Marks",
        "Total_Marks", "Grade",
    },
    "fees": {
        "Enrollment_No", "Semester", "Total_Fees", "Paid_Amount",
        "Pending_Amount", "Payment_Status",
    },
}

CSV_FILES = {
    "students": "students (2).csv",
    "attendance": "attendance.csv",
    "examination": "examination.csv",
    "fees": "fees.csv",
}


class DataValidationError(ValueError):
    """Raised when the configured source does not satisfy warehouse rules."""


@dataclass(frozen=True)
class DataQuality:
    row_counts: dict[str, int]
    missing_values: dict[str, dict[str, int]]
    duplicate_student_ids: int
    invalid_references: dict[str, int]
    reconciliation_errors: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_counts": self.row_counts,
            "missing_values": self.missing_values,
            "duplicate_student_ids": self.duplicate_student_ids,
            "invalid_references": self.invalid_references,
            "reconciliation_errors": self.reconciliation_errors,
        }


@dataclass
class DataStore:
    students: pd.DataFrame
    attendance: pd.DataFrame
    examination: pd.DataFrame
    fees: pd.DataFrame
    quality: DataQuality
    terms: pd.DataFrame | None = None
    faculty: pd.DataFrame | None = None
    notices: pd.DataFrame | None = None
    library: pd.DataFrame | None = None
    admissions: pd.DataFrame | None = None

    @classmethod
    def from_csv(cls, directory: str | os.PathLike[str] | None = None) -> "DataStore":
        source_dir = Path(directory or os.getenv("CSV_DATA_DIR", DEFAULT_CSV_DIR))
        frames: dict[str, pd.DataFrame] = {}

        # 1. Load primary tables
        for dataset, filename in CSV_FILES.items():
            path = source_dir / filename
            if not path.is_file():
                # Fallback to students.csv if students (2).csv is not present
                if dataset == "students" and (source_dir / "students.csv").is_file():
                    path = source_dir / "students.csv"
                else:
                    raise DataValidationError(f"Missing {dataset} source file: {path}")
            try:
                frames[dataset] = pd.read_csv(path)
            except (OSError, pd.errors.ParserError) as exc:
                raise DataValidationError(f"Unable to read {path}: {exc}") from exc

        _validate_schema(frames)
        _normalize_types(frames)
        quality = _quality_report(frames)
        _validate_quality(quality)

        # 2. Load supplementary institutional tables if available
        def load_optional(name: str) -> pd.DataFrame | None:
            p = DATA_DIR / f"{name}.csv"
            if p.is_file():
                try:
                    return pd.read_csv(p)
                except Exception:
                    return None
            return None

        terms_df = load_optional("student_terms")
        faculty_df = load_optional("faculty")
        notices_df = load_optional("notices")
        lib_df = load_optional("library_transactions")
        adm_df = load_optional("admissions")

        return cls(
            quality=quality,
            terms=terms_df,
            faculty=faculty_df,
            notices=notices_df,
            library=lib_df,
            admissions=adm_df,
            **frames,
        )

    def student(self, enrollment_no: str, user: dict[str, Any] | None = None) -> dict[str, Any] | None:
        key = enrollment_no.strip().upper()
        matches = self.students[self.students["Enrollment_No"].str.upper() == key]
        if matches.empty:
            return None
        row = _record(matches.iloc[0])

        # Server-side scope verification (RBAC)
        if user:
            role = user.get("role", "ADMIN")
            if role == "HOD":
                if row["Department"] != user.get("department"):
                    return None
            elif role == "FACULTY":
                if row["Department"] != user.get("department"):
                    return None
                if row.get("Section") not in user.get("assigned_sections", []):
                    return None

        attendance = self.attendance[self.attendance["Enrollment_No"] == row["Enrollment_No"]]
        examination = self.examination[self.examination["Enrollment_No"] == row["Enrollment_No"]]
        fees = self.fees[self.fees["Enrollment_No"] == row["Enrollment_No"]]
        fee = _record(fees.iloc[0]) if not fees.empty else None

        # Mask fee data for faculty
        if user and user.get("role") == "FACULTY":
            fee = None

        # Historical terms (Semesters 1 to 5)
        terms_history = []
        if self.terms is not None:
            st_matches = self.terms[self.terms["Enrollment_No"] == row["Enrollment_No"]].sort_values("Semester")
            terms_history = _records(st_matches)

        # Library circulation history
        lib_history = []
        if self.library is not None:
            lib_matches = self.library[self.library["Enrollment_No"] == row["Enrollment_No"]].sort_values("Issue_Date", ascending=False)
            lib_history = _records(lib_matches)

        # Admissions profile
        adm_profile = None
        if self.admissions is not None:
            adm_matches = self.admissions[self.admissions["Enrollment_No"] == row["Enrollment_No"]]
            if not adm_matches.empty:
                adm_profile = _record(adm_matches.iloc[0])

        # Machine Learning Risk Profile & What-If
        risk_service = RiskPredictorService.get_instance()
        risk_profile = risk_service.get_student_risk(row["Enrollment_No"])

        return {
            "personal": row,
            "attendance": {
                "records": _records(attendance),
                "average": _rounded_mean(attendance["Attendance_Percentage"]),
                "count": len(attendance),
            },
            "examination": {
                "records": _records(examination),
                "average": _rounded_mean(examination["Total_Marks"]),
                "count": len(examination),
            },
            "fees": fee,
            "terms_history": terms_history,
            "library_history": lib_history,
            "admissions": adm_profile,
            "ml_risk": risk_profile,
            "source_datasets": [
                {"name": name, "file": f"CSV: {CSV_FILES[name]}", "status": "Found"}
                for name in CSV_FILES
            ],
        }

    def kpis(self, user: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compute institutional KPIs, scoped to user role if provided."""
        st = self.students
        att = self.attendance
        exam = self.examination
        fee = self.fees

        if user:
            role = user.get("role", "ADMIN")
            if role == "HOD":
                dept = user.get("department")
                st = st[st["Department"] == dept]
                dept_enrs = set(st["Enrollment_No"])
                att = att[att["Enrollment_No"].isin(dept_enrs)]
                exam = exam[exam["Enrollment_No"].isin(dept_enrs)]
                fee = fee[fee["Enrollment_No"].isin(dept_enrs)]
            elif role == "FACULTY":
                assigned_sec = user.get("assigned_sections", [])
                st = st[st["Section"].isin(assigned_sec)]
                sec_enrs = set(st["Enrollment_No"])
                att = att[att["Enrollment_No"].isin(sec_enrs)]
                exam = exam[exam["Enrollment_No"].isin(sec_enrs)]
                # Faculty has no fee access
                fee = fee.iloc[0:0]

        student_attendance = att.groupby("Enrollment_No")["Attendance_Percentage"].mean()
        attention_ids = set(student_attendance[student_attendance < config.ATTENDANCE_WARNING_THRESHOLD].index)

        # ML Risk counts
        risk_service = RiskPredictorService.get_instance()
        risk_df = risk_service.predictions_df
        if risk_df is not None and not risk_df.empty:
            scoped_enrs = set(st["Enrollment_No"])
            scoped_risk = risk_df[risk_df["Enrollment_No"].isin(scoped_enrs)]
            high_risk_count = int((scoped_risk["Risk_Band"] == "High").sum())
            at_risk_total = int(scoped_risk["Predicted_At_Risk"].sum())
        else:
            high_risk_count = 0
            at_risk_total = len(attention_ids)

        return {
            "total_students": len(st),
            "avg_attendance": _rounded_mean(att["Attendance_Percentage"]),
            "avg_marks": _rounded_mean(exam["Total_Marks"]),
            "total_fees": int(fee["Total_Fees"].sum()) if not fee.empty else 0,
            "total_paid": int(fee["Paid_Amount"].sum()) if not fee.empty else 0,
            "total_pending": int(fee["Pending_Amount"].sum()) if not fee.empty else 0,
            "students_requiring_attention": len(attention_ids),
            "at_risk_students_count": at_risk_total,
            "high_risk_count": high_risk_count,
            "departments": sorted(st["Department"].dropna().unique().tolist()),
            "sections": sorted(st["Section"].dropna().unique().tolist()),
            "semesters": sorted(st["Semester"].dropna().unique().tolist()),
            "department_counts": st["Department"].value_counts().sort_index().to_dict(),
            "quality": self.quality.as_dict(),
        }


_STORE: DataStore | None = None


def get_store() -> DataStore:
    """Return process-cached CSV store for development mode."""
    global _STORE
    if _STORE is None:
        _STORE = DataStore.from_csv()
    return _STORE


def clear_store_cache() -> None:
    global _STORE
    _STORE = None


def _validate_schema(frames: dict[str, pd.DataFrame]) -> None:
    for dataset, required in REQUIRED_COLUMNS.items():
        actual = set(frames[dataset].columns)
        missing = sorted(required - actual)
        if missing:
            raise DataValidationError(f"{dataset} is missing required columns: {', '.join(missing)}")


def _normalize_types(frames: dict[str, pd.DataFrame]) -> None:
    for frame in frames.values():
        frame["Enrollment_No"] = frame["Enrollment_No"].astype("string").str.strip()
    students = frames["students"]
    students["Semester"] = pd.to_numeric(students["Semester"], errors="coerce").astype("Int64")
    students["CGPA"] = pd.to_numeric(students["CGPA"], errors="coerce")
    for name in ("attendance", "examination", "fees"):
        frame = frames[name]
        for column in frame.columns:
            if column in {"Enrollment_No", "Subject", "Grade", "Payment_Status"}:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _quality_report(frames: dict[str, pd.DataFrame]) -> DataQuality:
    students = frames["students"]
    student_ids = set(students["Enrollment_No"].dropna())
    invalid_references = {
        name: int((~frame["Enrollment_No"].isin(student_ids)).sum())
        for name, frame in frames.items()
        if name != "students"
    }
    examination = frames["examination"]
    fees = frames["fees"]
    reconciliation_errors = {
        "examination_total": int(
            (
                (examination["Internal_Marks"] + examination["External_Marks"])
                != examination["Total_Marks"]
            ).sum()
        ),
        "fee_total": int(
            (
                (fees["Paid_Amount"] + fees["Pending_Amount"]) != fees["Total_Fees"]
            ).sum()
        ),
    }
    missing_values = {
        name: {col: int(frame[col].isna().sum()) for col in frame.columns}
        for name, frame in frames.items()
    }
    return DataQuality(
        row_counts={name: len(frame) for name, frame in frames.items()},
        missing_values=missing_values,
        duplicate_student_ids=int(students["Enrollment_No"].duplicated().sum()),
        invalid_references=invalid_references,
        reconciliation_errors=reconciliation_errors,
    )


def _validate_quality(quality: DataQuality) -> None:
    if quality.duplicate_student_ids:
        raise DataValidationError(f"Found {quality.duplicate_student_ids} duplicate student IDs")
    for table, orphans in quality.invalid_references.items():
        if orphans:
            raise DataValidationError(f"{table} references {orphans} non-existent student IDs")
    for check, mismatches in quality.reconciliation_errors.items():
        if mismatches:
            raise DataValidationError(f"{check} failed for {mismatches} records")


def _record(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for col, value in row.items():
        if pd.isna(value):
            payload[col] = None
        elif isinstance(value, (pd.Timestamp,)):
            payload[col] = value.isoformat()
        elif hasattr(value, "item"):
            payload[col] = value.item()
        else:
            payload[col] = value
    return payload


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_record(row) for _, row in frame.iterrows()]


def _rounded_mean(series: pd.Series) -> float:
    return round(float(series.dropna().mean()), 1) if not series.dropna().empty else 0.0
