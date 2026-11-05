"""Authentication, Security, and Role-Based Access Control (RBAC).

Implements:
- Roles: ADMIN (institution-wide), HOD (department-scoped), FACULTY (assigned sections/courses).
- Password hashing with werkzeug.security.
- In-memory login rate-limiting (max 5 failed attempts per 5 minutes).
- CSRF protection for state-changing HTTP requests.
- Server-side scope enforcement for database and API queries.
"""

from __future__ import annotations
import os
import secrets
import time
from functools import wraps
from typing import Any, Callable
from flask import session, request, redirect, url_for, flash, abort, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

import config

# In-memory rate limiting store: key -> list of failed attempt timestamps
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}


def is_rate_limited(identifier: str) -> bool:
    """Check if the given IP or username has exceeded max failed attempts."""
    now = time.time()
    cutoff = now - config.LOGIN_LOCKOUT_SECONDS
    attempts = [t for t in _LOGIN_ATTEMPTS.get(identifier, []) if t > cutoff]
    _LOGIN_ATTEMPTS[identifier] = attempts
    return len(attempts) >= config.LOGIN_MAX_ATTEMPTS


def record_failed_attempt(identifier: str) -> None:
    now = time.time()
    if identifier not in _LOGIN_ATTEMPTS:
        _LOGIN_ATTEMPTS[identifier] = []
    _LOGIN_ATTEMPTS[identifier].append(now)


def clear_failed_attempts(identifier: str) -> None:
    _LOGIN_ATTEMPTS.pop(identifier, None)


# ==========================================
# SEEDED USERS WITH HASHED PASSWORDS
# ==========================================
def get_user_store() -> dict[str, dict[str, Any]]:
    admin_pw = os.environ.get("ADMIN_PASSWORD", "admin123")
    hod_pw = os.environ.get("HOD_PASSWORD", "hod123")
    faculty_pw = os.environ.get("FACULTY_PASSWORD", "faculty123")

    return {
        "admin": {
            "username": "admin",
            "password_hash": generate_password_hash(admin_pw),
            "role": "ADMIN",
            "name": "Institutional Administrator",
            "department": "ALL",
            "title": "Super Administrator",
            "faculty_id": None,
            "assigned_sections": ["ALL"],
        },
        "hod_cse": {
            "username": "hod_cse",
            "password_hash": generate_password_hash(hod_pw),
            "role": "HOD",
            "name": "Dr. Ramesh Sharma",
            "department": "CSE",
            "title": "Head of Department (CSE)",
            "faculty_id": "FAC001",
            "assigned_sections": ["CSE-A", "CSE-B", "CSE-C", "CSE-D"],
        },
        "hod_ece": {
            "username": "hod_ece",
            "password_hash": generate_password_hash(hod_pw),
            "role": "HOD",
            "name": "Dr. Suresh Rao",
            "department": "ECE",
            "title": "Head of Department (ECE)",
            "faculty_id": "FAC011",
            "assigned_sections": ["ECE-A", "ECE-B", "ECE-C", "ECE-D"],
        },
        "hod_eee": {
            "username": "hod_eee",
            "password_hash": generate_password_hash(hod_pw),
            "role": "HOD",
            "name": "Dr. Lakshmi Reddy",
            "department": "EEE",
            "title": "Head of Department (EEE)",
            "faculty_id": "FAC021",
            "assigned_sections": ["EEE-A", "EEE-B", "EEE-C", "EEE-D"],
        },
        "hod_it": {
            "username": "hod_it",
            "password_hash": generate_password_hash(hod_pw),
            "role": "HOD",
            "name": "Dr. Ananya Verma",
            "department": "IT",
            "title": "Head of Department (IT)",
            "faculty_id": "FAC030",
            "assigned_sections": ["IT-A", "IT-B", "IT-C", "IT-D"],
        },
        "hod_aids": {
            "username": "hod_aids",
            "password_hash": generate_password_hash(hod_pw),
            "role": "HOD",
            "name": "Dr. Venkatesh Murthy",
            "department": "AIDS",
            "title": "Head of Department (AIDS)",
            "faculty_id": "FAC040",
            "assigned_sections": ["AIDS-A", "AIDS-B", "AIDS-C", "AIDS-D"],
        },
        "hod_aiml": {
            "username": "hod_aiml",
            "password_hash": generate_password_hash(hod_pw),
            "role": "HOD",
            "name": "Dr. Pooja Gupta",
            "department": "AIML",
            "title": "Head of Department (AIML)",
            "faculty_id": "FAC050",
            "assigned_sections": ["AIML-A", "AIML-B", "AIML-C", "AIML-D"],
        },
        "fac_001": {
            "username": "fac_001",
            "password_hash": generate_password_hash(faculty_pw),
            "role": "FACULTY",
            "name": "Dr. Ramesh Sharma",
            "department": "CSE",
            "title": "Assistant Professor (CSE)",
            "faculty_id": "FAC001",
            "assigned_sections": ["CSE-A", "CSE-B"],
        },
    }


# Singleton user registry
USERS = get_user_store()


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    user = USERS.get(username.strip().lower())
    if not user:
        return None
    if check_password_hash(user["password_hash"], password):
        return user
    return None


# ==========================================
# CSRF PROTECTION
# ==========================================
def get_csrf_token() -> str:
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(24)
    return session["csrf_token"]


def validate_csrf():
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not token or token != session.get("csrf_token"):
            if request.is_json or request.path.startswith("/api/"):
                abort(400, description="Invalid or missing CSRF token.")
            flash("Session expired or invalid security token. Please try again.", "danger")
            abort(400)


# ==========================================
# RBAC ACCESS DECORATORS
# ==========================================
def login_required(f: Callable) -> Callable:
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized", "message": "Login required"}), 401
            flash("Please log in to access the institutional portal.", "warning")
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


def roles_required(*allowed_roles: str) -> Callable:
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user" not in session:
                if request.path.startswith("/api/"):
                    return jsonify({"error": "Unauthorized"}), 401
                return redirect(url_for("login"))

            user_role = session["user"].get("role", "")
            if user_role not in allowed_roles:
                if request.path.startswith("/api/"):
                    return jsonify({"error": "Forbidden", "message": "Access denied for your role"}), 403
                flash("Access denied: You do not have permission to view this resource.", "danger")
                abort(403)
            return f(*args, **kwargs)
        return decorated
    return decorator


def check_student_scope(student_dept: str, student_section: str | None = None) -> bool:
    """Verify whether the currently logged-in user is authorized to view this student."""
    user = session.get("user")
    if not user:
        return False
    role = user.get("role")
    if role == "ADMIN":
        return True
    if role == "HOD":
        return student_dept == user.get("department")
    if role == "FACULTY":
        if student_dept != user.get("department"):
            return False
        if student_section and student_section not in user.get("assigned_sections", []):
            return False
        return True
    return False

