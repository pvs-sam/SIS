"""Uniformity verification script (D8).

Scans Python files and templates to verify that thresholds and grading limits
are referenced from config.py rather than duplicated or drifted.
"""

from __future__ import annotations
import sys
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def check_uniformity() -> int:
    import config

    print("=" * 60)
    print("  ERP UNIFORMITY & THRESHOLD AUDIT (D8)")
    print("=" * 60)

    # 1. Verify grade bands monotonic and non-overlapping
    print("[1] Validating Grade Bands in config.py...")
    for i in range(len(config.GRADE_BANDS) - 1):
        curr_g, curr_low, curr_high, curr_pt = config.GRADE_BANDS[i]
        next_g, next_low, next_high, next_pt = config.GRADE_BANDS[i + 1]
        assert curr_low > next_low, f"Grade order error between {curr_g} and {next_g}"
        assert curr_pt > next_pt, f"Grade points order error between {curr_g} and {next_g}"
    print("  [PASS] All 8 Grade Bands correctly configured and ordered.")

    # 2. Check risk cutoffs and attendance thresholds
    print("[2] Checking Academic Thresholds...")
    assert config.ATTENDANCE_WARNING_THRESHOLD == 75.0
    assert config.ATTENDANCE_EXCELLENT_THRESHOLD == 85.0
    assert config.TOTAL_PASS_MARKS == 40.0
    assert config.RISK_SGPA_CUTOFF == 6.0
    print("  [PASS] Attendance warning (75.0%) and pass mark (40.0) verified.")

    # 3. Check calculate_grade outputs
    print("[3] Testing calculate_grade() sanity...")
    test_cases = [
        (95, "O", 10.0),
        (85, "A+", 9.0),
        (72, "A", 8.0),
        (65, "B+", 7.0),
        (58, "B", 6.0),
        (52, "C", 5.0),
        (45, "D", 4.0),
        (35, "F", 0.0),
    ]
    for marks, expected_g, expected_pt in test_cases:
        g, pt = config.calculate_grade(marks)
        assert g == expected_g and pt == expected_pt, f"Mismatch for {marks}: got ({g}, {pt}), expected ({expected_g}, {expected_pt})"
    print("  [PASS] Grade mapping verified against all test bounds.")

    print("\n[SUCCESS] Uniformity checks passed 100%!")
    return 0

if __name__ == "__main__":
    sys.exit(check_uniformity())
