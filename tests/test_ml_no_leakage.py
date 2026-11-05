"""Leakage-safe assertions for machine learning features (Phase 6).

Asserts that no label source, identifier, or post-internals examination attribute
leaks into the predictive feature space.
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.features import (
    FEATURE_COLUMNS,
    EXCLUDED_LABEL_SOURCES,
    EXCLUDED_IDENTIFIERS,
    extract_features_and_labels,
)


class MlNoLeakageTests(unittest.TestCase):
    def test_feature_set_disjoint_from_label_sources(self):
        feature_set = set(FEATURE_COLUMNS)
        # 1. Feature set must be strictly disjoint from label sources
        intersection = feature_set.intersection(EXCLUDED_LABEL_SOURCES)
        self.assertEqual(
            intersection,
            set(),
            f"Data leakage detected! Features contain label sources: {intersection}"
        )

    def test_feature_set_disjoint_from_identifiers(self):
        feature_set = set(FEATURE_COLUMNS)
        # 2. Feature set must not contain demographic or student identifiers
        intersection = feature_set.intersection(EXCLUDED_IDENTIFIERS)
        self.assertEqual(
            intersection,
            set(),
            f"Feature set contains student identity information: {intersection}"
        )

    def test_no_external_marks_or_current_semester_outcomes(self):
        for col in FEATURE_COLUMNS:
            col_lower = col.lower()
            self.assertNotIn("external", col_lower, f"External marks leak found in {col}")
            self.assertNotIn("total_marks", col_lower, f"Total marks leak found in {col}")
            self.assertNotIn("grade", col_lower, f"Grade leak found in {col}")
            # Ensure any current semester SGPA is NOT in feature columns
            if "sgpa" in col_lower:
                self.assertTrue(
                    col_lower.startswith("prior_"),
                    f"SGPA feature must be historical only (prefix prior_): {col}"
                )

    def test_feature_extraction_shape_and_types(self):
        X, y, meta = extract_features_and_labels()
        self.assertEqual(len(X), 1000)
        self.assertEqual(len(y), 1000)
        self.assertEqual(len(meta), 1000)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertFalse(X.isnull().any().any(), "Features contain unexpected NaN values")
        self.assertTrue(set(y.unique()).issubset({0, 1}), "Label must be binary (0 or 1)")


if __name__ == "__main__":
    unittest.main()

