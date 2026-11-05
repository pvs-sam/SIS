# Model Card — Institutional Student Academic At-Risk Predictor

## 1. Model Overview
- **Model Architecture**: Random Forest Classifier (`n_estimators=300`, `class_weight='balanced_subsample'`, `min_samples_leaf=3`)
- **Tuned Threshold**: `0.29` (calibrated to ensure target recall $\ge 0.80$)
- **Task**: Binary classification of mid-term student academic risk ($y = 1$ for at-risk, $y = 0$ for safe)
- **Prediction Point**: Mid-Semester 5, immediately upon compilation of internal assessment marks and prior to final external exams.

## 2. Intended Use & Ethical Boundaries
> [!IMPORTANT]
> **Advisory Use Only**: This model is trained on a synthetic demonstration dataset representing Semester 5, Academic Year 2025–2026.
> Its predictions are purely advisory decision support aids for academic advisers and department heads. It must NEVER be used for automated punitive actions, grade adjustments, or admissions sanctions. All flagged students require human review and pastoral evaluation.

## 3. Data & Leakage-Safe Feature Design
- **Cohort**: 1,000 undergraduate engineering students across CSE, ECE, EEE, IT, AIDS, and AIML.
- **Strictly Excluded**: Names, gender, date of birth, residential addresses, contact details, enrollment numbers, and all Semester 5 external exam marks, semester grade points, and final grades.
- **Input Features (14 metrics + 6 department indicators)**:
  - Attendance metrics (overall percentage, count of subjects < 75%, lowest subject attendance)
  - Internal assessment metrics (average percentage out of 40, minimum subject score)
  - Historical academic trajectory (Sem 1–4 average SGPA, last SGPA, trend, cumulative backlogs)
  - Prior term engagement (Semester 4 attendance)
  - Financial & library signals (fee pending ratio, fee overdue status, circulation count)
  - Departmental indicators (one-hot)

## 4. Performance Metrics (70/30 Stratified Split)
| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | `0.9251` |
| **PR-AUC** | `0.731` |
| **Recall (At-Risk)** | `81.4%` |
| **Precision** | `56.5%` |
| **F1 Score** | `0.6667` |
| **Brier Score** | `0.0676` |

- **SGPA Regression Predictor**: MAE = `0.412`, RMSE = `0.530`

## 5. Global Feature Importance (Top Predictors)
- **internal_marks_avg_pct**: 0.015
- **fee_pending_ratio**: 0.0013
- **dept_EEE**: 0.001
- **dept_ECE**: 0.0007
- **credits_registered**: 0.0
- **dept_CSE**: -0.0003

## 6. Model Limitations
- Trained on synthetic distribution curves calibrated for institutional demonstration.
- Out-of-distribution shifts (such as a curriculum overhaul or policy changes to attendance rules) require re-calibrating the model thresholds.
