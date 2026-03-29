# airman-student-risk-model
End-to-end data science solution to identify at-risk students using synthetic data, machine learning models, and actionable recommendations.
# AIRMAN Data Science Technical Assessment

##  Overview
This project builds an end-to-end data science solution to identify students at risk and provide actionable insights for instructors. The system analyzes student engagement, performance, and behavioral data to classify students into risk categories and generate recommendations.

---

##  Objectives
- Identify students at risk (on_track / needs_attention / critical)
- Understand key drivers of performance
- Build an interpretable predictive model
- Provide actionable recommendations for instructors

---

##  Dataset
Since no dataset was provided, a synthetic dataset was generated to simulate real-world student training data.

### Key Features:
- **Behavioral**: attendance_pct, missed_sessions_count, late_submission_count
- **Performance**: quiz_avg_score, simulator_score_avg
- **Engagement**: days_since_last_activity
- **Ratings**: technical_rating, instructor_eval_avg
- **Operational**: prior_intervention_flag, schedule_changes_count

The dataset was designed with realistic relationships:
- Lower attendance → lower performance
- Higher inactivity → higher risk
- Behavioral patterns influence outcomes

---

##  Approach

### 1. Data Understanding
- No missing values or duplicates
- Features fall within realistic ranges
- Moderate class imbalance observed

### 2. Exploratory Data Analysis
Key findings:
- Attendance strongly impacts outcomes
- Inactivity is a strong risk signal
- Performance metrics correlate with success
- Behavioral features (missed sessions) are critical indicators

---

##  Models Used
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost

###  Model Performance
- Random Forest and XGBoost achieved highest accuracy (~88%)
- Logistic Regression struggled with intermediate class
- Ensemble models captured non-linear relationships better

---

## Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Key Insight:
- False negatives (missing critical students) are most dangerous

---

##  Feature Importance
Top drivers of risk:
- attendance_pct
- quiz_avg_score
- missed_sessions_count
- days_since_last_activity

These align with real-world expectations of student performance.

---

## Product Output (Key Highlight)

For each student, the system generates:

- Risk Label
- Risk Score
- Top Contributing Factors
- Recommendation

### Example Output:
```json
{
  "student_id": "STD_042",
  "risk_label": "needs_attention",
  "risk_score": 71,
  "top_factors": ["Low attendance", "High missed sessions"],
  "recommendation": "Monitor and provide targeted support"
}
