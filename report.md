# AIRMAN Data Science Technical Assessment  
## Student Performance & Risk Intelligence

---

## 1. Problem Overview

The objective of this project is to build a data-driven system to identify students who are at risk and require intervention. The system aims to support instructors and administrators by providing insights into student performance, key risk drivers, and actionable recommendations.

This solution focuses on predicting student outcomes and translating those predictions into meaningful product-level outputs that can be integrated into AIRMAN platforms such as Maverick and Skynet.

---

## 2. Dataset Description

Since no dataset was provided, a synthetic dataset was generated to simulate real-world student training data. The dataset contains 800 student records and 19 features, including:

- Behavioral features (attendance, missed sessions, late submissions)
- Performance features (quiz scores, simulator scores)
- Engagement features (days since last activity)
- Instructor and skill ratings
- Operational features (prior intervention, schedule changes)

The dataset was designed with realistic relationships:
- Lower attendance leads to lower performance
- Higher inactivity indicates disengagement
- Behavioral patterns influence outcomes

Additionally, a small amount of noise was introduced to simulate real-world imperfections.

---

## 3. Data Understanding & Quality

The dataset was inspected using standard methods such as `.info()` and `.describe()`.

- No missing values were found across any features
- No duplicate records were observed
- All variables fall within realistic ranges (e.g., attendance: 40–100, ratings: 1–5)

Some mild outliers were observed in features such as missed sessions and inactivity, but these were retained as they represent meaningful real-world scenarios.

Overall, the dataset is clean, consistent, and suitable for analysis.

---

## 4. Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and relationships in the data.

### Key Insights:

- **Attendance is a strong predictor**: Students with higher attendance are more likely to be on track.
- **Inactivity is a major risk signal**: Higher days since last activity is associated with critical students.
- **Performance metrics matter**: Quiz and simulator scores strongly influence outcomes.
- **Behavioral factors are important**: Missed sessions and late submissions contribute to risk.
- **Class imbalance exists**: Most students are on track, with fewer in critical categories.
- **Institutional differences**: Some variation exists across academies, indicating possible operational influences.

These insights align with real-world expectations and validate the dataset design.

---

## 5. Modeling Approach

Four models were implemented:

- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost

Logistic Regression was used as a simple baseline model, while ensemble methods were used to capture non-linear relationships.

Categorical variables were encoded using one-hot encoding, and the target variable was label encoded.

---

## 6. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Key Findings:

- Random Forest and XGBoost achieved the highest accuracy (~88%)
- Logistic Regression struggled with intermediate cases (needs_attention)
- Ensemble models performed better due to their ability to capture complex patterns

The “critical” class was easiest to identify, while “needs_attention” was the most difficult due to overlapping characteristics.

### Important Consideration:

False negatives (failing to identify a critical student) are the most dangerous errors, as they prevent necessary intervention.

---

## 7. Feature Importance & Explainability

Feature importance analysis showed that the most influential variables include:

- attendance_pct
- quiz_avg_score
- technical_rating
- simulator_score_avg
- missed_sessions_count
- days_since_last_activity

These features represent engagement, performance, and behavior, which are key drivers of student success.

The results align with domain intuition, increasing confidence in the model’s reliability.

Potential leakage-prone features such as progress_pct and prior_intervention_flag were used carefully to avoid bias.

---

## 8. Product Insight Layer

A product-oriented output layer was designed to translate model predictions into actionable insights.

For each student, the system generates:

- Risk label (on_track / needs_attention / critical)
- Risk score (numeric)
- Top contributing factors
- Recommendation

### Example Output:

```json
{
  "student_id": "STD_042",
  "risk_label": "needs_attention",
  "risk_score": 71,
  "top_factors": [
    "Low attendance",
    "High missed sessions"
  ],
  "recommendation": "Monitor closely and provide targeted support"
}