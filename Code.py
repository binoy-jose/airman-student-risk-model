

**Creating a synthetic data set**

import pandas as pd
import numpy as np

np.random.seed(42)
n = 800

# -----------------------------
# 1. BASIC IDENTIFIERS
# -----------------------------
data = pd.DataFrame({
    "student_id": [f"STD_{i}" for i in range(n)],
    "academy_id": np.random.choice(["A1", "A2", "A3", "A4"], n, p=[0.3, 0.3, 0.2, 0.2]),
    "course_type": np.random.choice(["Arts", "Science", "Technical"], n),
    "batch": np.random.choice(["B1", "B2", "B3"], n)
})

# -----------------------------
# 2. BEHAVIORAL FEATURES
# -----------------------------
data["attendance_pct"] = np.clip(np.random.normal(75, 15, n), 40, 100)
data["missed_sessions_count"] = np.clip((100 - data["attendance_pct"]) / 10 + np.random.randint(0, 3, n), 0, 10)
data["late_submission_count"] = np.random.randint(0, 8, n)

# Engagement proxy
data["days_since_last_activity"] = np.clip(
    np.random.normal(5, 5, n) + data["missed_sessions_count"], 0, 30
)

# -----------------------------
# 3. PERFORMANCE FEATURES
# -----------------------------
data["quiz_attempt_count"] = np.random.randint(1, 15, n)

data["quiz_avg_score"] = np.clip(
    data["attendance_pct"] - np.random.normal(10, 10, n),
    30, 100
)

data["simulator_score_avg"] = np.clip(
    data["quiz_avg_score"] + np.random.normal(0, 8, n),
    30, 100
)

# Ratings (scaled 1–5)
data["technical_rating"] = np.clip(
    data["quiz_avg_score"] / 20 + np.random.normal(0, 0.5, n),
    1, 5
)

data["non_technical_rating"] = np.clip(
    data["attendance_pct"] / 20 + np.random.normal(0, 0.5, n),
    1, 5
)

data["instructor_eval_avg"] = (
    data["technical_rating"] + data["non_technical_rating"]
) / 2 + np.random.normal(0, 0.3, n)

data["instructor_eval_avg"] = np.clip(data["instructor_eval_avg"], 1, 5)

# -----------------------------
# 4. ACTIVITY / PROGRESSION
# -----------------------------
data["flight_or_sim_hours"] = np.clip(
    np.random.normal(50, 20, n) + data["attendance_pct"] / 2,
    10, 150
)

data["progress_pct"] = np.clip(
    data["attendance_pct"] * 0.6 + data["quiz_avg_score"] * 0.4 + np.random.normal(0, 5, n),
    0, 100
)

# -----------------------------
# 5. OPERATIONAL FEATURES
# -----------------------------
data["schedule_changes_count"] = np.random.randint(0, 6, n)

# Prior intervention (probabilistic)
data["prior_intervention_flag"] = (
    (data["attendance_pct"] < 60) |
    (data["quiz_avg_score"] < 50)
).astype(int)

# -----------------------------
# 6. TARGET CREATION (MULTI-FACTOR RISK)
# -----------------------------
def assign_risk(row):
    score = 0

    # Strong signals
    if row["attendance_pct"] < 65:
        score += 2
    if row["quiz_avg_score"] < 60:
        score += 2
    if row["simulator_score_avg"] < 55:
        score += 1

    # Behavior
    if row["missed_sessions_count"] > 5:
        score += 2
    if row["late_submission_count"] > 4:
        score += 1

    # Engagement
    if row["days_since_last_activity"] > 10:
        score += 1

    # Ratings
    if row["technical_rating"] < 2.5:
        score += 2
    if row["instructor_eval_avg"] < 3:
        score += 1

    # Intervention history
    if row["prior_intervention_flag"] == 1:
        score += 1

    # Final classification
    if score >= 7:
        return "critical"
    elif score >= 4:
        return "needs_attention"
    else:
        return "on_track"

data["final_outcome"] = data.apply(assign_risk, axis=1)

# -----------------------------
# 7. ADD SOME REAL-WORLD NOISE
# -----------------------------
noise_idx = np.random.choice(n, size=int(0.05 * n), replace=False)

# flip some labels to simulate imperfect data
data.loc[noise_idx, "final_outcome"] = np.random.choice(
    ["on_track", "needs_attention", "critical"], len(noise_idx)
)

# -----------------------------
# 8. SAVE DATA
# -----------------------------
import os
os.makedirs("data", exist_ok=True)

data.to_csv("data/synthetic_airman_data.csv", index=False)

print("Dataset created successfully!")
print(data.head())
print("\nClass distribution:")
print(data["final_outcome"].value_counts())

# Downloaded the file for furthur calculations
from google.colab import files
files.download("data/synthetic_airman_data.csv")

"""# **Exploratory Data Analysis (EDA)**"""

data.info()

data.describe()

data.isnull().sum()


# Distribution of Student Outcomes


import matplotlib.pyplot as plt

data['final_outcome'].value_counts().plot(kind='bar')
plt.title("Distribution of Student Outcomes")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Attendance vs Outcome

import seaborn as sns

sns.boxplot(x='final_outcome', y='attendance_pct', data=data)
plt.title("Attendance vs Outcome")
plt.show()


# Days Since Last Activity vs Outcome

sns.boxplot(x='final_outcome', y='days_since_last_activity', data=data)
plt.title("Days Since Last Activity vs Outcome")
plt.show()


# Feature Correlation Heatmap


plt.figure(figsize=(10,8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# Attendance Distribution

data['attendance_pct'].hist(bins=20)
plt.title("Attendance Distribution")
plt.xlabel("Attendance %")
plt.ylabel("Frequency")
plt.show()



# Outcome Distribution by Academy

pd.crosstab(data['academy_id'], data['final_outcome']).plot(kind='bar', stacked=True)
plt.title("Outcome Distribution by Academy")
plt.xlabel("Academy")
plt.ylabel("Count")
plt.show()



# **Model Building**

#preparing data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("data/synthetic_airman_data.csv")

# Encode target
le = LabelEncoder()
data["final_outcome_encoded"] = le.fit_transform(data["final_outcome"])

# Drop non-useful columns
X = data.drop(columns=["student_id", "final_outcome", "final_outcome_encoded"])
y = data["final_outcome_encoded"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""**Baseline Model - Logistic Regression**"""

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_val)

"""**Random Forest**"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_val)

"""**Gradiesnt Boosting**"""

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_val)

"""**XGBoost**"""

from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_val)

"""# **Evaluation**"""

from sklearn.metrics import classification_report, confusion_matrix

print("Logistic Regression:\n")
print(classification_report(y_val, y_pred_lr))

print("Random Forest:\n")
print(classification_report(y_val, y_pred_rf))

print("Gradient Boosting:\n")
print(classification_report(y_val, y_pred_gb))

print("XG Boost:\n")
print(classification_report(y_val, y_pred_xgb))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion matrices
cm_rf = confusion_matrix(y_val, y_pred_rf)
cm_xgb = confusion_matrix(y_val, y_pred_xgb)

labels = ["on_track", "needs_attention", "critical"]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',  xticklabels=labels,yticklabels=labels,ax=axes[0])
axes[0].set_title("Random Forest")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# XGBoost
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens',  xticklabels=labels,yticklabels=labels, ax=axes[1])
axes[1].set_title("XGBoost")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

#**Feature Imporatnace**


#Importanat features from Random Forest
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importance
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)

# Sort and plot
feature_importance.sort_values(ascending=True).plot(kind='barh', figsize=(10,6))
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.show()



# **AIRMAN Product Insight Layer**


# Creating a scoring function
def calculate_risk_score(row):
    score = 0

    # Attendance
    score += (100 - row['attendance_pct']) * 0.4

    # Performance
    score += (70 - row['quiz_avg_score']) * 0.3 if row['quiz_avg_score'] < 70 else 0

    # Behavior
    score += row['missed_sessions_count'] * 3

    # Engagement
    score += row['days_since_last_activity'] * 1.5

    # Ratings
    if row['technical_rating'] < 3:
        score += 10

    return int(score)

# Assigning risk label
def assign_label(score):
    if score >= 70:
        return "critical"
    elif score >= 40:
        return "needs_attention"
    else:
        return "on_track"

# Defining top factors
def get_top_factors(row):
    factors = []

    if row['attendance_pct'] < 70:
        factors.append("Low attendance")

    if row['quiz_avg_score'] < 60:
        factors.append("Low quiz performance")

    if row['missed_sessions_count'] > 5:
        factors.append("High missed sessions")

    if row['days_since_last_activity'] > 10:
        factors.append("Low recent activity")

    if row['technical_rating'] < 3:
        factors.append("Low technical skills")

    return factors[:3]

# Recomendation function
def generate_recommendation(label, factors):

    if label == "critical":
        return "Immediate instructor intervention required. Schedule review session and assign targeted training."

    elif label == "needs_attention":
        return "Monitor closely and provide additional practice materials. Encourage regular participation."

    else:
        return "Student is performing well. Continue current training plan."

# Generating Final Output
outputs = []

for _, row in data.iterrows():

    risk_score = calculate_risk_score(row)
    risk_label = assign_label(risk_score)
    factors = get_top_factors(row)
    recommendation = generate_recommendation(risk_label, factors)

    outputs.append({
        "student_id": row["student_id"],
        "risk_label": risk_label,
        "risk_score": risk_score,
        "top_factors": factors,
        "recommendation": recommendation
    })

import os
import json
import pandas as pd
from google.colab import files

# 1. Create outputs folder
os.makedirs("outputs", exist_ok=True)

# 2. Convert to DataFrame
output_df = pd.DataFrame(outputs)

# 3. Fix list column for CSV
output_df["top_factors"] = output_df["top_factors"].apply(lambda x: ", ".join(x))

# 4. Save JSON
with open("outputs/risk_predictions.json", "w") as f:
    json.dump(outputs, f, indent=2)

# 5. Save CSV
output_df.to_csv("outputs/risk_predictions.csv", index=False)

print("Files saved successfully!")

# 6. Download files
files.download("outputs/risk_predictions.csv")
files.download("outputs/risk_predictions.json")



