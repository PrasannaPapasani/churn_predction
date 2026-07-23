"""
Telecom Customer Churn Prediction
==================================
Dataset : IBM/Kaggle "Telco Customer Churn" (blastchar), 7,043 customers x 21 columns
Source  : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
          (mirrored here from https://github.com/IBM/telco-customer-churn-on-icp4d)

This script:
1. Loads and cleans the data
2. Does light EDA and saves plots
3. Engineers features and preprocesses (encoding + scaling)
4. Trains & compares 4 models (Logistic Regression, Random Forest,
   Gradient Boosting, XGBoost) with class-imbalance handling
5. Evaluates on a held-out test set (accuracy, precision, recall, F1, ROC-AUC)
6. Picks the best model by ROC-AUC, tunes it briefly, and saves it to disk
7. Prints top drivers of churn (feature importance)

Run:
    python churn_prediction.py

To use your OWN data instead of the sample Kaggle dataset, just replace
DATA_PATH below with the path to your CSV, as long as it has a "Churn"
column (Yes/No) and a similar customer-level structure. The pipeline
will adapt automatically to whatever columns are present.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
DATA_PATH = "telco_churn.csv"      # swap this for your own CSV
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.head(3))

# ---------------------------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Cleaning data")
print("=" * 70)

# TotalCharges is loaded as object because of blank strings for new customers
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isna().sum()
print(f"Rows with missing/blank TotalCharges: {n_missing} -> filled with 0 (tenure=0 customers)")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop the ID column, it carries no predictive signal
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Target column -> binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(f"Churn rate: {df['Churn'].mean():.2%}")

# ---------------------------------------------------------------------------
# 3. LIGHT EDA (saved as PNGs, not shown interactively)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: EDA -> saving plots to ./outputs/")
print("=" * 70)

sns.set_style("whitegrid")

# Churn distribution
plt.figure(figsize=(5, 4))
sns.countplot(x="Churn", data=df, palette=["#2ecc71", "#e74c3c"])
plt.title("Churn Distribution (0=Stayed, 1=Churned)")
plt.savefig(f"{OUTPUT_DIR}/01_churn_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# Churn by contract type
plt.figure(figsize=(6, 4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.savefig(f"{OUTPUT_DIR}/02_churn_by_contract.png", dpi=120, bbox_inches="tight")
plt.close()

# Tenure distribution by churn
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, common_norm=False)
plt.title("Tenure Distribution by Churn")
plt.savefig(f"{OUTPUT_DIR}/03_tenure_by_churn.png", dpi=120, bbox_inches="tight")
plt.close()

# Monthly charges by churn
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False)
plt.title("Monthly Charges Distribution by Churn")
plt.savefig(f"{OUTPUT_DIR}/04_monthlycharges_by_churn.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"Saved 4 EDA plots to {OUTPUT_DIR}/")

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Feature engineering")
print("=" * 70)

# A couple of engineered features that typically help churn models
df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
df["TenureGroup"] = pd.cut(
    df["tenure"], bins=[-1, 12, 24, 48, 60, 100],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"]
)

X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")

# ---------------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 6. PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
    ]
)

# ---------------------------------------------------------------------------
# 7. MODELS TO COMPARE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Training & comparing models")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
if HAS_XGB:
    # scale_pos_weight handles the ~73/27 class imbalance
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1
    )

results = []
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    results.append(metrics)
    fitted_pipelines[name] = pipe
    print(f"\n{name}:")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:10s}: {v:.4f}")

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n" + "-" * 70)
print("MODEL COMPARISON (sorted by ROC-AUC)")
print("-" * 70)
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

# ROC curves for all models
plt.figure(figsize=(7, 6))
for name, pipe in fitted_pipelines.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/05_roc_curves.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 8. PICK BEST MODEL, TUNE IT, EVALUATE IN DETAIL
# ---------------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
print(f"\n{'=' * 70}")
print(f"STEP 6: Best model = {best_model_name} -> quick hyperparameter tuning")
print("=" * 70)

if best_model_name == "Random Forest":
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [8, 12, None],
        "classifier__min_samples_leaf": [1, 3],
    }
elif best_model_name == "XGBoost" and HAS_XGB:
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [3, 4, 6],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
    }
elif best_model_name == "Gradient Boosting":
    param_grid = {
        "classifier__n_estimators": [150, 300],
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.05, 0.1],
    }
else:  # Logistic Regression
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
    }

base_pipe = fitted_pipelines[best_model_name]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(base_pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

best_pipe = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

y_pred = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:, 1]

print("\nFinal tuned model performance on test set:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"  F1       : {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix - {best_model_name} (tuned)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(f"{OUTPUT_DIR}/06_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 9. FEATURE IMPORTANCE / DRIVERS OF CHURN
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Top drivers of churn")
print("=" * 70)

feat_names = (
    numeric_cols
    + list(best_pipe.named_steps["preprocessor"]
           .named_transformers_["cat"]
           .get_feature_names_out(categorical_cols))
)
clf_step = best_pipe.named_steps["classifier"]

if hasattr(clf_step, "feature_importances_"):
    importances = clf_step.feature_importances_
elif hasattr(clf_step, "coef_"):
    importances = np.abs(clf_step.coef_[0])
else:
    importances = None

if importances is not None:
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)
    print(fi_df.to_string(index=False))

    plt.figure(figsize=(7, 6))
    sns.barplot(x="importance", y="feature", data=fi_df, palette="viridis")
    plt.title(f"Top 15 Churn Drivers - {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    fi_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

# ---------------------------------------------------------------------------
# 10. SAVE FINAL MODEL
# ---------------------------------------------------------------------------
model_path = f"{OUTPUT_DIR}/churn_model.pkl"
joblib.dump(best_pipe, model_path)
print(f"\nFinal model pipeline (preprocessing + classifier) saved to: {model_path}")

print("\n" + "=" * 70)
print("DONE. All outputs saved in ./outputs/")
print("=" * 70)


# ---------------------------------------------------------------------------
# EXAMPLE: how to load the model later and score a NEW customer
# ---------------------------------------------------------------------------
def predict_new_customer(customer_dict, model_path=model_path):
    """
    customer_dict: a dict with the same raw columns as the training data
                   (before the customerID/Churn columns), e.g.
                   {"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", ...}
    Returns churn probability (0-1).
    """
    pipe = joblib.load(model_path)
    new_df = pd.DataFrame([customer_dict])
    new_df["AvgMonthlySpend"] = new_df["TotalCharges"] / new_df["tenure"].replace(0, 1)
    new_df["TenureGroup"] = pd.cut(
        new_df["tenure"], bins=[-1, 12, 24, 48, 60, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"]
    )
    prob = pipe.predict_proba(new_df)[0, 1]
    return prob
