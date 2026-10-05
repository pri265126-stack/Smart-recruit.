import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv("train.csv")
validation_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

print("Train      :", train_df.shape)
print("Validation :", validation_df.shape)
print("Test       :", test_df.shape)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

FEATURES = [
    "face_presence_percent",
    "maximum_faces_detected",
    "longest_face_absence_sec",
    "multiple_face_duration_sec",
    "looking_away_events",
    "looking_center_percent",
    "looking_left_percent",
    "looking_right_percent"
]

TARGET = "activity_label"


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# 3. LABEL ENCODING FOR XGBOOST
# ============================================================

print("\nEncoding labels for XGBoost...")

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_validation_encoded = label_encoder.transform(y_validation)
y_test_encoded = label_encoder.transform(y_test)

print("\nLabel mapping:")

for number, label in enumerate(label_encoder.classes_):
    print(f"{number} -> {label}")


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ]),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 5. TRAIN MODELS
# ============================================================

results = []

trained_models = {}

print("\n")
print("=" * 70)
print("TRAINING MODELS")
print("=" * 70)


for name, model in models.items():

    print(f"\nTraining: {name}")

    # --------------------------------------------------------
    # XGBoost needs encoded labels
    # --------------------------------------------------------

    if name == "XGBoost":

        model.fit(
            X_train,
            y_train_encoded
        )

    else:

        model.fit(
            X_train,
            y_train
        )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    val_pred = model.predict(X_validation)

    # Convert XGBoost numeric predictions
    # back to original labels

    if name == "XGBoost":

        val_pred = label_encoder.inverse_transform(
            val_pred.astype(int)
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        val_pred
    )

    precision = precision_score(
        y_validation,
        val_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        val_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_validation,
        val_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_validation,
        val_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

    trained_models[name] = model

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

# Save comparison

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nSaved: model_comparison.csv")


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best Model: {best_model_name}"
)

print(
    "Selection metric: Macro F1"
)


# ============================================================
# 8. FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


# XGBoost uses encoded test labels internally,
# but predictions are converted back to strings.

test_pred = best_model.predict(
    X_test
)

if best_model_name == "XGBoost":

    test_pred = label_encoder.inverse_transform(
        test_pred.astype(int)
    )


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_precision = precision_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)

test_weighted_f1 = f1_score(
    y_test,
    test_pred,
    average="weighted",
    zero_division=0
)


print(f"\nAccuracy    : {test_accuracy:.4f}")
print(f"Precision   : {test_precision:.4f}")
print(f"Recall      : {test_recall:.4f}")
print(f"Macro F1    : {test_f1:.4f}")
print(f"Weighted F1 : {test_weighted_f1:.4f}")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    test_pred,
    zero_division=0
)

print(report)


# Save classification report

with open(
    "classification_report.txt",
    "w"
) as file:

    file.write(report)

print(
    "Saved: classification_report.txt"
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


labels = label_encoder.classes_

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)


# Save confusion matrix

cm_df.to_csv(
    "confusion_matrix.csv"
)

print(
    "\nSaved: confusion_matrix.csv"
)


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "best_face_activity_model.pkl"
)

print("\n")
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    "File: best_face_activity_model.pkl"
)


# ============================================================
# 12. SAVE LABEL ENCODER
# ============================================================

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

print(
    "File: label_encoder.pkl"
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest Model      : {best_model_name}"
)

print(
    f"Test Accuracy   : {test_accuracy:.4f}"
)

print(
    f"Test Macro F1   : {test_f1:.4f}"
)

print("\nGenerated files:")

print("1. model_comparison.csv")
print("2. confusion_matrix.csv")
print("3. classification_report.txt")
print("4. best_face_activity_model.pkl")
print("5. label_encoder.pkl")

print("\nDone!")