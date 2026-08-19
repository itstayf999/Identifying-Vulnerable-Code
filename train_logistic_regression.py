# -*- coding: utf-8 -*-

import os
import re
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =========================================================
# 1) CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "Binary_LSTM_RAW_v2.csv")
OUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_SEED = 42

# =========================================================
# 2) HELPERS
# =========================================================
def clean_code_text(code):
    if pd.isna(code):
        return ""
    code = str(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = code.replace("\t", "    ")
    code = re.sub(r"[ \f\v]+", " ", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    code = code.strip()
    if len(code) > 10000:
        code = code[:10000]
    return code

def plot_conf_matrix(cm, title, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    classes = ["Secure (0)", "Vulnerable (1)"]
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=20)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# =========================================================
# 3) LOAD DATA
# =========================================================
print("=" * 60)
print("1) Loading data...")
print("=" * 60)
print("CSV_PATH:", CSV_PATH)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

df = pd.read_csv(
    CSV_PATH,
    sep=",",
    engine="python",
    on_bad_lines="skip",
    encoding="latin1"
)


# Remove extra columns created during parsing
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()

print("Loaded successfully.")
print("Columns found:", list(df.columns))

if "code" not in df.columns:
    raise ValueError("CSV must contain 'code' column.")

if "label" in df.columns:
    label_col = "label"
elif "label_int" in df.columns:
    label_col = "label_int"
else:
    raise ValueError("CSV must contain 'label' or 'label_int' column.")

df = df.dropna(subset=["code", label_col]).copy()
df["code_clean"] = df["code"].apply(clean_code_text)
df = df[df["code_clean"].str.len() > 0].copy()

df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
df = df.dropna(subset=[label_col]).copy()
df[label_col] = df[label_col].astype(int)

# Keep only valid binary labels
df = df[df[label_col].isin([0, 1])].copy()

if df.empty:
    raise ValueError("No valid binary rows left after cleaning.")

if df[label_col].nunique() < 2:
    raise ValueError("Need both classes 0 and 1 after cleaning.")

print("Dataset shape after cleaning:", df.shape)
print("\nBinary label distribution:")
print(df[label_col].value_counts())

X = df["code_clean"].values
y = df[label_col].values

# =========================================================
# 4) SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size :", len(X_test))

# =========================================================
# 5) TF-IDF
# =========================================================
print("\n" + "=" * 60)
print("2) TF-IDF Vectorization...")
print("=" * 60)

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=30000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

with open(os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("Train vectorized shape:", X_train_vec.shape)
print("Test vectorized shape :", X_test_vec.shape)

# =========================================================
# 6) TRAIN MODEL
# =========================================================
print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_SEED
)

train_start = time.perf_counter()
model.fit(X_train_vec, y_train)
train_end = time.perf_counter()
train_time = train_end - train_start

with open(os.path.join(OUT_DIR, "logistic_regression_model.pkl"), "wb") as f:
    pickle.dump(model, f)

# =========================================================
# 7) TEST MODEL
# =========================================================
test_start = time.perf_counter()
y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]
test_end = time.perf_counter()
test_time = test_end - test_start

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4, zero_division=0)

print("\n" + "=" * 60)
print("Logistic Regression Results")
print("=" * 60)
print("Accuracy     :", acc)
print("Precision    :", prec)
print("Recall       :", rec)
print("F1-score     :", f1)
print("ROC-AUC      :", auc)
print("Train Time   :", train_time, "seconds")
print("Test Time    :", test_time, "seconds")
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", cm)

# =========================================================
# 8) SAVE OUTPUTS
# =========================================================
metrics = {
    "model_name": "Logistic Regression",
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "roc_auc": float(auc),
    "train_time_seconds": float(train_time),
    "test_time_seconds": float(test_time),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test))
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

pred_df = pd.DataFrame({
    "code": X_test,
    "true_label": y_test,
    "pred_label": y_pred,
    "prob_vulnerable": y_prob
})
pred_df.to_csv(
    os.path.join(OUT_DIR, "test_predictions.csv"),
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

plot_conf_matrix(
    cm,
    "Logistic Regression - Confusion Matrix",
    os.path.join(OUT_DIR, "confusion_matrix.png")
)

print("\nSaved all outputs in:", OUT_DIR)
print("DONE")
