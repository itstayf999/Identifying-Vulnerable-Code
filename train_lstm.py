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

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "TensorFlow is not installed in this Python environment. "
        "Install it first using: pip install tensorflow"
    )

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

warnings.filterwarnings("ignore")

# =========================================================
# 1) CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "Binary_LSTM_RAW_v2.csv")
OUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE_FROM_TRAIN = 0.2

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LEN = 300
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10
LSTM_UNITS = 64

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================================================
# 2) HELPERS
# =========================================================
def clean_code_text(code):
    if pd.isna(code):
        return ""
    code = str(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = code.replace("\t", "    ")
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

def plot_history(history, out_path):
    hist = history.history
    plt.figure(figsize=(7, 4))
    plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
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

# Keep only valid binary values
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
# 5) TOKENIZATION
# =========================================================
print("\n" + "=" * 60)
print("2) Tokenization and Padding...")
print("=" * 60)

tokenizer = Tokenizer(
    num_words=MAX_NUM_WORDS,
    oov_token="<OOV>",
    filters="",
    lower=False
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_SEQUENCE_LEN,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_SEQUENCE_LEN,
    padding="post",
    truncating="post"
)

vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)

print("Vocabulary size:", vocab_size)
print("Train shape:", X_train_pad.shape)
print("Test shape :", X_test_pad.shape)

with open(os.path.join(OUT_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# =========================================================
# 6) BUILD MODEL
# =========================================================
print("\nBuilding LSTM model...")

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN),
    SpatialDropout1D(0.2),
    LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.0),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

best_model_path = os.path.join(OUT_DIR, "best_lstm_model.keras")

callbacks = [
    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]

# =========================================================
# 7) TRAIN MODEL
# =========================================================
print("\nTraining LSTM...")

train_start = time.perf_counter()
history = model.fit(
    X_train_pad,
    y_train,
    validation_split=VAL_SIZE_FROM_TRAIN,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    shuffle=True,
    callbacks=callbacks
)
train_end = time.perf_counter()
train_time = train_end - train_start

if os.path.exists(best_model_path):
    model = load_model(best_model_path)

# =========================================================
# 8) TEST MODEL
# =========================================================
test_start = time.perf_counter()
y_prob = model.predict(X_test_pad, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
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
print("LSTM Results")
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
# 9) SAVE OUTPUTS
# =========================================================
metrics = {
    "model_name": "LSTM Balanced Fixed",
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "roc_auc": float(auc),
    "train_time_seconds": float(train_time),
    "test_time_seconds": float(test_time),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "vocab_size": int(vocab_size),
    "max_sequence_len": int(MAX_SEQUENCE_LEN)
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
    "LSTM Balanced Fixed - Confusion Matrix",
    os.path.join(OUT_DIR, "confusion_matrix.png")
)

plot_history(history, os.path.join(OUT_DIR, "training_loss.png"))

print("\nSaved all outputs in:", OUT_DIR)
print("DONE")
