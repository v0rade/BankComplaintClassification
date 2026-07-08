import os
import re
import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ==========================================================
# PATH PROJECT
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "complaint.csv"

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("MEMBACA DATASET...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(df.head())

print("\nJumlah Data :", len(df))

print("\nKolom Dataset :")
print(df.columns)


# ==========================================================
# PILIH KOLOM
# ==========================================================

df = df[["narrative", "product"]]

df.columns = ["text", "label"]

df.dropna(inplace=True)

print("\nJumlah data setelah drop NA :", len(df))


# ==========================================================
# CLEANING
# ==========================================================

def clean(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning Text...")

df["clean"] = df["text"].apply(clean)


# ==========================================================
# TF-IDF
# ==========================================================

print("\nVectorization TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean"])

y = df["label"]


# ==========================================================
# SPLIT DATA
# ==========================================================

print("\nSplit Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Train :", X_train.shape)

print("Test  :", X_test.shape)


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
)

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, pred_lr)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

print("Accuracy :", acc_lr)

print()

print(classification_report(y_test, pred_lr))


# ==========================================================
# NAIVE BAYES
# ==========================================================

print("\nTraining Naive Bayes...")

nb = MultinomialNB()

nb.fit(X_train, y_train)

pred_nb = nb.predict(X_test)

acc_nb = accuracy_score(y_test, pred_nb)

print("\n" + "=" * 60)
print("NAIVE BAYES")
print("=" * 60)

print("Accuracy :", acc_nb)

print()

print(classification_report(y_test, pred_nb))


# ==========================================================
# PERBANDINGAN MODEL
# ==========================================================

print("\n" + "=" * 60)
print("PERBANDINGAN MODEL")
print("=" * 60)

print(f"Logistic Regression : {acc_lr:.4f}")

print(f"Naive Bayes         : {acc_nb:.4f}")

if acc_lr > acc_nb:
    print("\nModel Terbaik : Logistic Regression")
    best_model = lr
else:
    print("\nModel Terbaik : Naive Bayes")
    best_model = nb


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\nMembuat Confusion Matrix...")

cm = confusion_matrix(
    y_test,
    best_model.predict(X_test),
    labels=best_model.classes_,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_,
)

fig, ax = plt.subplots(figsize=(12, 12))

disp.plot(ax=ax, xticks_rotation=90)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(MODEL_DIR / "confusion_matrix.png")

plt.close()


# ==========================================================
# SAVE MODEL
# ==========================================================

print("\nMenyimpan Model...")

with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model berhasil disimpan.")

print("Lokasi :", MODEL_DIR)


print("\nSELESAI")