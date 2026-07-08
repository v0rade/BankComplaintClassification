import pickle
import re
from pathlib import Path

# ==========================================================
# PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"

VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("Model berhasil dimuat.")

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


# ==========================================================
# PREDICT FUNCTION
# ==========================================================

def predict(text):

    cleaned = clean(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max() * 100

    return prediction, confidence


# ==========================================================
# MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BANK COMPLAINT CLASSIFICATION")
    print("=" * 60)

    while True:

        complaint = input("\nMasukkan Complaint : ")

        if complaint.strip() == "":
            print("Complaint tidak boleh kosong.")
            continue

        label, score = predict(complaint)

        print("\n==============================")
        print("HASIL PREDIKSI")
        print("==============================")
        print("Kategori   :", label)
        print(f"Confidence : {score:.2f}%")

        lagi = input("\nPrediksi lagi? (y/n) : ").lower()

        if lagi != "y":
            print("\nTerima kasih.")
            break