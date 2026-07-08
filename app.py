import streamlit as st
import pickle
import re
import pandas as pd
from pathlib import Path

# ======================================================
# PATH
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Bank Complaint Classification",
    page_icon="🏦",
    layout="wide"
)

# ======================================================
# LOAD MODEL
# ======================================================

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    st.error(f"Model gagal dimuat : {e}")
    st.stop()

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🏦 AI Project")

st.sidebar.markdown("""
## Bank Complaint Classification

### AI Methods

- TF-IDF
- Logistic Regression
- Naive Bayes (Comparison)

---

Universitas Cakrawala

Artificial Intelligence

2026
""")

# ======================================================
# CLEANING
# ======================================================

def clean(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# ======================================================
# TITLE
# ======================================================

st.title("🏦 Bank Complaint Classification")

st.write("""
Aplikasi ini mengklasifikasikan keluhan pelanggan
perbankan menggunakan algoritma TF-IDF dan Logistic Regression.
""")

st.divider()

# ======================================================
# SESSION HISTORY
# ======================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# INPUT
# ======================================================

user_input = st.text_area(
    "Masukkan Complaint",
    height=180,
    placeholder="Example : My credit card was charged twice..."
)

# ======================================================
# BUTTON
# ======================================================

if st.button("🔍 Predict", use_container_width=True):

    if user_input.strip() == "":

        st.warning("Masukkan complaint terlebih dahulu.")

    else:

        with st.spinner("Sedang memproses..."):

            cleaned = clean(user_input)

            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)[0]

            probability = model.predict_proba(vector)[0]

            confidence = probability.max()*100

            classes = model.classes_

            result = pd.DataFrame({
                "Category": classes,
                "Probability": probability
            })

            result = result.sort_values(
                by="Probability",
                ascending=False
            )

            st.success("Prediksi Berhasil")

            col1,col2 = st.columns(2)

            with col1:

                st.subheader("Prediction")

                st.info(prediction)

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col2:

                st.subheader("Top Prediction")

                chart = result.head(5)

                chart = chart.set_index("Category")

                st.bar_chart(chart)

            st.subheader("Top 5 Categories")

            st.dataframe(
                result.head(5),
                use_container_width=True
            )

            st.subheader("Clean Text")

            st.code(cleaned)

            st.session_state.history.append({

                "Complaint": user_input,

                "Prediction": prediction,

                "Confidence": round(confidence,2)

            })

# ======================================================
# HISTORY
# ======================================================

if len(st.session_state.history)>0:

    st.divider()

    st.subheader("Prediction History")

    history = pd.DataFrame(st.session_state.history)

    st.dataframe(history,use_container_width=True)

    csv = history.to_csv(index=False)

    st.download_button(

        "📥 Download History",

        csv,

        "prediction_history.csv",

        "text/csv"

    )

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption(
    "Artificial Intelligence Final Project | Universitas Cakrawala | 2026"
)