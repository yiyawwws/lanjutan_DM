import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ============================
# LOAD MODELS & VECTORIZER
# ============================

with open("models/model_bernoulli_nb.pkl", "rb") as f:
    model_nb = pickle.load(f)

with open("models/model_linear_svm.pkl", "rb") as f:
    model_svm = pickle.load(f)

with open("models/model_ensemble_voting.pkl", "rb") as f:
    model_vote = pickle.load(f)

with open("preprocessing/vectorizer_tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ============================
# STREAMLIT UI
# ============================

st.title("Breast Cancer Prediction App")
st.write("Masukkan data untuk memprediksi diagnosis kanker payudara.")

# Input form
input_text = st.text_area(
    "Masukkan teks / data yang sudah dipreprocessing:",
    placeholder="Example: radius_mean=17.99, texture_mean=10.38, perimeter_mean=122.8 ..."
)

model_choice = st.selectbox(
    "Pilih Model Prediksi:",
    ["Naive Bayes", "SVM", "Voting Model"]
)

if st.button("Prediksi"):
    try:
        data_vectorized = vectorizer.transform([input_text])

        if model_choice == "Naive Bayes":
            pred = model_nb.predict(data_vectorized)[0]
        elif model_choice == "SVM":
            pred = model_svm.predict(data_vectorized)[0]
        else:
            pred = model_vote.predict(data_vectorized)[0]

        hasil = "Malignant (GANAS)" if pred == 1 else "Benign (TIDAK ganas)"

        st.success(f"Hasil Prediksi : **{hasil}**")

    except Exception as e:
        st.error(f"Error: {e}")
