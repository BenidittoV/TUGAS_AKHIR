import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "best_model"
MAX_LENGTH = 128

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_review(text, threshold):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()[0]

    probs = sigmoid(logits)

    results = []
    for i, prob in enumerate(probs):
        label = model.config.id2label[i]
        results.append({
            "Label": label,
            "Probability": round(float(prob), 4),
            "Predicted": "Ya" if prob >= threshold else "Tidak"
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Probability", ascending=False)
    return result_df

st.set_page_config(
    page_title="Klasifikasi Multi-Label Ulasan Produk",
    page_icon="🛒",
    layout="wide"
)

st.title("Klasifikasi Multi-Label Ulasan Produk E-Commerce")

st.write(
    "Prototype ini menggunakan model terbaik hasil fine-tuning: "
    "IndoBERT dengan metode Full Fine-Tuning Sigmoid-BCE."
)

review = st.text_area(
    "Masukkan ulasan produk:",
    placeholder="Contoh: Barang bagus, murah, packing aman, tapi pengiriman lama.",
    height=120
)

threshold = st.slider(
    "Threshold prediksi",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if st.button("Prediksi"):
    if not review.strip():
        st.warning("Masukkan teks ulasan terlebih dahulu.")
    else:
        result_df = predict_review(review, threshold)
        predicted_df = result_df[result_df["Predicted"] == "Ya"]

        st.subheader("Label Terdeteksi")
        if len(predicted_df) > 0:
            st.dataframe(predicted_df, use_container_width=True)
        else:
            st.warning("Tidak ada label yang melewati threshold. Coba turunkan threshold.")

        st.subheader("Semua Probabilitas Label")
        st.dataframe(result_df, use_container_width=True)
