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


# =========================================================
# KONFIGURASI DASAR
# =========================================================
MODEL_DIR = "best_model"
MAX_LENGTH = 128


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# =========================================================
# FUNGSI BANTUAN
# =========================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def format_label(label):
    """
    Mengubah nama label agar lebih nyaman dibaca di tampilan.
    Contoh:
    harga_negatif -> Harga Negatif
    kualitas_positif -> Kualitas Positif
    """
    return str(label).replace("_", " ").title()


def predict_review(text, threshold):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.detach().cpu().numpy()[0]

    probs = sigmoid(logits)

    results = []
    for i, prob in enumerate(probs):
        label = model.config.id2label.get(i, f"LABEL_{i}")
        is_predicted = prob >= threshold

        results.append({
            "Label": format_label(label),
            "Label Asli": label,
            "Probabilitas": round(float(prob), 4),
            "Hasil": "Terdeteksi" if is_predicted else "Tidak Terdeteksi"
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Probabilitas", ascending=False).reset_index(drop=True)

    return result_df


# =========================================================
# SETUP HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Ulasan Produk",
    page_icon="🛒",
    layout="wide"
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Pengaturan")

    threshold = st.slider(
        "Threshold prediksi",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Threshold digunakan sebagai batas minimal probabilitas agar suatu label dianggap terdeteksi. "
            "Nilai 0.50 digunakan sebagai batas umum, sedangkan perubahan threshold dapat digunakan "
            "untuk melihat sensitivitas model."
        )
    )

    st.divider()

    st.caption("Model")
    st.write("IndoBERT")
    st.caption("Metode")
    st.write("Full Fine-Tuning")
    st.caption("Tipe Prediksi")
    st.write("Multi-label classification")

    st.divider()

    st.info(
        "Pada penelitian, threshold final sebaiknya ditentukan dari validation set. "
        "Slider ini digunakan untuk simulasi dan analisis sensitivitas prediksi."
    )


# =========================================================
# HEADER
# =========================================================
st.title("Analisis Multi-Label Ulasan Produk E-Commerce")

st.write(
    "Aplikasi ini digunakan untuk menguji model klasifikasi ulasan produk. "
    "Satu ulasan dapat memiliki lebih dari satu label, misalnya terkait kualitas, harga, "
    "pengiriman, kemasan, atau pelayanan penjual."
)


# =========================================================
# INPUT AREA
# =========================================================
st.subheader("Masukkan Ulasan")

review = st.text_area(
    label="Teks ulasan",
    placeholder="Contoh: Barangnya bagus, harga sesuai, packing aman, tapi pengiriman agak lama.",
    height=140,
    label_visibility="collapsed"
)

col_button, col_info = st.columns([1, 3])

with col_button:
    predict_button = st.button("Prediksi Ulasan", use_container_width=True)

with col_info:
    st.caption(
        "Model akan menghasilkan probabilitas untuk setiap label. "
        "Label dianggap terdeteksi jika probabilitasnya melewati threshold yang dipilih."
    )


# =========================================================
# HASIL PREDIKSI
# =========================================================
if predict_button:
    if not review.strip():
        st.warning("Teks ulasan masih kosong. Masukkan ulasan terlebih dahulu.")
    else:
        result_df = predict_review(review, threshold)
        detected_df = result_df[result_df["Hasil"] == "Terdeteksi"].copy()

        st.divider()

        st.subheader("Ringkasan Hasil")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Jumlah Label Terdeteksi", len(detected_df))

        with col2:
            st.metric("Threshold", f"{threshold:.2f}")

        with col3:
            top_probability = result_df["Probabilitas"].max()
            st.metric("Probabilitas Tertinggi", f"{top_probability:.4f}")

        st.divider()

        left_col, right_col = st.columns([1, 1.3])

        with left_col:
            st.subheader("Label Terdeteksi")

            if len(detected_df) > 0:
                display_detected = detected_df[["Label", "Probabilitas"]]
                st.dataframe(
                    display_detected,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(
                    "Belum ada label yang melewati threshold. "
                    "Coba periksa kembali teks ulasan atau turunkan threshold untuk melihat prediksi yang lebih sensitif."
                )

        with right_col:
            st.subheader("Seluruh Probabilitas Label")

            display_all = result_df[["Label", "Probabilitas", "Hasil"]]
            st.dataframe(
                display_all,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        with st.expander("Catatan interpretasi"):
            st.write(
                "Probabilitas menunjukkan tingkat keyakinan model terhadap masing-masing label. "
                "Karena tugas ini bersifat multi-label, satu ulasan dapat memiliki beberapa label sekaligus. "
                "Threshold berfungsi sebagai batas keputusan, bukan sebagai nilai akurasi."
            )
