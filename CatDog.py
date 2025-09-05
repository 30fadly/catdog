# app.py — Streamlit Cat vs Dog (Simplified)

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import keras

# ===================== UI BASIC =====================
st.set_page_config(page_title="MBC Lab · Cat vs Dog", page_icon="🐶", layout="centered")
st.title("🐶🐱 Cat vs Dog Classifier")
st.caption("jadii ini adalah model yang dibuat untuk membedakan kucing dan anjing. Preprocess: RGB → 96×96 → float32 → MobileNetV2 preprocess_input")

# ===================== LOAD MODEL =====================
@st.cache_resource(show_spinner=True)
def load_model():
    model_path = hf_hub_download(repo_id="fadlyy/CatDog", filename="CatDog.h5")
    # safe_mode=False agar model standar bisa dimuat tanpa custom scope
    return keras.models.load_model(model_path, compile=False, safe_mode=False)

# ===================== PREPROCESS (SINGKAT) =====================
def preprocess(img: Image.Image, model) -> np.ndarray:
    # contoh input: (None, 96, 96, 3) — ambil H, W dari model
    _, H, W, _ = model.input_shape
    img = img.convert("RGB").resize((W, H))
    x = np.asarray(img, dtype="float32")             # [0..255]
    x = preprocess_input(x)                          # MobileNetV2
    return np.expand_dims(x, axis=0)                 # (1, H, W, 3)

# ===================== SIDEBAR =====================
with st.sidebar:
    uploaded = st.file_uploader("📤 Upload gambar", type=["jpg", "jpeg", "png"])
    run = st.button("🔮 Prediksi", use_container_width=True)
    st.markdown("---")
    st.caption("Model: MobileNetV2 · Output sigmoid (threshold 0.5)")

# ===================== ACTION =====================
if uploaded and run:
    img = Image.open(uploaded)
    st.image(img, caption="Gambar diunggah", use_column_width=True)

    try:
        with st.spinner("Memuat model & memproses…"):
            model = load_model()
            x = preprocess(img, model)
            y = model.predict(x, verbose=0)

        # y bisa berbentuk (1,) atau (1,1). Anggap nilai adalah prob 'Anjing'
        p = float(np.squeeze(y))
        label = "Anjing 🐶" if p >= 0.5 else "Kucing 🐱"
        conf = p if label.startswith("Anjing") else (1 - p)

        st.success(f"Prediksi: **{label}** (confidence {conf:.2%})")

        with st.expander("Detail input model"):
            st.write("Input shape:", model.input_shape)

    except Exception as e:
        st.error(f"Gagal memuat/memprediksi: {e}")

else:
    st.info("Upload gambar di sidebar, lalu klik **Prediksi**.")
