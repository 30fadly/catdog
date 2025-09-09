import os, json, numpy as np
import streamlit as st
from PIL import Image, ImageFile
from huggingface_hub import hf_hub_download
from tensorflow import keras

# -------------------- Page config --------------------
st.set_page_config(
    page_title="Cat vs Dog",
    page_icon="🐾",
    layout="centered",
)

# -------------------- Constants ----------------------
REPO_ID   = "fadlyy/CatDog"     # << ganti ke repo model HF kamu
MODEL_FN  = "catdog_cnn_final.h5"
LABELS_FN = "labels.json"
CONFIG_FN = "config.json"
HF_TOKEN  = os.environ.get("HF_TOKEN")  # if private model

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_CFG = {"img_size": [128, 128], "threshold": 0.5}

# -------------------- Helpers ------------------------
@st.cache_resource
def hf_path(filename: str):
    return hf_hub_download(repo_id=REPO_ID, filename=filename, token=HF_TOKEN)

@st.cache_resource
def load_model():
    path = hf_path(MODEL_FN)
    return keras.models.load_model(path, compile=False, safe_mode=False)

@st.cache_resource
def load_labels():
    try:
        with open(hf_path(LABELS_FN)) as f:
            return json.load(f)
    except Exception:
        return ["Cat", "Dog"]

@st.cache_resource
def load_config():
    cfg = dict(DEFAULT_CFG)
    try:
        with open(hf_path(CONFIG_FN)) as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg

def preprocess(pil_img, size):
    pil_img = pil_img.convert("RGB").resize(tuple(size))
    arr = (np.asarray(pil_img, dtype=np.float32) / 255.0)[None, ...]
    return pil_img, arr

# -------------------- Header -------------------------
st.markdown(
    """
    <div style="text-align:center">
      <h1 style="margin-bottom:0">🐱🐶 Cat vs Dog</h1>
      <p style="color:#9aa0a6;margin-top:4px">Upload gambar, dapatkan prediksi instan</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Load assets --------------------
with st.spinner("Memuat model..."):
    model = load_model()
    class_names = load_labels()
    cfg = load_config()
    img_size = cfg.get("img_size", DEFAULT_CFG["img_size"])
    th = cfg.get("threshold", 0.5)   # fixed threshold

# -------------------- Uploader -----------------------
uploaded = st.file_uploader(
    "Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False
)

if uploaded:
    try:
        raw_img = Image.open(uploaded)
        vis, x = preprocess(raw_img, img_size)

        with st.spinner("Menghitung prediksi..."):
            p = float(model.predict(x, verbose=0).ravel()[0])   # sigmoid
            pred = int(p >= th)
            label = class_names[pred]

        # ---- Result card ----
        st.markdown(
            f"""
            <div style="
                border:1px solid #30363d; padding:16px; border-radius:12px;
                background:#0e1117; margin-top:8px;">
              <h3 style="margin:0 0 8px 0;">Hasil Prediksi</h3>
              <div style="display:flex; gap:16px; align-items:center;">
                <div style="min-width:120px;">
                  <div style="font-size:28px;font-weight:700;">{label}</div>
                  <div style="color:#9aa0a6;">Prob: {p:.3f}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(vis, caption=f"Prediksi: {label} (p={p:.3f})",
                 use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memproses: {e}")

# -------------------- Footer -------------------------
st.markdown(
    "<p style='text-align:center;color:#9aa0a6;margin-top:24px'>"
    "Made with ❤️ · Model: Keras (.h5) · Loaded from Hugging Face"
    "</p>",
    unsafe_allow_html=True,
)
