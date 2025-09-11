import os
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Hugging Face Model URL
MODEL_URL = "https://huggingface.co/Akshit04/Akshit04-pneumonia-model/resolve/main/pneumonia_model.h5"
MODEL_PATH = "pneumonia_model.h5"

# Download model if not available
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Hugging Face..."):
            response = requests.get(MODEL_URL, stream=True)
            total = int(response.headers.get("content-length", 0))
            with open(MODEL_PATH, "wb") as f:
                downloaded = 0
                for data in response.iter_content(chunk_size=1024*1024):  # 1 MB chunks
                    downloaded += len(data)
                    f.write(data)
                    st.progress(min(downloaded / total, 1.0))
        st.success("✅ Model downloaded successfully!")

# Load the model
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# App UI
st.title("🩺 Pneumonia Detection App")
st.write("Upload a chest X-ray image to check for **Pneumonia** vs **Normal**.")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", width=300)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    confidence = float(pred) if pred > 0.5 else float(1 - pred)

    st.subheader("📊 Prediction Results")
    if pred > 0.5:
        st.error(f"⚠️ Pneumonia Detected with {confidence*100:.2f}% confidence")
        st.progress(confidence)
    else:
        st.success(f"✅ Normal with {confidence*100:.2f}% confidence")
        st.progress(confidence)
    
