import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model (ignore compile metrics warning)
model = load_model("pneumonia_model.h5", compile=False)

# App title
st.title("🩺 Pneumonia Detection App")
st.write("Upload a chest X-ray image, and the model will predict whether it shows **Pneumonia** or **Normal**.")

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width="stretch")

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    pred = model.predict(img_array)[0][0]
    confidence = (pred * 100) if pred > 0.5 else ((1 - pred) * 100)

    # Show result with confidence
    if pred > 0.5:
        st.error(f"Prediction: **Pneumonia Detected ❌** (Confidence: {confidence:.2f}%)")
    else:
        st.success(f"Prediction: **Normal ✅** (Confidence: {confidence:.2f}%)")

    # Show probability bar
    st.write("### Prediction Probability")
    st.progress(int(confidence))
