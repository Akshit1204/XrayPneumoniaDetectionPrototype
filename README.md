# 🩺 Pneumonia Detection Web App

A simple **Streamlit** app that detects Pneumonia from Chest X-ray images using a deep learning model (`pneumonia_model.h5`).
<img width="1793" height="1027" alt="image" src="https://github.com/user-attachments/assets/81651379-4322-437a-9a42-b727c1cf4dfc" />
<img width="1792" height="1030" alt="image" src="https://github.com/user-attachments/assets/5b2d08a5-8b34-4cf5-9c15-f9ee9c0f7346" />

---

## 🚀 Features
- Upload a chest X-ray image (`.jpg`, `.jpeg`, `.png`)
- Model predicts whether the patient has **Pneumonia** or **Normal lungs**
- Confidence score shown for each prediction
- Built with **TensorFlow/Keras** and **Streamlit**

---

## 📂 Project Structure
XrayPneumoniaWebApp/
- │──app.py # Streamlit app
- │──train_model.py # (Optional) Training script used to train the model
- │──evaluate_model.py # Evaluate model accuracy on test data
- │──pneumonia_model.h5 # Trained model (trained on the given dataset and uploaded to https://huggingface.co/Akshit04/Akshit04-pneumonia-model/tree/main)
- │──requirements.txt # Dependencies
- │──chest_xray/ # Dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

---

## ⚙️ Installation & Running

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/XrayPneumoniaWebApp.git
   cd XrayPneumoniaWebApp

2. Create virtual environment and install dependencies:
   - python -m venv venv
   - venv\Scripts\activate     # On Windows
   - source venv/bin/activate  # On Mac/Linux

   - pip install -r requirements.txt

3. Download the Dataset for further testing/improvements/re-training for better accuracy
   - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

4. Download the model file
   - 👉https://huggingface.co/Akshit04/Akshit04-pneumonia-model/tree/main
   - Place it inside the project root (XrayPneumoniaWebApp/)

   - streamlit run app.py

---

📊 Model Accuracy

Evaluated on the test dataset:

- ✅ Test Accuracy: ~82.5%

- 📉 Test Loss: ~0.39

---

📝 Notes

The .h5 file is too large for GitHub (130 MB). So i have uploaded it to my Hugging face model repository 👉 https://huggingface.co/Akshit04/Akshit04-pneumonia-model/tree/main.

---

