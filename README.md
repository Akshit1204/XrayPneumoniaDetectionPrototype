# 🩺 Pneumonia Detection Web App

A simple **Streamlit** app that detects Pneumonia from Chest X-ray images using a deep learning model (`pneumonia_model.h5`).

---

## 🚀 Features
- Upload a chest X-ray image (`.jpg`, `.jpeg`, `.png`)
- Model predicts whether the patient has **Pneumonia** or **Normal lungs**
- Confidence score shown for each prediction
- Built with **TensorFlow/Keras** and **Streamlit**

---

## 📂 Project Structure
XrayPneumoniaWebApp/
-│── app.py # Streamlit app
-│── train_model.py # (Optional) Training script used to train the model
-│── evaluate_model.py # Evaluate model accuracy on test data
-│── pneumonia_model.h5 # Trained model (trained on the given dataset and uploaded to https://huggingface.co/Akshit04/Akshit04-pneumonia-model/tree/main)
-│── requirements.txt # Dependencies
-│── chest_xray/ # Dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

---

## ⚙️ Installation & Running

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/XrayPneumoniaWebApp.git
   cd XrayPneumoniaWebApp

Create virtual environment and install dependencies:
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt

Download the model file
👉https://huggingface.co/Akshit04/Akshit04-pneumonia-model/tree/main
Place it inside the project root (XrayPneumoniaWebApp/)

streamlit run app.py
