# 🩺 Pneumonia Detection Web App Prototype

A simple **Streamlit** app that detects Pneumonia from Chest X-ray images using a deep learning model (`pneumonia_model.h5`).
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3f2c23bf-8f93-4efc-bdeb-15c77262605e" />
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/ce0cfa31-42f7-45cc-8213-456479a3f2d3" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5563968d-bd26-48e5-9ec7-ec9996d24693" />
<img width="1919" height="605" alt="image" src="https://github.com/user-attachments/assets/e3750832-f47f-44b8-910f-a463e75b8bce" />
<img width="1920" height="1080" alt="Screenshot (183)" src="https://github.com/user-attachments/assets/a0418874-c192-4bde-b62e-599ee216673d" />

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

4. You can also download the trained model file from my repo(optional)
   - 👉https://huggingface.co/Akshit04/Akshit04-pneumonia-model/tree/main

   - streamlit run app.py

---

📊 Model Accuracy

Evaluated on the test dataset:

- ✅ Test Accuracy: ~82.5%

- 📉 Test Loss: ~0.39

---

📝 Notes

The pneumonia_model.h5 file is too large for GitHub (130 MB). So i have uploaded it to my Hugging face model repository 👉 https://huggingface.co/Akshit04/Akshit04-pneumonia-model/tree/main.

---

