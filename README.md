# 😊 Face Emotion Detection with GUI (FER2013 Based)

This project is an AI-based GUI application that detects facial emotions using a trained model based on the **FER2013 dataset**. It offers real-time emotion detection using webcam and also supports image-based emotion recognition via an interactive graphical interface.

---

## 📊 Dataset

We used the **FER2013 (Facial Expression Recognition 2013)** dataset to train our machine learning model. It contains grayscale images of human faces with 7 emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 🧠 Model Training

- The model is trained using **Scikit-learn (sklearn)**.
- Preprocessing and training includes:
  - Image resizing
  - Feature extraction
  - Model fitting using classifiers like SVM/Random Forest
- The trained model is saved using `joblib`.


