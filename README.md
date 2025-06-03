# 📦 Image Classification using CNN on CIFAR-10 Dataset

This project performs image classification using a **Convolutional Neural Network (CNN)** trained on the **CIFAR-10 dataset**. It includes:

- Training and saving a CNN model using **Joblib**  
- Loading the trained model for predictions  
- A **Graphical User Interface (GUI)** built with Tkinter for image classification

---

## 📁 Project Structure

```
Image-Classification-CIFAR10/
│
├── model_training.py         # CNN model creation and training code
├── mymodel.joblib                # Saved trained model using Joblib
├── model_loader.py           # Code to load model and classify images
├── gui_app.py                # GUI interface for image classification
└── README.md                 # Project overview (you are reading this!)
```

---

## 📌 Features

- ✅ Trains a CNN on the CIFAR-10 dataset  
- ✅ Saves the trained model using Joblib  
- ✅ Loads the model in a separate file for prediction  
- ✅ Provides a Tkinter-based GUI for classifying images  
- ✅ Displays predicted class on GUI & Text-to-Speech (TTS) feature to speak the predicted class aloud

---

## 📊 Dataset: CIFAR-10

- 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 🧠 Model Architecture

- Convolutional Layers  
- MaxPooling  
- Flatten  
- Dense (Fully Connected) Layers  
- Activation: ReLU and Softmax  
- Optimizer: Adam  
- Loss: Categorical Crossentropy

---

## 🖼️ GUI Features

- Upload image button  
- Predict class on button click  
- Display result on the interface  
- Easy and interactive design
- Text-to-Speech (TTS) feature

---

## 📚 Libraries Used

- TensorFlow / Keras  
- NumPy  
- Matplotlib (optional)  
- Joblib  
- Tkinter  

---

## 📦 Installation

```bash
pip install tensorflow joblib numpy pillow
```

