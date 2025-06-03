import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import numpy as np
from joblib import load
import pyttsx3

# Load trained model
model = load("mymodel.joblib")

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Main GUI window
root = tk.Tk()
root.title("CIFAR-10 Image Classifier")

# Centering the window on screen
window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
root.configure(bg="#e6e6e6")

selected_file = None

# Main frame
main_frame = tk.Frame(root, bg="white", bd=2, relief=tk.RIDGE)
main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=750, height=550)

# Title
title_label = tk.Label(main_frame, text="CIFAR-10 Image Classifier", bg="white", fg="#003366",
                       font=("Helvetica", 22, "bold"))
title_label.pack(pady=20)

# Image display panel
panel = tk.Label(main_frame, bg="white")
panel.pack(pady=10)

# Label to show status
lbl_status = tk.Label(main_frame, text="No image selected", bg="white", fg="red", font=("Arial", 12))
lbl_status.pack()

# Function to select image
def select_image():
    global selected_file
    selected_file = fd.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if selected_file:
        display_image(selected_file)
        lbl_status.config(text="Image selected successfully!")

# Function to display image
def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

# Function to predict
def make_prediction():
    if selected_file:
        img = Image.open(selected_file).resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        lbl_result.config(text=f"Prediction: {class_names[predicted_class]}")
        engine = pyttsx3.init()
        engine.say(f"Prediction is {class_names[predicted_class]}")
        engine.runAndWait()

# Buttons frame
btn_frame = tk.Frame(main_frame, bg="white")
btn_frame.pack(pady=10)

# Browse Button
browse_btn = tk.Button(btn_frame, text="📂 Browse Image", command=select_image,
                       bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
browse_btn.grid(row=0, column=0, padx=10)

# Predict Button
predict_btn = tk.Button(btn_frame, text="🔍 Predict", command=make_prediction,
                        bg="#673AB7", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
predict_btn.grid(row=0, column=1, padx=10)

# Result Label
lbl_result = tk.Label(main_frame, text="Prediction: ", bg="white", fg="green", font=("Arial", 14, "bold"))
lbl_result.pack(pady=10)

root.mainloop()
