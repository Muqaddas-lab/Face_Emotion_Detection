import cv2
import numpy as np
from keras.models import model_from_json
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

# Load the model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('emotion_model.weights.h5')

# Emotion dictionary
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Create Tkinter window
window = Tk()
window.title("Face Emotion Detection System")
window.configure(bg='#4CAF50')  # Attractive green background color
window.geometry("900x600")

# Add a bold title
title_label = Label(window, text="Emotion Detection System", font=("Arial", 24, "bold"), fg="white", bg="#4CAF50")
title_label.pack(pady=20)

# Global variables
cap = None
camera_active = False
camera_thread = None

# Camera feed function with emotion detection and screenshot capture
def detect_emotion_live():
    global cap, camera_active
    if not camera_active:  # Check if camera is already running
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            return
        camera_active = True

        while camera_active:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame not read properly.")
                break

            frame = cv2.resize(frame, (1280, 720))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                emotion = emotion_dict[maxindex]

                # Display emotion on the frame
                cv2.putText(frame, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection - Camera', frame)

            # Press 's' to save a screenshot
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite("screenshot.png", frame)
                print("Screenshot saved as 'screenshot.png'")

            # Exit when 'q' is pressed or camera_active is set to False
            if cv2.waitKey(1) & 0xFF == ord('q') or not camera_active:
                break

        cap.release()
        cv2.destroyAllWindows()

# Function to start the camera in a new thread
def start_camera_thread():
    global camera_thread
    if camera_thread is None or not camera_thread.is_alive():  # Check if the thread is already running
        camera_thread = threading.Thread(target=detect_emotion_live)
        camera_thread.start()

# Close camera function
def close_camera():
    global camera_active
    camera_active = False  # Setting it to False will stop the camera feed loop

# Image file selection function
def detect_emotion_file():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_img = gray_img[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_img, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]

        # Display emotion on the image
        cv2.putText(img, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show image with detected emotion
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    img_label.config(image=img_tk)
    img_label.image = img_tk

def close_image():
    img_label.config(image='')

# Load the camera icon image
camera_icon = Image.open("camera-icon.png")
camera_icon_resized = camera_icon.resize((50, 50), Image.Resampling.LANCZOS)
camera_icon_tk = ImageTk.PhotoImage(camera_icon_resized)

# Load the file upload icon image
upload_icon = Image.open("My-Pictures-icon.png")  # Ensure correct path
upload_icon_resized = upload_icon.resize((50, 50), Image.Resampling.LANCZOS)
upload_icon_tk = ImageTk.PhotoImage(upload_icon_resized)

# Button to open live camera feed with the icon and text
camera_button = Button(window, image=camera_icon_tk, text=" Open Camera", compound=LEFT, command=start_camera_thread, bg="blue", fg="white", font=("Arial", 14, "bold"))
camera_button.pack(pady=20)

# Button to close the camera feed
close_camera_button = Button(window, text="Close Camera", command=close_camera, bg="darkblue", fg="white", font=("Arial", 14, "bold"))
close_camera_button.pack(pady=10)

# Button to open image file with the upload icon and text
file_button = Button(window, image=upload_icon_tk, text=" Upload Image", compound=LEFT, command=detect_emotion_file, bg="red", fg="white", font=("Arial", 14, "bold"))
file_button.pack(pady=20)

# Button to close the displayed image
close_image_button = Button(window, text="Close Image", command=close_image, bg="darkred", fg="white", font=("Arial", 14, "bold"))
close_image_button.pack(pady=10)

# Label for displaying image
img_label = Label(window)
img_label.pack(pady=10)

# Exit button to close the application
exit_button = Button(window, text="Exit", command=window.quit, bg="#555555", fg="white", font=("Arial", 14, "bold"))
exit_button.pack(pady=20)

# Start the Tkinter loop
window.mainloop()
