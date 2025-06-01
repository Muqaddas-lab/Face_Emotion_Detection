import cv2
import numpy as np
from keras.models import model_from_json

# Load model structure and weights
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('emotion_model.weights.h5')

# Dictionary for emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
else:
    print("Camera successfully opened.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not read properly.")
        break

    # Resize frame
    frame = cv2.resize(frame, (1280, 720))

    # Create a face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    print(f"Faces detected: {len(num_faces)}")

    # Process each detected face
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]
        print(f"Emotion detected: {emotion}")

        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 