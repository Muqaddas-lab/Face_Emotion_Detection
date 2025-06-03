# Required imports
from tensorflow.keras import datasets, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from joblib import load
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image

# Load your model (assuming it's a joblib model, otherwise use models.load_model() for Keras models)
model = load("mymodel.joblib")  # Make sure the model is correctly loaded.

# Dataset loading and normalization
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display a single image from the dataset and its prediction
img_index = 4999
predictions = model.predict(train_images[img_index].reshape(1, 32, 32, 3))
i = np.argmax(predictions)
print(f'Predicted class: {class_names[i]}')

# Display the image
plt.imshow(train_images[img_index])
plt.xlabel(class_names[i])
plt.show()

# Tkinter setup for image selection
root = Tk()
root.withdraw()  # Hide the root window
image_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])

if image_path:
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize to match CIFAR-10 dimensions
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 32, 32, 3)  # Reshape

    # Predict using the model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]
    print(f'Predicted class: {predicted_label}')

    # Display the selected image with predicted label
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()

# Confusion Matrix and Classification Report
test_predictions = model.predict(test_images)
test_predicted_classes = np.argmax(test_predictions, axis=1)
test_true_classes = test_labels.flatten()

# Compute confusion matrix
cm = confusion_matrix(test_true_classes, test_predicted_classes)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Print classification report
print("\nClassification Report:\n", classification_report(test_true_classes, test_predicted_classes, target_names=class_names))



# import pyttsx3
# # Text-to-speech for the predicted label
# engine = pyttsx3.init()
# engine.say(f"The predicted class is {predicted_label}")
# engine.runAndWait()