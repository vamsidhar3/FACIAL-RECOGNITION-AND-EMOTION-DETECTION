import streamlit as st
import numpy as np
import cv2
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from PIL import Image

# Load the trained model
def load_model():
    with open("fer.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("fer.h5")
    return model

# Preprocess image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to 48x48 pixels
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Emotion Labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Streamlit UI
st.title("Facial Emotion Recognition")
st.write("Upload an image to predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model = load_model()

    # Preprocess image
    img_array = preprocess_image(image)

    # Make prediction
    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]

    # Display result
    st.subheader(f"Predicted Emotion: {emotion}")
