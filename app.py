import os
import json
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


# Load class indices
class_indices_path = "class_indices.json"  # Ensure this file exists
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}


# Preprocess image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to 48x48 pixels
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array


# Predict image class
def predict_image_class(model, image, class_indices):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100  # Confidence percentage
    predicted_class_name = class_indices.get(predicted_class_index, "Unknown")
    return predicted_class_name, confidence


# Emotion Labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Streamlit UI
st.title("Facial Emotion Recognition")
st.write("Upload an image to predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button('🔍 Classify'):
            # Load model
            model = load_model()

            # Predict emotion
            emotion, confidence = predict_image_class(model, image, class_indices)

            # Display results
            st.success(f'Prediction: **{emotion}**')
            st.info(f'Confidence: **{confidence:.2f}%**')
