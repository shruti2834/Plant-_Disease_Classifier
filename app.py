import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🌿 Plant disese classifier ")
st.write("Upload a leaf image to detect its disease.")

model = tf.keras.models.load_model("plant_disease_model.h5")  # Make sure this file exists
class_names = ['Healthy', 'Powdery', 'Rust']  # Example classes

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.write("Classifying...")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"Prediction: {predicted_class}")
