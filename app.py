import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("cnn_model.h5")

# Streamlit UI
st.title("Image Classification with CNN")
st.text("Upload an image to classify")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Preprocess the image
    img = load_img(uploaded_file, target_size=(32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    # Display the result
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Class: {class_labels[class_idx]}")
