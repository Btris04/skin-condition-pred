import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
MODEL_PATH = "model.h5"  # Ensure this file exists
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ["Acne", "Clear", "Comedo"]

# Streamlit UI
st.title("🤖 Acne Classification AI")
st.write("Upload an image to classify whether it's **Acne, Clear, or Comedo**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image_pil.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Model expects batch

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_class_index]

    # Show result
    st.success(f"✅ Prediction: **{predicted_label}**")
