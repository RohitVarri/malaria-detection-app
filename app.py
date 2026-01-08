import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ----------------------------------
# Page settings
# ----------------------------------
st.set_page_config(
    page_title="Malaria Detection",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 Malaria Detection System")
st.write("Upload a blood smear image to detect malaria")

# ----------------------------------
# Load trained model
# ----------------------------------
@st.cache_resource
def load_model():
    import os
    model_path = "malaria_hybrid_model.keras"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.warning("⚠️ Model file not found!")
        st.info("To deploy this app with predictions, you need to upload the trained model.")
        return None

model = load_model()

# ----------------------------------
# Class labels (same order as training)
# ----------------------------------
class_names = ["Parasitized", "Uninfected"]

# ----------------------------------
# Image uploader
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload Blood Smear Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----------------------------------
    # Preprocessing
    # ----------------------------------
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ----------------------------------
    # Prediction
    # ----------------------------------
    if st.button("Predict"):
        if model is None:
            st.error("❌ Model not loaded. Cannot make predictions without the trained model file.")
        else:
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.success(f"Prediction: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
