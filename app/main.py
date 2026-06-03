import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Automotive Image Classifier",
    page_icon="🚗",
    layout="centered"
)

# Application title
st.title("🚗 Automotive Image Classifier")
st.write("Upload an image of a car to see its predicted make and model.")

# Load the model and classes
@st.cache_resource
def load_model_and_classes():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'car_classifier.keras')
    classes_path = os.path.join(base_dir, 'models', 'classes.txt')
    
    if not os.path.exists(model_path):
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    return model, class_names

model, class_names = load_model_and_classes()

if model is None:
    st.warning("⚠️ Model not found. Please run the training script first.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        st.write("")
        st.write("Classifying...")
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        # Predict
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = 100 * np.max(score) # Or just np.max(predictions[0]) if already softmaxed
        
        # Display results
        st.success(f"**Prediction:** {predicted_class}")
        
        # Optional: Display top 3 predictions
        st.write("### Top 3 Predictions:")
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        for i in top_3_indices:
            st.write(f"- **{class_names[i]}**: {predictions[0][i]*100:.2f}%")
