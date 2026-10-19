import sys
import os

# Ensure project root is in sys.path when running from any working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np

try:
    from app.utils import (
        DEFAULT_MODEL_PATH,
        DEFAULT_CLASSES_PATH,
        preprocess_image,
        format_predictions,
        load_model_and_classes,
        validate_image,
        get_model_metadata,
        get_file_mtime
    )
except ModuleNotFoundError:
    from utils import (
        DEFAULT_MODEL_PATH,
        DEFAULT_CLASSES_PATH,
        preprocess_image,
        format_predictions,
        load_model_and_classes,
        validate_image,
        get_model_metadata,
        get_file_mtime
    )

# Set page configuration
st.set_page_config(
    page_title="Automotive Image Classifier",
    layout="centered"
)

# Application title
st.title("Automotive Image Classifier")
st.write("Upload an image of a car to see its predicted make and model.")

# Collapsible Model Information Section
metadata = get_model_metadata()
with st.expander("Model Information", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Architecture:** {metadata['architecture']}")
        st.write(f"**Input Dimensions:** {metadata['input_dimensions']}")
        st.write(f"**Supported Classes:** {metadata['num_classes']}")
    with col2:
        st.write(f"**Status:** {metadata['status_label']}")
        st.write(f"**Model ID:** {metadata['version_or_run_id']}")

    if metadata.get("metrics") is not None:
        st.markdown("---")
        st.write("**Evaluation Metrics:**")
        m_col1, m_col2, m_col3 = st.columns(3)
        metrics = metadata["metrics"]
        if "macro_f1" in metrics and metrics["macro_f1"] is not None:
            m_col1.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        if "accuracy" in metrics and metrics["accuracy"] is not None:
            m_col2.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        if "loss" in metrics and metrics["loss"] is not None:
            m_col3.metric("Loss", f"{metrics['loss']:.4f}")
        if metadata.get("evaluation_context"):
            st.caption(f"Context: {metadata['evaluation_context']}")
    else:
        st.caption("Version metadata unavailable.")

# Cached model and classes loading
# Reloading is automatically triggered if the model file modification time changes
model_mtime = get_file_mtime(DEFAULT_MODEL_PATH)

@st.cache_resource
def get_cached_model_and_classes(model_path: str, classes_path: str, mtime: float):
    return load_model_and_classes(model_path, classes_path)

model, class_names = get_cached_model_and_classes(DEFAULT_MODEL_PATH, DEFAULT_CLASSES_PATH, model_mtime)

if model is None or class_names is None:
    st.warning("Serving model not found or could not be loaded. Please ensure models/car_classifier.keras and models/classes.txt are present.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        is_valid, image, err_msg = validate_image(uploaded_file)
        if not is_valid:
            st.error(err_msg)
        else:
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write("")
            st.write("Classifying...")

            try:
                # Preprocess the image
                img_array = preprocess_image(image, target_size=(224, 224))

                # Predict (single-pass softmax from EfficientNet classification head)
                predictions = model.predict(img_array, verbose=0)
                top_3 = format_predictions(predictions, class_names, top_k=3)

                predicted_class, top_conf = top_3[0]

                # Display results
                st.success(f"**Prediction:** {predicted_class}")

                # Display top 3 predictions
                st.write("### Top 3 Predictions:")
                for class_name, confidence in top_3:
                    st.write(f"- **{class_name}**: {confidence:.2f}%")

            except Exception:
                st.error("An error occurred while generating predictions for this image. Please try another image.")
