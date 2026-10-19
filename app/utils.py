import os
import io
import base64
import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "car_classifier.keras")
DEFAULT_CLASSES_PATH = os.path.join(PROJECT_ROOT, "models", "classes.txt")

def image_to_base64(image: Image.Image, max_size: Tuple[int, int] = (800, 800)) -> str:
    """
    Converts a PIL Image into an inlined Base64 data URI (data:image/jpeg;base64,...).
    Downsamples a copy if dimensions exceed max_size to maintain fast rendering.
    Enables reliable HTML rendering across cloud proxies and VPNs without relying on MediaFileManager.
    """
    img_copy = image.copy()
    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img_copy.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resizes PIL Image and converts to RGB batch array of shape (1, H, W, 3) in [0, 255] float range.
    Matches training and evaluation preprocessing without double normalization.
    """
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array

def format_predictions(
    probabilities: np.ndarray,
    class_names: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Extracts top_k predictions from a 1D probability distribution.
    Does NOT apply double-softmax since EfficientNet classification head already outputs softmax.
    Returns list of (class_name, confidence_percentage) sorted in descending order.
    """
    if len(probabilities.shape) > 1:
        probs = probabilities[0]
    else:
        probs = probabilities

    if len(probs) != len(class_names):
        raise ValueError(
            f"Dimension mismatch: {len(probs)} probabilities for {len(class_names)} classes."
        )

    # Sort descending
    top_indices = np.argsort(probs)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        confidence_pct = float(probs[idx]) * 100.0
        results.append((class_names[idx], round(confidence_pct, 2)))

    return results

def get_file_mtime(file_path: str) -> float:
    """Safely returns file modification timestamp or 0.0 if not found."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0.0

def load_model_and_classes(
    model_path: Optional[str] = None,
    classes_path: Optional[str] = None
) -> Tuple[Optional[tf.keras.Model], Optional[List[str]]]:
    """
    Safely loads serving model and classes list.
    Returns (None, None) gracefully if either file does not exist or fails to load.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if classes_path is None:
        classes_path = DEFAULT_CLASSES_PATH

    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return None, None

    try:
        model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        return model, class_names
    except Exception as e:
        print(f"Warning: Failed to load model or classes: {e}")
        return None, None

def validate_image(image_file: Any) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
    """
    Validates that the uploaded file can be opened and parsed as a valid image.
    Handles corrupt files, unreadable bytes, or unsupported formats gracefully.
    """
    if image_file is None:
        return False, None, "No image file provided."

    try:
        if hasattr(image_file, "seek"):
            image_file.seek(0)

        image = Image.open(image_file)
        rgb_image = image.convert("RGB")
        rgb_image.load()  # Force decode all pixel data into memory

        if hasattr(image_file, "seek"):
            image_file.seek(0)

        return True, rgb_image, None
    except Exception:
        return False, None, "Unable to read the uploaded image. Please ensure it is a valid JPEG or PNG file."

def get_model_metadata(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves verified model metadata from serving_manifest.json or baseline_manifest.json.
    Returns clear fallback dict if metadata is missing or invalid.
    """
    if models_dir is None:
        models_dir = os.path.join(PROJECT_ROOT, "models")

    serving_manifest_path = os.path.join(models_dir, "serving_manifest.json")
    baseline_manifest_path = os.path.join(models_dir, "baseline_manifest.json")

    metadata = {
        "architecture": "EfficientNetB0 (Transfer Learning)",
        "input_dimensions": "224 x 224 x 3",
        "num_classes": 196,
        "is_available": False,
        "status_label": "Baseline Model",
        "version_or_run_id": "Version metadata unavailable.",
        "metrics": None,
        "evaluation_context": None
    }

    # 1. Check for promoted serving manifest first
    if os.path.exists(serving_manifest_path):
        try:
            with open(serving_manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata["is_available"] = True
            metadata["status_label"] = "Promoted Model"
            metadata["version_or_run_id"] = data.get("run_id", "Promoted Run")
            metadata["metrics"] = data.get("metrics")
            metadata["evaluation_context"] = "Candidate validation split (macro F1 >= baseline)"
            return metadata
        except Exception:
            pass

    # 2. Check for baseline manifest
    if os.path.exists(baseline_manifest_path):
        try:
            with open(baseline_manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata["is_available"] = True
            metadata["status_label"] = "Verified Baseline"
            metadata["version_or_run_id"] = "Baseline Model (Pre-promotion)"
            metadata["metrics"] = {
                "macro_f1": data.get("macro_f1"),
                "accuracy": data.get("accuracy"),
                "loss": data.get("loss"),
                "macro_precision": data.get("macro_precision"),
                "macro_recall": data.get("macro_recall")
            }
            total_samples = data.get("total_samples", 1629)
            num_classes = data.get("num_classes", 196)
            metadata["evaluation_context"] = f"Original validation split ({total_samples:,} images, {num_classes} classes, zero train contamination)"
            return metadata
        except Exception:
            pass

    return metadata
