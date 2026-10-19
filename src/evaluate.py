import argparse
import json
import os
import sys
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

import pathlib
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_val_dataset

def evaluate_model(
    model_path: str,
    val_dir: str = None,
    classes_path: str = None,
    output_dir: str = None,
    batch_size: int = 32,
    image_size: tuple = (224, 224),
    manifest_name: str = "metrics.json",
    log_mlflow: bool = False,
    mlflow_run_name: str = None
) -> dict:
    """
    Evaluates a saved Keras model on the un-shuffled validation set.
    Computes Loss, Accuracy, Macro F1, Macro Precision/Recall, and Per-class metrics.
    Outputs machine-readable metrics JSON and confusion matrix.
    Optionally logs metrics and evaluation artifacts to local MLflow.
    """
    if val_dir is None:
        val_dir = os.path.join(PROJECT_ROOT, "data", "processed", "val")
    if classes_path is None:
        classes_path = os.path.join(PROJECT_ROOT, "models", "classes.txt")
    if output_dir is None:
        output_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading validation data from: {val_dir} (shuffle=False)...")
    val_ds, dataset_class_names = get_val_dataset(val_dir, batch_size=batch_size, image_size=image_size)
    if val_ds is None:
        raise FileNotFoundError(f"Validation data not found at: {val_dir}")

    # Load classes
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        classes = dataset_class_names

    # Collect ground-truth labels and model predictions
    y_true_indices = []
    y_pred_probs = []

    print("Running evaluation across validation batches...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        # labels are categorical (one-hot)
        y_true_indices.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true_indices, axis=0)
    y_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Compute loss using CategoricalCrossentropy
    cce = tf.keras.losses.CategoricalCrossentropy()
    # convert y_true to one-hot for loss calculation
    y_true_one_hot = tf.one_hot(y_true, depth=len(classes)).numpy()
    val_loss = float(cce(y_true_one_hot, y_probs).numpy())

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Compute per-class classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes[:len(set(y_true).union(set(y_pred)))],
        output_dict=True,
        zero_division=0
    )

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics_payload = {
        "model_path": os.path.abspath(model_path),
        "architecture": "EfficientNetB0",
        "validation_dataset": os.path.relpath(val_dir, PROJECT_ROOT),
        "class_mapping_path": os.path.relpath(classes_path, PROJECT_ROOT) if os.path.exists(classes_path) else None,
        "preprocessing": {
            "image_size": list(image_size),
            "normalization": "EfficientNet built-in (0-255 scale)",
            "color_mode": "RGB"
        },
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": int(len(y_true)),
        "num_classes": int(len(classes)),
        "loss": round(val_loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "classification_report": report
    }

    # Save metrics JSON
    metrics_file = os.path.join(output_dir, manifest_name)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")

    # Save confusion matrix JSON
    cm_file = os.path.join(output_dir, "confusion_matrix.json")
    with open(cm_file, "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f)
    print(f"Confusion matrix saved to: {cm_file}")

    # Optional MLflow logging
    if log_mlflow:
        tracking_dir = os.path.join(PROJECT_ROOT, "mlruns")
        tracking_uri = pathlib.Path(tracking_dir).as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("automotive-image-classifier")
        
        run_name = mlflow_run_name or f"eval_{os.path.basename(model_path)}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("run_type", "evaluation")
            mlflow.set_tag("model_path", os.path.basename(model_path))
            mlflow.set_tag("val_dataset", os.path.relpath(val_dir, PROJECT_ROOT))
            
            mlflow.log_metric("val_loss", metrics_payload["loss"])
            mlflow.log_metric("val_accuracy", metrics_payload["accuracy"])
            mlflow.log_metric("val_macro_f1", metrics_payload["macro_f1"])
            mlflow.log_metric("val_macro_precision", metrics_payload["macro_precision"])
            mlflow.log_metric("val_macro_recall", metrics_payload["macro_recall"])
            
            mlflow.log_artifact(metrics_file, artifact_path="evaluation")
            mlflow.log_artifact(cm_file, artifact_path="evaluation")
            print(f"Evaluation logged to MLflow under run: '{run_name}'")

    # Print summary table
    print("\n" + "=" * 50)
    print("           EVALUATION SUMMARY")
    print("=" * 50)
    print(f" Total Samples Evaluated : {len(y_true)}")
    print(f" Number of Classes       : {len(classes)}")
    print(f" Validation Loss         : {val_loss:.4f}")
    print(f" Validation Accuracy     : {acc * 100:.2f}%")
    print(f" Validation Macro F1     : {macro_f1:.4f}")
    print(f" Validation Macro Prec   : {macro_precision:.4f}")
    print(f" Validation Macro Recall : {macro_recall:.4f}")
    print("=" * 50 + "\n")

    return metrics_payload

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained car classifier model.")
    parser.add_argument("--model-path", type=str, default="models/car_classifier.keras", help="Path to .keras model")
    parser.add_argument("--val-dir", type=str, default="data/processed/val", help="Path to val dataset")
    parser.add_argument("--classes-path", type=str, default="models/classes.txt", help="Path to classes file")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for metrics")
    parser.add_argument("--manifest-name", type=str, default="metrics.json", help="Filename for metrics json")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--log-mlflow", action="store_true", help="Log evaluation metrics and artifacts to local MLflow")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Custom run name for MLflow")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        val_dir=args.val_dir,
        classes_path=args.classes_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        manifest_name=args.manifest_name,
        log_mlflow=args.log_mlflow,
        mlflow_run_name=args.mlflow_run_name
    )

if __name__ == "__main__":
    main()
