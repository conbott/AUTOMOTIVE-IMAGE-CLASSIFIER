import argparse
import json
import os
import pathlib
import random
import sys
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
import yaml

# Enable local filesystem tracking backend in MLflow 3+
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config
from src.dataset import get_datasets
from src.model import build_model, unfreeze_model
from src.evaluate import evaluate_model

class MLflowEpochCallback(tf.keras.callbacks.Callback):
    """
    Logs per-epoch training and validation loss/accuracy to the active MLflow run.
    """
    def __init__(self, step_offset: int = 0):
        super().__init__()
        self.step_offset = step_offset

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            current_step = epoch + self.step_offset
            for metric_name, val in logs.items():
                mlflow.log_metric(f"epoch_{metric_name}", float(val), step=current_step)

def set_deterministic_seeds(seed: int = 42):
    """
    Sets random seeds for Python, NumPy, and TensorFlow.
    
    NOTE ON NONDETERMINISM:
    While random seeds ensure deterministic dataset shuffling, data augmentation,
    and initial weight generation, minor nondeterminism can still occur on multi-threaded
    CPUs or GPUs due to asynchronous floating-point atomic reductions (e.g. oneDNN / cuDNN).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print(f"Random seed set to: {seed}")

def train(
    config_path: str = None,
    epochs_override: int = None,
    run_id: str = None,
    dry_run: bool = False
) -> str:
    """
    Executes configuration-driven training with local MLflow tracking.
    Saves candidate artifacts into: artifacts/runs/<run_id>/.
    Does NOT overwrite the active serving model.
    """
    config = load_config(config_path)
    seed = config.get("seed", 42)
    set_deterministic_seeds(seed)

    # Generate or use provided run ID
    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    artifacts_root = os.path.join(PROJECT_ROOT, config["output"].get("artifacts_dir", "artifacts/runs"))
    run_dir = os.path.join(artifacts_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    candidate_model_path = os.path.join(run_dir, "candidate_model.keras")

    print(f"\n==========================================")
    print(f"Starting Training Run: {run_id}")
    print(f"Run Artifacts Directory: {run_dir}")
    print(f"==========================================\n")

    # Configure local MLflow tracking
    tracking_rel = config.get("mlflow", {}).get("tracking_uri", "mlruns")
    tracking_dir = os.path.join(PROJECT_ROOT, tracking_rel)
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_uri = pathlib.Path(tracking_dir).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("mlflow", {}).get("experiment_name", "automotive-image-classifier")
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI : {tracking_uri}")
    print(f"MLflow Experiment   : {experiment_name}")

    # Save run configuration snapshot
    config_snapshot_path = os.path.join(run_dir, "run_config.yaml")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    image_size = tuple(config["data"]["image_size"])
    batch_size = config["data"]["batch_size"]
    train_dir = os.path.join(PROJECT_ROOT, config["data"]["train_dir"])
    val_dir = os.path.join(PROJECT_ROOT, config["data"]["val_dir"])

    with mlflow.start_run(run_name=run_id) as active_run:
        mlflow_run_id = active_run.info.run_id
        print(f"Active MLflow Run ID: {mlflow_run_id}")

        # Log system and run tags
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("mlflow_run_id", mlflow_run_id)
        mlflow.set_tag("model_architecture", config["model"].get("architecture", "EfficientNetB0"))
        mlflow.set_tag("model_role", "candidate")
        mlflow.set_tag("framework", "TensorFlow/Keras")
        mlflow.set_tag("validation_dataset", "data/processed/val")

        # Log hyperparameters
        mlflow.log_params({
            "seed": seed,
            "image_size": f"{image_size[0]}x{image_size[1]}",
            "batch_size": batch_size,
            "num_classes": config["model"]["num_classes"],
            "dropout_rate": config["model"].get("dropout_rate", 0.2),
            "initial_lr": config["training"].get("initial_lr", 0.001),
            "fine_tune_lr": config["training"].get("fine_tune_lr", 1e-5),
            "initial_epochs": epochs_override if epochs_override is not None else config["training"].get("initial_epochs", 10),
            "fine_tune_epochs": epochs_override if epochs_override is not None else config["training"].get("fine_tune_epochs", 10),
            "early_stopping_patience": config["training"].get("early_stopping_patience", 3)
        })

        if dry_run:
            print("\n[DRY RUN MODE] Initializing dry run check (synthetic validation without heavy training)...")
            num_classes = config["model"]["num_classes"]
            class_names = [f"class_{i}" for i in range(num_classes)]
            model, base_model = build_model(num_classes=num_classes, image_size=image_size)
            model.save(candidate_model_path)
            
            # Save lightweight model summary
            summary_path = os.path.join(run_dir, "model_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
            
            metrics = {
                "loss": 1.7500,
                "accuracy": 0.5500,
                "macro_f1": 0.5400,
                "macro_precision": 0.5600,
                "macro_recall": 0.5500
            }
            # Write metrics and manifest
            metrics_path = os.path.join(run_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            
            cm_path = os.path.join(run_dir, "confusion_matrix.json")
            with open(cm_path, "w", encoding="utf-8") as f:
                json.dump([[0]], f)

            class_mapping_path = os.path.join(run_dir, "class_mapping.json")
            with open(class_mapping_path, "w", encoding="utf-8") as f:
                json.dump({"total_classes": num_classes}, f, indent=2)

        else:
            print(f"Loading datasets (image_size={image_size}, batch_size={batch_size})...")
            train_ds, val_ds, class_names = get_datasets(
                train_dir=train_dir,
                val_dir=val_dir,
                batch_size=batch_size,
                image_size=image_size,
                seed=seed
            )

            if train_ds is None:
                raise RuntimeError("Failed to load training dataset.")

            num_classes = len(class_names)
            print(f"Loaded dataset with {num_classes} classes.")

            # Save machine-readable class mapping
            class_mapping = {
                "index_to_class": {i: name for i, name in enumerate(class_names)},
                "class_to_index": {name: i for i, name in enumerate(class_names)},
                "num_classes": num_classes
            }
            class_mapping_path = os.path.join(run_dir, "class_mapping.json")
            with open(class_mapping_path, "w", encoding="utf-8") as f:
                json.dump(class_mapping, f, indent=2)

            # Build model
            dropout_rate = config["model"].get("dropout_rate", 0.2)
            initial_lr = config["training"].get("initial_lr", 0.001)
            print(f"Building EfficientNetB0 model (dropout={dropout_rate}, initial_lr={initial_lr})...")
            model, base_model = build_model(
                num_classes=num_classes,
                image_size=image_size,
                learning_rate=initial_lr
            )

            # Save model architecture summary
            summary_path = os.path.join(run_dir, "model_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))

            # Callbacks
            patience = config["training"].get("early_stopping_patience", 3)
            monitor_metric = config["training"].get("monitor_metric", "val_loss")
            checkpoint_metric = config["training"].get("checkpoint_metric", "val_accuracy")

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric, patience=patience, restore_best_weights=True
            )
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                candidate_model_path, monitor=checkpoint_metric, save_best_only=True
            )

            # Stage 1: Initial training (frozen base)
            initial_epochs = epochs_override if epochs_override is not None else config["training"].get("initial_epochs", 10)
            print(f"\n--- Stage 1: Training top layers ({initial_epochs} epochs) ---")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=initial_epochs,
                callbacks=[early_stopping, model_checkpoint, MLflowEpochCallback(step_offset=0)]
            )

            # Stage 2: Fine-tuning (unfrozen base)
            fine_tune_epochs = epochs_override if epochs_override is not None else config["training"].get("fine_tune_epochs", 10)
            if fine_tune_epochs > 0:
                fine_tune_lr = config["training"].get("fine_tune_lr", 1e-5)
                print(f"\n--- Stage 2: Fine-tuning ({fine_tune_epochs} epochs, lr={fine_tune_lr}) ---")
                model = unfreeze_model(model, base_model, learning_rate=fine_tune_lr)
                total_epochs = initial_epochs + fine_tune_epochs
                initial_epoch_stage2 = len(history.epoch)

                history_fine = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=total_epochs,
                    initial_epoch=initial_epoch_stage2,
                    callbacks=[early_stopping, model_checkpoint, MLflowEpochCallback(step_offset=initial_epoch_stage2)]
                )

            # Ensure best weights are saved
            if not os.path.exists(candidate_model_path):
                model.save(candidate_model_path)

            print(f"\nTraining complete. Candidate model saved to: {candidate_model_path}")

            # Run evaluation on candidate model to generate metrics.json
            print("\nEvaluating candidate model on validation set...")
            metrics = evaluate_model(
                model_path=candidate_model_path,
                val_dir=val_dir,
                output_dir=run_dir,
                batch_size=batch_size,
                image_size=image_size,
                manifest_name="metrics.json"
            )

        # Log final validation metrics to MLflow
        mlflow.log_metrics({
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_macro_precision": metrics["macro_precision"],
            "val_macro_recall": metrics["macro_recall"]
        })

        # Create run manifest
        manifest = {
            "run_id": run_id,
            "mlflow_run_id": mlflow_run_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "candidate_model_path": candidate_model_path,
            "metrics": {
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"]
            },
            "config": config
        }
        manifest_path = os.path.join(run_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Log lightweight artifacts to MLflow
        mlflow.log_artifact(config_snapshot_path, artifact_path="config")
        mlflow.log_artifact(class_mapping_path, artifact_path="classes")
        mlflow.log_artifact(os.path.join(run_dir, "metrics.json"), artifact_path="evaluation")
        mlflow.log_artifact(os.path.join(run_dir, "confusion_matrix.json"), artifact_path="evaluation")
        mlflow.log_artifact(manifest_path, artifact_path="manifest")
        mlflow.log_artifact(summary_path, artifact_path="model")

        print(f"MLflow artifacts successfully logged for run: {run_id}")
        print(f"Run manifest generated at: {manifest_path}")

    return run_id

def main():
    parser = argparse.ArgumentParser(description="Train Automotive Image Classifier with MLflow Tracking")
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for rapid testing")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run identifier")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry-run to verify tracking pipeline without full training")
    args = parser.parse_args()

    train(
        config_path=args.config,
        epochs_override=args.epochs,
        run_id=args.run_id,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
