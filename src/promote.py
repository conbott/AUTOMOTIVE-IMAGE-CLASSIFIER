import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SERVING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "car_classifier.keras")
BASELINE_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_manifest.json")
BACKUPS_DIR = os.path.join(PROJECT_ROOT, "models", "backups")
HISTORY_FILE_PATH = os.path.join(PROJECT_ROOT, "models", "promotion_history.json")
SERVING_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "models", "serving_manifest.json")
ORIGINAL_BASELINE_BACKUP = os.path.join(PROJECT_ROOT, "models", "car_classifier_baseline_backup.keras")

def calculate_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def validate_model_integrity(
    model_path: str,
    expected_input_shape: Tuple = (None, 224, 224, 3),
    expected_output_classes: int = 196
) -> Tuple[bool, Optional[str]]:
    """
    Loads model and validates architecture shapes and forward inference.
    """
    if not os.path.exists(model_path):
        return False, f"Model file not found at: {model_path}"
    try:
        model = tf.keras.models.load_model(model_path)
        in_shape = model.input_shape
        out_shape = model.output_shape

        if in_shape[1:] != expected_input_shape[1:]:
            return False, f"Input shape mismatch: expected {expected_input_shape}, got {in_shape}"
        if out_shape[-1] != expected_output_classes:
            return False, f"Output classes mismatch: expected {expected_output_classes}, got {out_shape[-1]}"

        # Forward pass on synthetic input
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        preds = model.predict(dummy, verbose=0)
        if preds.shape != (1, expected_output_classes):
            return False, f"Prediction shape mismatch: {preds.shape}"

        return True, None
    except Exception as e:
        return False, f"Model loading/validation exception: {str(e)}"

def record_promotion_event(event: Dict[str, Any], history_file: Optional[str] = None):
    """Appends promotion/rollback attempt to durable JSON history."""
    if history_file is None:
        history_file = HISTORY_FILE_PATH
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(event)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def get_baseline_macro_f1() -> float:
    """Retrieves verified baseline macro F1 from baseline_manifest.json."""
    if not os.path.exists(BASELINE_MANIFEST_PATH):
        raise FileNotFoundError(f"Baseline manifest not found at: {BASELINE_MANIFEST_PATH}")
    with open(BASELINE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    f1 = data.get("macro_f1")
    if f1 is None:
        raise ValueError(f"Baseline manifest at {BASELINE_MANIFEST_PATH} does not contain 'macro_f1'")
    return float(f1)

def promote_candidate(
    run_id: str,
    min_f1_delta: float = 0.0,
    force: bool = False,
    serving_model_path: str = SERVING_MODEL_PATH
) -> bool:
    """
    Evaluates candidate quality gates and executes safe atomic model promotion.
    """
    run_dir = os.path.join(PROJECT_ROOT, "artifacts", "runs", run_id)
    if not os.path.isdir(run_dir):
        print(f"ERROR: Candidate run directory not found: {run_dir}")
        return False

    candidate_model_path = os.path.join(run_dir, "candidate_model.keras")
    metrics_path = os.path.join(run_dir, "metrics.json")
    manifest_path = os.path.join(run_dir, "manifest.json")
    class_mapping_path = os.path.join(run_dir, "class_mapping.json")

    # Required files check
    missing_files = []
    for fpath in [candidate_model_path, metrics_path, manifest_path, class_mapping_path]:
        if not os.path.exists(fpath):
            missing_files.append(os.path.basename(fpath))

    if missing_files:
        error_msg = f"Candidate run '{run_id}' is missing required files: {missing_files}"
        print(f"REJECTION: {error_msg}")
        record_promotion_event({
            "run_id": run_id,
            "decision": "REJECTED",
            "reason": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return False

    # Load candidate metrics
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            cand_metrics = json.load(f)
        candidate_f1 = float(cand_metrics["macro_f1"])
        candidate_acc = float(cand_metrics["accuracy"])
        candidate_loss = float(cand_metrics["loss"])
    except Exception as e:
        error_msg = f"Failed to parse candidate metrics from {metrics_path}: {str(e)}"
        print(f"REJECTION: {error_msg}")
        record_promotion_event({
            "run_id": run_id,
            "decision": "REJECTED",
            "reason": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return False

    # Retrieve baseline macro F1
    baseline_f1 = get_baseline_macro_f1()
    target_f1 = baseline_f1 + min_f1_delta

    print("\n" + "=" * 55)
    print("             QUALITY GATE EVALUATION")
    print("=" * 55)
    print(f" Candidate Run ID         : {run_id}")
    print(f" Baseline Macro F1        : {baseline_f1:.4f}")
    print(f" Required Threshold (>=)  : {target_f1:.4f} (delta={min_f1_delta:+.4f})")
    print(f" Candidate Macro F1       : {candidate_f1:.4f}")
    print(f" Candidate Accuracy       : {candidate_acc * 100:.2f}%")
    print(f" Candidate Loss           : {candidate_loss:.4f}")

    passes_gate = candidate_f1 >= target_f1

    if not passes_gate and not force:
        decision = "REJECTED"
        reason = f"Candidate Macro F1 ({candidate_f1:.4f}) does not meet required threshold ({target_f1:.4f})"
        print(f" Decision                 : \033[91m{decision}\033[0m")
        print(f" Reason                   : {reason}")
        print(f" Serving Model Action     : Unchanged ({serving_model_path})")
        print("=" * 55 + "\n")

        record_promotion_event({
            "run_id": run_id,
            "decision": decision,
            "baseline_macro_f1": baseline_f1,
            "candidate_macro_f1": candidate_f1,
            "required_threshold": target_f1,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return False

    decision = "PROMOTED (FORCED)" if (not passes_gate and force) else "PROMOTED"
    print(f" Decision                 : \033[92m{decision}\033[0m")
    print("=" * 55 + "\n")

    # Validate candidate model structure before modifying anything
    print("Validating candidate model structure and inference...")
    is_valid, val_err = validate_model_integrity(candidate_model_path)
    if not is_valid:
        error_msg = f"Candidate model validation failed: {val_err}"
        print(f"ERROR: {error_msg}")
        record_promotion_event({
            "run_id": run_id,
            "decision": "FAILED_VALIDATION",
            "reason": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return False

    candidate_hash = calculate_sha256(candidate_model_path)
    print(f"Candidate Model SHA-256   : {candidate_hash}")

    # Create timestamped backup of current serving model
    pre_serving_hash = None
    backup_path = None
    os.makedirs(BACKUPS_DIR, exist_ok=True)

    if os.path.exists(serving_model_path):
        pre_serving_hash = calculate_sha256(serving_model_path)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUPS_DIR, f"car_classifier_backup_{timestamp_str}.keras")
        shutil.copy2(serving_model_path, backup_path)
        print(f"Current serving model backed up to: {backup_path}")
        print(f"Pre-promotion Serving SHA-256     : {pre_serving_hash}")

    # Safe atomic replacement using same-filesystem staging
    staging_path = os.path.join(os.path.dirname(serving_model_path), f".temp_staged_{run_id}.keras")
    try:
        shutil.copy2(candidate_model_path, staging_path)
        # Verify staged file
        staged_valid, staged_err = validate_model_integrity(staging_path)
        if not staged_valid:
            raise RuntimeError(f"Staged model verification failed: {staged_err}")

        # Atomic replacement on NTFS
        os.replace(staging_path, serving_model_path)

        # Post-replacement validation
        post_valid, post_err = validate_model_integrity(serving_model_path)
        post_serving_hash = calculate_sha256(serving_model_path)

        if not post_valid or post_serving_hash != candidate_hash:
            raise RuntimeError(f"Post-promotion verification failed: {post_err or 'Hash mismatch'}")

        print(f"Model successfully promoted to   : {serving_model_path}")
        print(f"Post-promotion Serving SHA-256    : {post_serving_hash}")

    except Exception as e:
        print(f"\nCRITICAL: Promotion failed during replacement! Error: {str(e)}")
        if backup_path and os.path.exists(backup_path):
            print("Attempting automatic rollback from backup...")
            os.replace(backup_path, serving_model_path)
            print(f"Restored serving model from: {backup_path}")
        if os.path.exists(staging_path):
            try:
                os.remove(staging_path)
            except Exception:
                pass
        return False

    # Update serving manifest
    serving_manifest = {
        "run_id": run_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "model_path": serving_model_path,
        "sha256": candidate_hash,
        "pre_promotion_sha256": pre_serving_hash,
        "backup_path": backup_path,
        "baseline_macro_f1": baseline_f1,
        "metrics": cand_metrics
    }
    with open(SERVING_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(serving_manifest, f, indent=2)

    # Record durable promotion history
    record_promotion_event({
        "run_id": run_id,
        "decision": decision,
        "candidate_sha256": candidate_hash,
        "pre_promotion_sha256": pre_serving_hash,
        "backup_path": backup_path,
        "baseline_macro_f1": baseline_f1,
        "candidate_macro_f1": candidate_f1,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    print(f"Serving manifest updated at      : {SERVING_MANIFEST_PATH}")
    print(f"Promotion history logged at      : {HISTORY_FILE_PATH}")
    return True

def rollback_model(
    backup_path: Optional[str] = None,
    serving_model_path: str = SERVING_MODEL_PATH
) -> bool:
    """
    Rolls back the serving model to a specified or most recent verified backup.
    Never deletes the original baseline backup.
    """
    if backup_path is None or backup_path.lower() == "latest":
        # Find latest timestamped backup or fall back to baseline backup
        if os.path.isdir(BACKUPS_DIR):
            backups = [
                os.path.join(BACKUPS_DIR, f)
                for f in os.listdir(BACKUPS_DIR)
                if f.endswith(".keras")
            ]
            backups.sort(key=os.path.getmtime, reverse=True)
            if backups:
                backup_path = backups[0]

        if backup_path is None or not os.path.exists(backup_path):
            if os.path.exists(ORIGINAL_BASELINE_BACKUP):
                backup_path = ORIGINAL_BASELINE_BACKUP
            else:
                print("ERROR: No backup found to restore.")
                return False

    print(f"Selected backup for rollback: {backup_path}")
    if not os.path.exists(backup_path):
        print(f"ERROR: Backup file does not exist: {backup_path}")
        return False

    # Validate backup integrity before restoring
    print("Validating backup model integrity...")
    is_valid, err = validate_model_integrity(backup_path)
    if not is_valid:
        print(f"ERROR: Backup model failed integrity check: {err}")
        return False

    backup_hash = calculate_sha256(backup_path)
    pre_hash = calculate_sha256(serving_model_path) if os.path.exists(serving_model_path) else None

    # Safe restore
    staging_path = os.path.join(os.path.dirname(serving_model_path), ".temp_rollback.keras")
    try:
        shutil.copy2(backup_path, staging_path)
        os.replace(staging_path, serving_model_path)
        post_hash = calculate_sha256(serving_model_path)
        assert post_hash == backup_hash, "Hash mismatch after rollback"

        print(f"\nSUCCESS: Rollback complete. Restored model from: {backup_path}")
        print(f"Serving Model SHA-256: {post_hash}")

        # Record rollback event
        record_promotion_event({
            "action": "ROLLBACK",
            "restored_from": backup_path,
            "pre_rollback_sha256": pre_hash,
            "restored_sha256": post_hash,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return True
    except Exception as e:
        print(f"ERROR: Failed to complete rollback: {str(e)}")
        if os.path.exists(staging_path):
            try:
                os.remove(staging_path)
            except Exception:
                pass
        return False

def show_status():
    """Prints current serving model status, hash, and baseline metrics."""
    print("\n" + "=" * 55)
    print("           CURRENT SERVING MODEL STATUS")
    print("=" * 55)
    if os.path.exists(SERVING_MODEL_PATH):
        print(f" Serving Model Path : {SERVING_MODEL_PATH}")
        print(f" SHA-256 Checksum   : {calculate_sha256(SERVING_MODEL_PATH)}")
    else:
        print(f" Serving Model      : NOT FOUND")

    if os.path.exists(ORIGINAL_BASELINE_BACKUP):
        print(f" Baseline Backup    : {ORIGINAL_BASELINE_BACKUP}")
        print(f" Baseline SHA-256   : {calculate_sha256(ORIGINAL_BASELINE_BACKUP)}")

    if os.path.exists(BASELINE_MANIFEST_PATH):
        with open(BASELINE_MANIFEST_PATH, "r", encoding="utf-8") as f:
            b = json.load(f)
        print(f" Baseline Macro F1  : {b.get('macro_f1')}")
        print(f" Baseline Accuracy  : {b.get('accuracy') * 100:.2f}%")

    if os.path.exists(SERVING_MANIFEST_PATH):
        with open(SERVING_MANIFEST_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        print(f" Active Run ID      : {s.get('run_id')}")
        print(f" Promoted At        : {s.get('promoted_at')}")
    print("=" * 55 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Model Promotion and Quality Gate Management")
    parser.add_argument("--run-id", type=str, default=None, help="Candidate run ID to evaluate and promote")
    parser.add_argument("--min-f1-delta", type=float, default=0.0, help="Required Macro F1 delta above baseline (default: 0.0)")
    parser.add_argument("--force", action="store_true", help="Force promotion regardless of quality gate")
    parser.add_argument("--rollback", type=str, nargs="?", const="latest", default=None, help="Roll back to latest or specified backup")
    parser.add_argument("--status", action="store_true", help="Display serving model status and baseline metrics")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.rollback is not None:
        rollback_model(args.rollback)
        return

    if args.run_id is not None:
        promote_candidate(args.run_id, min_f1_delta=args.min_f1_delta, force=args.force)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
