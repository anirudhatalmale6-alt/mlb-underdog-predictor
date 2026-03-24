"""
Model persistence — save and load trained models with metadata.
"""

import json
import joblib
from datetime import datetime
from pathlib import Path

from config.settings import MODELS_DIR
from src.utils.logging import get_logger

log = get_logger(__name__)


def save_model(model, metadata: dict, name: str = "xgb_underdog") -> Path:
    """Save model and metadata to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = MODELS_DIR / f"{name}_{ts}.joblib"
    meta_path = MODELS_DIR / f"{name}_{ts}_meta.json"

    joblib.dump(model, model_path)

    # Convert non-serializable types
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (list, dict, str, int, float, bool, type(None))):
            clean_meta[k] = v
        else:
            clean_meta[k] = str(v)

    with open(meta_path, "w") as f:
        json.dump(clean_meta, f, indent=2, default=str)

    # Also save as "latest"
    latest_model = MODELS_DIR / f"{name}_latest.joblib"
    latest_meta = MODELS_DIR / f"{name}_latest_meta.json"
    joblib.dump(model, latest_model)
    with open(latest_meta, "w") as f:
        json.dump(clean_meta, f, indent=2, default=str)

    log.info(f"Saved model to {model_path}")
    return model_path


def load_latest_model(name: str = "xgb_underdog"):
    """Load the most recently saved model."""
    model_path = MODELS_DIR / f"{name}_latest.joblib"
    meta_path = MODELS_DIR / f"{name}_latest_meta.json"

    if not model_path.exists():
        # Fall back to finding the newest timestamped file
        files = sorted(MODELS_DIR.glob(f"{name}_*.joblib"), reverse=True)
        files = [f for f in files if "latest" not in f.name]
        if not files:
            raise FileNotFoundError(f"No saved model found with name '{name}'")
        model_path = files[0]
        meta_path = model_path.with_name(model_path.stem + "_meta.json")

    model = joblib.load(model_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    log.info(f"Loaded model from {model_path}")
    return model, metadata
