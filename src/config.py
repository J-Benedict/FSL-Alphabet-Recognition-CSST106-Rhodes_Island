"""Central configuration for the FSL Alphabet Recognition System."""

from __future__ import annotations

import os

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Raw FSL dataset used for the alphabet reference pictures.
# Set FSL_DATASET_DIR to the folder that contains A/, B/, ... Z/.
_env_dataset_dir = os.environ.get("FSL_DATASET_DIR")
_dataset_candidates = (
    Path(_env_dataset_dir).expanduser() if _env_dataset_dir else None,
    PROJECT_ROOT / "data" / "FSL Dataset" / "Collated",
    PROJECT_ROOT / "data" / "FSL_Dataset" / "Collated",
    PROJECT_ROOT / "FSL Dataset" / "Collated",
)
DATASET_DIR = next(
    (path for path in _dataset_candidates if path is not None and path.is_dir()),
    PROJECT_ROOT / "data" / "FSL Dataset" / "Collated",
)

# Classes A-Z
CLASSES = [chr(ord("A") + i) for i in range(26)]
NUM_CLASSES = len(CLASSES)
FEATURE_DIM = 63  # 21 hand landmarks * (x, y, z)

# FSL-105 Dynamic Phrase Dataset & Model Configuration
_env_fsl105_dir = os.environ.get("FSL105_DATASET_DIR")
_fsl105_candidates = (
    Path(_env_fsl105_dir).expanduser() if _env_fsl105_dir else None,
    DATA_DIR / "FSL-105 A dataset for recognizing 105 Filipino sign language videos",
    PROJECT_ROOT / "FSL-105 A dataset for recognizing 105 Filipino sign language videos",
    DATA_DIR if (DATA_DIR / "clips").is_dir() else None,
)
FSL105_DIR = next(
    (path for path in _fsl105_candidates if path is not None and path.is_dir()),
    DATA_DIR / "FSL-105 A dataset for recognizing 105 Filipino sign language videos",
)
FSL105_LANDMARKS_PATH = DATA_DIR / "fsl105_landmarks.npz"
FSL105_THUMBS_DIR = DATA_DIR / "fsl105_thumbs"
FSL105_LABEL_MAP_PATH = MODELS_DIR / "fsl105_label_map.json"
FSL105_MODEL_PATH = MODELS_DIR / "fsl105_model.pt"
FSL105_SCALER_PATH = MODELS_DIR / "fsl105_scaler.joblib"
FSL105_SVM_PATH = MODELS_DIR / "fsl105_svm_baseline.joblib"

FSL105_NUM_CLASSES = 105
FSL105_SEQ_LEN = 40
FSL105_FEATURE_DIM = 293  # 15*3 pose + 21*3 lh_body + 21*3 rh_body + 20*3 lh_local + 20*3 rh_local + 2 flags

PHRASE_CONFIDENCE_THRESHOLD = 0.35
PHRASE_MIN_ACTIVE_FRAMES = 8
PHRASE_MAX_BUFFER_FRAMES = 48
PHRASE_IDLE_END_SECONDS = 0.55
PHRASE_COOLDOWN_SECONDS = 1.4

# Dataset split ratios (stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Camera & Preprocessing
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Post-Processing & Word Formation
CONFIDENCE_THRESHOLD = 0.35  # tau (matches PHRASE_CONFIDENCE_THRESHOLD)
SMOOTHING_WINDOW = 10        # frames for sliding-window majority vote
DWELL_TIME_SECONDS = 1.0     # steady hold duration before confirming a letter
SPACE_PAUSE_SECONDS = 2.0    # optional neutral/no-hand pause to insert space

# Training hyperparameters
MLP_HIDDEN_DIMS = (256, 128, 64)
MLP_DROPOUT = 0.30
MLP_BATCH_SIZE = 64
MLP_EPOCHS = 120
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_PATIENCE = 15
