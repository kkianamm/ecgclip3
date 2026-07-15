"""
Central configuration for BiomedCLIP experiments on PTB-XL or LUDB.

Select the dataset with:
    export ECG_DATASET=ptbxl   # default
or:
    export ECG_DATASET=ludb

DATA_DIR and WORK_DIR can still be overridden with environment variables.
"""
from __future__ import annotations

import os


DATASET = os.environ.get("ECG_DATASET", "ptbxl").strip().lower()
if DATASET not in {"ptbxl", "ludb"}:
    raise ValueError(
        f"Unsupported ECG_DATASET={DATASET!r}. Expected 'ptbxl' or 'ludb'."
    )

# ----------------------------------------------------------------------------
# Dataset-specific paths, labels, and signal settings
# ----------------------------------------------------------------------------
if DATASET == "ptbxl":
    DEFAULT_DATA_DIR = "/lambda/nfs/Kiana2/ecgclip/data/ptbxl"
    DEFAULT_WORK_DIR = "./work"

    SAMPLING_RATE = int(os.environ.get("SAMPLING_RATE", "100"))
    if SAMPLING_RATE not in {100, 500}:
        raise ValueError("PTB-XL SAMPLING_RATE must be 100 or 500")
    FILENAME_COL = "filename_lr" if SAMPLING_RATE == 100 else "filename_hr"

    CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
    CLASS_DESCRIPTIONS = {
        "NORM": "normal ECG",
        "MI": "myocardial infarction",
        "STTC": "ST/T wave change",
        "CD": "conduction disturbance",
        "HYP": "cardiac hypertrophy",
    }

else:  # LUDB
    # This folder should contain ludb.csv, RECORDS, and data/1.hea ... data/200.dat.
    DEFAULT_DATA_DIR = "/lambda/nfs/Kiana2/ecgclip/data/ludb/1.0.1"
    DEFAULT_WORK_DIR = "./work_ludb"

    SAMPLING_RATE = 500
    FILENAME_COL = "record"

    # These are broad, derived record-level categories for using LUDB with the
    # existing multi-label classification pipeline. They are not official
    # LUDB superclasses. See prepare_ludb.py for the exact mapping.
    CLASSES = ["NORM", "ARR", "CD", "HYP", "ISCHEMIA", "REPOL"]
    CLASS_DESCRIPTIONS = {
        "NORM": "normal ECG without a listed abnormality",
        "ARR": "cardiac arrhythmia or ectopic beats",
        "CD": "cardiac conduction disturbance or pacing",
        "HYP": "cardiac chamber hypertrophy or overload",
        "ISCHEMIA": "myocardial ischemia, infarction, or scar pattern",
        "REPOL": "non-specific repolarization abnormality",
    }

DATA_DIR = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)
WORK_DIR = os.environ.get("WORK_DIR", DEFAULT_WORK_DIR)
IMG_DIR = os.path.join(WORK_DIR, "images")
FEAT_DIR = os.path.join(WORK_DIR, "features")
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")

for _directory in (WORK_DIR, IMG_DIR, FEAT_DIR, CKPT_DIR):
    os.makedirs(_directory, exist_ok=True)

# Both datasets use folds 1-8 for training, 9 for validation, and 10 for test.
# PTB-XL provides official folds. prepare_ludb.py creates reproducible
# multi-label-stratified folds for LUDB.
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10

# ----------------------------------------------------------------------------
# Zero-shot prompt engineering
# ----------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    "this is a photo of {}",
    "an electrocardiogram showing {}",
    "a 12-lead ECG with {}",
    "ECG tracing consistent with {}",
]

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CONTEXT_LENGTH = 256

# ----------------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4

LP_EPOCHS = 50
LP_LR = 1e-3
LP_WEIGHT_DECAY = 1e-4

FT_EPOCHS = 5
FT_LR = 1e-5
FT_WEIGHT_DECAY = 0.1
FT_FREEZE_TEXT = True
