# LUDB adapter for ecgclip

This adapter keeps the existing ecgclip model and training scripts unchanged. It prepares LUDB into the same interface they already consume:

- `WORK_DIR/labels.csv`, indexed by `ecg_id`
- `WORK_DIR/images/<ecg_id>.png`
- folds 1–8 for train, 9 for validation, and 10 for test
- one binary column per class in `config.CLASSES`

## 1. Copy files

Copy `config.py` over the repository's current `config.py`, and add `prepare_ludb.py` to the repository root. Add the line from `requirements_ludb.txt` to the existing `requirements.txt`.

## 2. Download LUDB

```bash
mkdir -p data/ludb
wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/
```

Depending on how `wget` lays out the folders, set `DATA_DIR` to the directory that directly contains:

```text
ludb.csv
RECORDS
data/1.hea
data/1.dat
...
```

## 3. Prepare metadata and images

```bash
export ECG_DATASET=ludb
export DATA_DIR=/absolute/path/to/ludb/1.0.1
export WORK_DIR=./work_ludb

pip install -r requirements.txt
python prepare_ludb.py --limit 40   # smoke test
python prepare_ludb.py              # all 200 records
```

For metadata only:

```bash
python prepare_ludb.py --no-render
```

## 4. Run the existing pipeline

```bash
python zero_shot_eval.py --task multi
python extract_features.py
python linear_probe.py
python finetune_clip.py --caption label
python zero_shot_eval.py --task multi --ckpt work_ludb/checkpoints/biomedclip_ft.pt
```

All downstream scripts continue to import `config.py`, so keep `ECG_DATASET=ludb`, `DATA_DIR`, and `WORK_DIR` set in the same shell or tmux session.

## Label definition

LUDB does not provide PTB-XL's five SCP diagnostic superclasses. This adapter defines a new record-level multi-label task from the official `ludb.csv` columns:

- `ARR`: non-sinus-family rhythm or extrasystoles
- `CD`: conduction abnormality or cardiac pacing
- `HYP`: hypertrophy or chamber overload
- `ISCHEMIA`: ischemia, STEMI, NSTEMI-like, or scar entry
- `REPOL`: non-specific or early repolarization abnormality
- `NORM`: none of the five abnormal categories above

This mapping is derived for compatibility with ecgclip and should be documented as such in experiments.

## Important distinction from med-ts-llm

The linked `med-ts-llm/datasets/ludb.py` does not read the raw PhysioNet release. It reads project-specific preprocessed `train.csv`/`test.csv` files, treats each ECG lead as an individual clip, and uses P/QRS/T delineation labels for semantic segmentation. The adapter here instead keeps ecgclip's existing record-level, 12-lead image-classification design.
