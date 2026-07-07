"""
MIMIC-III data loader for ICU mortality prediction.

Requires credentialed MIMIC-III access. Set MIMIC_PATH to the directory
containing CHARTEVENTS.csv, ADMISSIONS.csv, ICUSTAYS.csv, PATIENTS.csv.

ITEMID reference (CareVue + MetaVision):
  Heart Rate       : 211, 220045
  SpO2             : 646, 220277
  Systolic BP      : 51, 442, 455, 6701, 220179, 220050
  Diastolic BP     : 8368, 8440, 8441, 8555, 220180, 220051
  Mean BP          : 456, 52, 6702, 443, 220052, 220181
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MIMIC_PATH = os.environ.get("MIMIC_PATH", "")

VITALS_ITEMIDS = {
    "hr":   [211, 220045],
    "spo2": [646, 220277],
    "sbp":  [51, 442, 455, 6701, 220179, 220050],
    "dbp":  [8368, 8440, 8441, 8555, 220180, 220051],
    "mbp":  [456, 52, 6702, 443, 220052, 220181],
}

VITAL_COLS = list(VITALS_ITEMIDS.keys())

# Physiological plausibility bounds for outlier removal
VITAL_BOUNDS = {
    "hr":   (0, 300),
    "spo2": (50, 100),
    "sbp":  (0, 300),
    "dbp":  (0, 200),
    "mbp":  (0, 250),
}

WINDOW_HOURS = 48
FREQ = "1h"


def load_mimic(mimic_path: str = MIMIC_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X has shape (N, 48, num_features) and y in {0,1}."""
    mimic_path = Path(mimic_path)
    if not mimic_path.exists():
        raise FileNotFoundError(
            f"MIMIC path '{mimic_path}' not found. "
            "Set MIMIC_PATH env var or pass mimic_path argument."
        )

    print("Loading ADMISSIONS …")
    adm = pd.read_csv(mimic_path / "ADMISSIONS.csv", usecols=[
        "HADM_ID", "SUBJECT_ID", "HOSPITAL_EXPIRE_FLAG",
        "ADMITTIME", "DISCHTIME",
    ], parse_dates=["ADMITTIME", "DISCHTIME"])
    adm.columns = adm.columns.str.lower()

    print("Loading ICUSTAYS …")
    icu = pd.read_csv(mimic_path / "ICUSTAYS.csv", usecols=[
        "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS",
    ], parse_dates=["INTIME", "OUTTIME"])
    icu.columns = icu.columns.str.lower()

    # Keep first ICU stay per admission; require ≥ 48 h LOS
    icu = icu.sort_values(["hadm_id", "intime"]).drop_duplicates("hadm_id", keep="first")
    icu = icu[icu["los"] >= 2.0]

    stays = icu.merge(adm[["hadm_id", "hospital_expire_flag"]], on="hadm_id")

    all_item_ids = [iid for ids in VITALS_ITEMIDS.values() for iid in ids]

    print("Loading CHARTEVENTS (large file, may take minutes) …")
    chunks = pd.read_csv(
        mimic_path / "CHARTEVENTS.csv",
        usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM", "ERROR"],
        parse_dates=["CHARTTIME"],
        chunksize=500_000,
        low_memory=False,
    )

    keep_stays = set(stays["icustay_id"].values)
    frames = []
    for chunk in chunks:
        chunk.columns = chunk.columns.str.lower()
        chunk = chunk[
            (chunk["icustay_id"].isin(keep_stays)) &
            (chunk["itemid"].isin(all_item_ids)) &
            (chunk["error"].isna() | (chunk["error"] == 0)) &
            chunk["valuenum"].notna()
        ]
        frames.append(chunk)
    charts = pd.concat(frames, ignore_index=True)

    # Map itemid → vital name
    itemid_to_vital: dict[int, str] = {}
    for name, ids in VITALS_ITEMIDS.items():
        for iid in ids:
            itemid_to_vital[iid] = name
    charts["vital"] = charts["itemid"].map(itemid_to_vital)

    print("Building 48-hour windows …")
    X_list, y_list = [], []
    stays_indexed = stays.set_index("icustay_id")

    for icustay_id, group in charts.groupby("icustay_id"):
        if icustay_id not in stays_indexed.index:
            continue
        row = stays_indexed.loc[icustay_id]
        intime = row["intime"]
        label  = int(row["hospital_expire_flag"])

        window_end = intime + pd.Timedelta(hours=WINDOW_HOURS)
        group = group[(group["charttime"] >= intime) & (group["charttime"] < window_end)].copy()
        group["hours"] = (group["charttime"] - intime).dt.total_seconds() / 3600

        hourly_idx = pd.RangeIndex(WINDOW_HOURS)
        ts = pd.DataFrame(index=hourly_idx, columns=VITAL_COLS, dtype=float)

        for vital, vgroup in group.groupby("vital"):
            if vital not in VITAL_BOUNDS:
                continue
            lo, hi = VITAL_BOUNDS[vital]
            vgroup = vgroup[(vgroup["valuenum"] >= lo) & (vgroup["valuenum"] <= hi)]
            vgroup = vgroup.copy()
            vgroup["hour_bin"] = vgroup["hours"].astype(int).clip(0, WINDOW_HOURS - 1)
            hourly = vgroup.groupby("hour_bin")["valuenum"].mean()
            ts[vital] = hourly

        # Forward-fill then back-fill, then global median imputation
        ts = ts.ffill().bfill()
        for col in VITAL_COLS:
            if ts[col].isna().any():
                ts[col] = ts[col].fillna(ts[col].median() if ts[col].notna().any() else 0)

        X_list.append(ts.values.astype(np.float32))
        y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    print(f"Dataset: {X.shape[0]} stays, {y.mean():.2%} mortality")
    return X, y


def normalise(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """Z-score per feature using train statistics."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std
