
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import pandas as pd

def load_nab_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    val_col = "value" if "value" in df.columns else df.columns[1]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    df = df.rename(columns={val_col: "value"})
    return df[["value"]]

def load_combined_windows(labels_json: Path) -> dict:
    with open(labels_json, "r") as f:
        windows = json.load(f)
    return windows

def windows_for_series(rel_path_in_nab_data: str, windows_dict: dict) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ranges = windows_dict.get(rel_path_in_nab_data, [])
    return [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in ranges]

def label_series_with_windows(df: pd.DataFrame, windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    if not windows:
        return pd.Series(False, index=df.index, name="is_anomaly_gt")
    mask = pd.Series(False, index=df.index)
    for start, end in windows:
        mask |= (df.index >= start) & (df.index <= end)
    mask.name = "is_anomaly_gt"
    return mask
