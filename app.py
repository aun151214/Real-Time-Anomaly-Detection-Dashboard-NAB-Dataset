import time
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics import precision_score, recall_score, f1_score

from src.data_loader import (
    load_nab_series,
    load_combined_windows,
    windows_for_series,
    label_series_with_windows,
)
from src.models import IFDetector
from src.streaming import stream_series

st.set_page_config(page_title="Real-Time Anomaly Detection (NAB)", layout="wide")
st.title("🚨 Real-Time Anomaly Detection Dashboard (NAB)")

with st.sidebar:
    st.header("Settings")
    nab_root = st.text_input(
        "Path to NAB root", value="./NAB", help="Folder that contains the official NAB checkout"
    )
    rel_csv = st.text_input(
        "Relative CSV in NAB/data/",
        value="realKnownCause/machine_temperature_system_failure.csv",
    )
    warmup = st.number_input(
        "Warm-up points (train only on presumed normal)",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
    )
    contamination = st.slider(
        "Contamination (IF)", min_value=0.001, max_value=0.1, value=0.01, step=0.001
    )
    batch_size = st.slider("Batch size", min_value=1, max_value=50, value=5, step=1)
    delay_ms = st.slider(
        "Delay per batch (ms)", min_value=0, max_value=500, value=20, step=5
    )
    use_gt = st.checkbox("Show ground-truth windows (if labels found)", value=True)

# Resolve paths
nab_root = Path(nab_root).expanduser()
csv_path = nab_root / "data" / rel_csv
labels_path = nab_root / "labels" / "combined_windows.json"

# Load data
if not csv_path.exists():
    st.error(f"CSV not found: {csv_path}. Please set a valid path.")
    st.stop()

df = load_nab_series(csv_path)

# Ensure index is unique (some NAB files have duplicate timestamps)
df = df[~df.index.duplicated(keep="first")]

# Load windows if available
windows = []
if use_gt and labels_path.exists():
    try:
        wdict = load_combined_windows(labels_path)
        windows = windows_for_series(rel_csv, wdict)
    except Exception as e:
        st.warning(f"Could not load labels: {e}")

# Train on warm-up window
if warmup >= len(df):
    st.warning("Warm-up larger than series length; decreasing warm-up.")
    warmup = max(50, len(df) // 5)

y_warm = df.iloc[:warmup]["value"]
detector = IFDetector(contamination=contamination).fit(y_warm)

# Prepare UI
left, right = st.columns([3, 1])
chart_placeholder = left.empty()
metrics_placeholder = right.empty()
table_placeholder = left.empty()

# Streaming loop
buffer = df.iloc[:warmup].copy()
preds = pd.Series(dtype=int)

# Optional GT labels
gt_series = label_series_with_windows(df, windows) if windows else pd.Series(False, index=df.index)

anomaly_count = 0
rows_shown = warmup

# Initialize chart with warm-up
chart_df = buffer.copy()
chart_df["pred"] = 0
chart_df["gt"] = gt_series.loc[chart_df.index].astype(int) if not gt_series.empty else 0

for idx, vals in stream_series(
    df.iloc[warmup:], delay_sec=delay_ms / 1000.0, batch_size=batch_size
):
    new_chunk = pd.DataFrame({"value": vals.values}, index=idx)

    # Predict on new chunk
    chunk_preds = detector.predict_labels(new_chunk["value"])

    # Update preds safely (fix FutureWarning)
    if preds.empty:
        preds = chunk_preds
    else:
        preds = pd.concat([preds, chunk_preds])

    chart_df = pd.concat([chart_df, new_chunk])
    rows_shown += len(new_chunk)

    # Update anomaly counts
    anomaly_count += int((chunk_preds == 1).sum())

    # Build chart with anomalies highlighted
    chart_data = chart_df.copy()
    chart_data["pred"] = preds.reindex(chart_df.index).fillna(0).astype(int)
    chart_data["gt"] = gt_series.reindex(chart_df.index).fillna(0).astype(int)

    line = alt.Chart(chart_data.reset_index()).mark_line().encode(
        x="index:T", y="value:Q"
    )

    anomalies = (
        alt.Chart(chart_data.reset_index())
        .mark_circle(size=60, color="red")
        .encode(x="index:T", y="value:Q")
        .transform_filter("datum.pred == 1")
    )

    chart_placeholder.altair_chart(line + anomalies, use_container_width=True)

    # ✅ Calculate metrics live (aligned indexes only)
    if not gt_series.empty and not preds.empty:
        common_index = gt_series.index.intersection(preds.index)
        if len(common_index) > 0:
            y_true = gt_series.loc[common_index].astype(int)
            y_pred = preds.loc[common_index].astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            precision = recall = f1 = 0.0
    else:
        precision = recall = f1 = 0.0

    with metrics_placeholder.container():
        st.metric("Points streamed", rows_shown)
        st.metric("Anomalies flagged", anomaly_count)
        st.metric("Series length", len(df))
        st.write(f"**Precision:** {precision:.3f}")
        st.write(f"**Recall:** {recall:.3f}")
        st.write(f"**F1 Score:** {f1:.3f}")

    # Recent table with highlighting
    recent = pd.concat(
        [
            chart_df[["value"]].tail(50),
            preds.reindex(chart_df.index).rename("pred").tail(50).fillna(0).astype(int),
            (
                gt_series.reindex(chart_df.index).tail(50).astype(int)
                if not gt_series.empty
                else pd.Series(0, index=chart_df.tail(50).index, name="gt")
            ),
        ],
        axis=1,
    )

    def highlight_anomalies(row):
        color = "background-color: salmon" if row["pred"] == 1 else ""
        return [color] * len(row)

    table_placeholder.dataframe(recent.style.apply(highlight_anomalies, axis=1))

st.success("Streaming complete.")
