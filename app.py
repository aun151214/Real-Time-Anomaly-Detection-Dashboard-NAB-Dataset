
import time
from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_loader import load_nab_series, load_combined_windows, windows_for_series, label_series_with_windows
from src.models import IFDetector
from src.streaming import stream_series

st.set_page_config(page_title="Real-Time Anomaly Detection (NAB)", layout="wide")
st.title("🚨 Real-Time Anomaly Detection Dashboard (NAB)")

with st.sidebar:
    st.header("Settings")
    nab_root = st.text_input("Path to NAB root", value="../NAB", help="Folder that contains the official NAB checkout")
    rel_csv = st.text_input("Relative CSV in NAB/data/", value="realKnownCause/machine_temperature_system_failure.csv")
    warmup = st.number_input("Warm-up points (train on presumed normal)", min_value=50, max_value=5000, value=500, step=50)
    contamination = st.slider("Contamination (IF)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    batch_size = st.slider("Batch size", min_value=1, max_value=50, value=5, step=1)
    delay_ms = st.slider("Delay per batch (ms)", min_value=0, max_value=500, value=20, step=5)
    use_gt = st.checkbox("Show ground-truth windows (if labels found)", value=True)

nab_root = Path(nab_root).expanduser()
csv_path = nab_root / "data" / rel_csv
labels_path = nab_root / "labels" / "combined_windows.json"

if not csv_path.exists():
    st.error(f"CSV not found: {csv_path}. Please set a valid path.")
    st.stop()

df = load_nab_series(csv_path)

windows = []
if use_gt and labels_path.exists():
    wdict = load_combined_windows(labels_path)
    windows = windows_for_series(rel_csv, wdict)

if warmup >= len(df):
    warmup = max(50, len(df) // 5)
y_warm = df.iloc[:warmup]["value"]
detector = IFDetector(contamination=contamination).fit(y_warm)

left, right = st.columns([3, 1])
line_placeholder = left.empty()
metrics_placeholder = right.empty()
table_placeholder = left.empty()

buffer = df.iloc[:warmup].copy()
scores = pd.Series(dtype=float)
preds = pd.Series(dtype=int)

gt_series = label_series_with_windows(df, windows) if windows else pd.Series(False, index=df.index)

anomaly_count = 0
rows_shown = warmup

chart_df = buffer.copy()
chart_df["pred"] = 0
chart_df["gt"] = gt_series.loc[chart_df.index].astype(int) if not gt_series.empty else 0
line_placeholder.line_chart(chart_df[["value"]])

for idx, vals in stream_series(df.iloc[warmup:], delay_sec=delay_ms / 1000.0, batch_size=batch_size):
    new_chunk = pd.DataFrame({"value": vals.values}, index=idx)
    chunk_scores = detector.predict_scores(new_chunk["value"])
    chunk_preds = detector.predict_labels(new_chunk["value"])

    scores = pd.concat([scores, chunk_scores])
    preds = pd.concat([preds, chunk_preds])

    chart_df = pd.concat([chart_df, new_chunk])
    rows_shown += len(new_chunk)

    anomaly_count += int((chunk_preds == 1).sum())

    line_placeholder.line_chart(chart_df[["value"]])
    with metrics_placeholder.container():
        st.metric("Points streamed", rows_shown)
        st.metric("Anomalies flagged", anomaly_count)
        st.metric("Series length", len(df))

    recent = pd.concat([
        chart_df[["value"]].tail(50),
        preds.reindex(chart_df.index).rename("pred").tail(50).fillna(0).astype(int),
        (gt_series.reindex(chart_df.index).tail(50).astype(int) if not gt_series.empty else pd.Series(0, index=chart_df.tail(50).index, name="gt"))
    ], axis=1)
    table_placeholder.dataframe(recent)

st.success("Streaming complete.")
