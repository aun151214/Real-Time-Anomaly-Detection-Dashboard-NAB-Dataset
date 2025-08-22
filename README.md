# 🚨 Real-Time Anomaly Detection Dashboard (NAB)

This project is a from-scratch, portfolio-ready **real-time (simulated) anomaly detection dashboard** built on the **Numenta Anomaly Benchmark (NAB)** dataset.

## Features
- Load any NAB time series and (optionally) its ground-truth anomaly windows
- Simulate streaming with adjustable speed and window size
- Detect anomalies with Isolation Forest (easy to swap for PyOD models)
- Live dashboard via Streamlit with chart and anomaly counters
- Clean, extensible code structure

## Quickstart
```bash
# 1) Clone your repo, then set up env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2) Get NAB (either beside or inside this repo)
git clone https://github.com/numenta/NAB.git

# 3) Run the dashboard
streamlit run app.py
