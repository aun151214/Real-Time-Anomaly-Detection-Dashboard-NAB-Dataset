
# Real-Time Anomaly Detection Dashboard (NAB)

This is a from-scratch, portfolio-ready real-time (simulated) anomaly detection dashboard built on the Numenta Anomaly Benchmark (NAB) dataset.

## Quickstart
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
git clone https://github.com/numenta/NAB.git
streamlit run app.py
```

## References
- NAB: https://github.com/numenta/NAB
- NAB paper: https://arxiv.org/abs/1510.03336
- Streamlit docs: https://docs.streamlit.io/
- scikit-learn IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- PyOD docs: https://pyod.readthedocs.io/
