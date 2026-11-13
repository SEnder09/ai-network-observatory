# AI Network Observatory

A small Streamlit portfolio prototype for exploring enterprise campus network monitoring and AI-assisted operations.

## Current status

**Step 1:** Interactive dashboard using synthetic telemetry for five scenarios:
- Normal Network
- Network Congestion
- Packet Loss
- High CPU Usage
- Device Offline

The app does not connect to real Huawei or enterprise network equipment. AI anomaly detection and diagnostic recommendations are planned for the next development step.

## Run locally (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Technologies

Python, Streamlit, Pandas, NumPy, Plotly, Scikit-learn (planned for the AI detection stage).

## Huawei product concepts referenced

- NetEngine: routing
- HiSecEngine: network security
- CloudEngine: switching
- AirEngine: wireless access

These are conceptual product mappings for learning and are not tested integrations or configurations.
