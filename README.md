
# AI Network Observatory

An AI-powered enterprise network monitoring and anomaly detection dashboard built with Python and Streamlit.

## Overview

AI Network Observatory is a portfolio prototype demonstrating how AI-assisted analytics can support enterprise network monitoring and incident management.

The dashboard uses simulated network telemetry to demonstrate anomaly detection, network health monitoring, fault diagnosis and incident tracking.

## Key Features

- **Network Monitoring:** Visualise latency, packet loss, bandwidth utilisation and device CPU usage.
- **AI Anomaly Detection:** Apply the Isolation Forest algorithm to identify unusual network behaviour.
- **Fault Diagnosis:** Use rule-based analysis to suggest potential network issues.
- **Incident Management:** Create, review and update network incident records.
- **Network Topology:** Visualise a conceptual enterprise network topology and simulated device status.
- **Telemetry Export:** Download monitoring data as CSV.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## Machine Learning

The project uses the Isolation Forest algorithm from Scikit-learn to identify observations that differ from a simulated normal network baseline.

## Demo Scenarios

- Normal Network
- Network Congestion
- Packet Loss
- High CPU Usage
- Device Offline

## Run Locally

Clone the repository:

```bash
git clone https://github.com/SEnder09/ai-network-observatory.git
cd ai-network-observatory
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Demo Limitations

This is an independent portfolio prototype.

- All network telemetry is synthetic and generated for demonstration.
- No real enterprise network or Huawei equipment is connected.
- Huawei product names in the topology are illustrative references only.
- Anomaly detection results are experimental and should not be treated as confirmed network faults.
- Incident records are stored in Streamlit session state and are not persistent.

## Author

Developed as an independent project to explore AI-assisted network monitoring, anomaly detection and network operations.
