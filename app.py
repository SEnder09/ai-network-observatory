import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
import uuid
from datetime import datetime

st.set_page_config(
    page_title="AI Network Observatory",
    page_icon="🌐",
    layout="wide",
)

# ==========================================
# STEP 6: PROFESSIONAL DASHBOARD STYLING
# ==========================================


st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #0F172A;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# PROFESSIONAL DASHBOARD HEADER
# ==========================================

header_left, header_right = st.columns(
    [3, 1]
)

with header_left:
    st.title("🌐 AI Network Observatory")
    st.caption(
        "AI-powered Enterprise Network Monitoring "
        "& Anomaly Detection Platform"
    )

with header_right:
    st.markdown(
        """
        <div style="
            background-color:#172033;
            border:1px solid #334155;
            padding:12px;
            border-radius:10px;
            text-align:center;
            margin-top:15px;
        ">
            <div style="color:#94A3B8;font-size:12px;">
                SYSTEM ENVIRONMENT
            </div>
            <div style="
                color:#22C55E;
                font-weight:bold;
                font-size:15px;
            ">
                ● DEMO MODE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

st.info(
    "DEMO MODE — all network measurements shown here are synthetic. "
    "This prototype does not connect to real Huawei or enterprise network devices."
)

SCENARIOS = [
    "Normal Network",
    "Network Congestion",
    "Packet Loss",
    "High CPU Usage",
    "Device Offline",
]

scenario = st.sidebar.selectbox("Select a network scenario", SCENARIOS)

# Dashboard time range filter
time_range = st.sidebar.selectbox(
    "Analysis Time Range",
    [
        "Last 15 minutes",
        "Last 30 minutes",
        "Last 60 minutes",
    ],
    index=2,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard Modules**")
st.sidebar.caption(
    "1. Network Monitoring\n\n"
    "2. AI Anomaly Detection\n\n"
    "3. Fault Diagnosis\n\n"
    "4. Incident Management\n\n"
    "5. Network Topology"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Prototype scope**")
st.sidebar.caption("Step 1: simulated network metrics and dashboard. AI detection and diagnosis will be added next.")


def generate_demo_data(selected_scenario: str, points: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=points, freq="min")

    latency = np.clip(rng.normal(25, 3, points), 5, None)
    packet_loss = np.clip(rng.normal(0.2, 0.08, points), 0, None)
    bandwidth = np.clip(rng.normal(55, 5, points), 0, 100)
    cpu = np.clip(rng.normal(38, 5, points), 0, 100)

    if selected_scenario == "Network Congestion":
        latency[-15:] += np.linspace(0, 115, 15)
        bandwidth[-15:] += np.linspace(0, 40, 15)
        cpu[-15:] += np.linspace(0, 28, 15)
    elif selected_scenario == "Packet Loss":
        packet_loss[-12:] += np.linspace(0, 12, 12)
        latency[-12:] += np.linspace(0, 45, 12)
    elif selected_scenario == "High CPU Usage":
        cpu[-12:] += np.linspace(0, 58, 12)
        latency[-12:] += np.linspace(0, 28, 12)
    elif selected_scenario == "Device Offline":
        latency[-5:] = np.nan
        packet_loss[-5:] = np.nan
        bandwidth[-5:] = np.nan
        cpu[-5:] = np.nan

    return pd.DataFrame(
        {
            "Time": timestamps,
            "Latency (ms)": latency,
            "Packet Loss (%)": packet_loss,
            "Bandwidth Utilisation (%)": np.clip(bandwidth, 0, 100),
            "Device CPU (%)": np.clip(cpu, 0, 100),
        }
    )


FEATURE_COLUMNS = [
    "Latency (ms)",
    "Packet Loss (%)",
    "Bandwidth Utilisation (%)",
    "Device CPU (%)",
]


@st.cache_resource
def train_anomaly_model():
    """
    Train an Isolation Forest model using
    simulated normal network telemetry.
    """
    rng = np.random.default_rng(2026)

    baseline = pd.DataFrame({
        "Latency (ms)": np.clip(
            rng.normal(25, 3, 1000), 5, None
        ),
        "Packet Loss (%)": np.clip(
            rng.normal(0.2, 0.08, 1000), 0, None
        ),
        "Bandwidth Utilisation (%)": np.clip(
            rng.normal(55, 5, 1000), 0, 100
        ),
        "Device CPU (%)": np.clip(
            rng.normal(38, 5, 1000), 0, 100
        ),
    })

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )

    model.fit(baseline[FEATURE_COLUMNS])

    return model


def detect_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect unusual network telemetry.
    Missing telemetry is handled separately.
    """
    result = data.copy()

    result["AI Anomaly"] = False
    result["Anomaly Score"] = np.nan

    # Only analyse rows with complete measurements.
    valid_mask = result[FEATURE_COLUMNS].notna().all(axis=1)

    if valid_mask.any():
        model = train_anomaly_model()

        valid_data = result.loc[
            valid_mask, FEATURE_COLUMNS
        ]

        predictions = model.predict(valid_data)

        # Higher score means more unusual.
        scores = -model.decision_function(valid_data)

        result.loc[valid_mask, "AI Anomaly"] = (
            predictions == -1
        )

        result.loc[valid_mask, "Anomaly Score"] = scores

    return result


def diagnose_network(data: pd.DataFrame):
    """Generate rule-based network diagnosis."""
    latest = data.iloc[-1]

    # Rule 1: Missing telemetry
    if latest[FEATURE_COLUMNS].isna().any():
        return {
            "severity": "Critical",
            "issue": "Possible Device Offline",
            "cause": "Network device is not reporting telemetry.",
            "recommendation": (
                "Check device power, network connectivity, "
                "management interface and telemetry connection."
            ),
        }

    latency = latest["Latency (ms)"]
    packet_loss = latest["Packet Loss (%)"]
    bandwidth = latest["Bandwidth Utilisation (%)"]
    cpu = latest["Device CPU (%)"]

    # Rule 2: Network congestion
    if latency > 80 and bandwidth > 85:
        return {
            "severity": "High",
            "issue": "Possible Network Congestion",
            "cause": (
                "High latency combined with high bandwidth utilisation."
            ),
            "recommendation": (
                "Inspect traffic utilisation, identify bandwidth-heavy "
                "applications and consider traffic optimisation or "
                "capacity expansion."
            ),
        }

    # Rule 3: Packet loss
    if packet_loss > 2:
        return {
            "severity": "High",
            "issue": "Possible Packet Loss",
            "cause": (
                "Packet loss exceeds the configured 2% threshold."
            ),
            "recommendation": (
                "Check interface errors, cable connections, "
                "wireless signal quality and network congestion."
            ),
        }

    # Rule 4: High CPU
    if cpu > 85:
        return {
            "severity": "High",
            "issue": "Possible Device Resource Overload",
            "cause": "Device CPU utilisation exceeds 85%.",
            "recommendation": (
                "Inspect device processes, traffic load and "
                "resource utilisation."
            ),
        }

    # Rule 5: Anomaly detected without a specific rule
    if bool(latest["AI Anomaly"]):
        return {
            "severity": "Medium",
            "issue": "Unusual Network Behaviour",
            "cause": (
                "Isolation Forest detected an unusual telemetry pattern, "
                "but no specific diagnostic rule was triggered."
            ),
            "recommendation": (
                "Review recent telemetry trends and investigate "
                "the affected network device."
            ),
        }

    return {
        "severity": "Normal",
        "issue": "No Significant Network Issue",
        "cause": "Current telemetry is within configured thresholds.",
        "recommendation": (
            "Continue monitoring network performance."
        ),
    }


# ==========================================
# STEP 5: INTERACTIVE NETWORK TOPOLOGY
# ==========================================

def get_network_devices(selected_scenario):
    """
    Generate simulated device health based on
    the selected network scenario.
    """

    devices = [
        {
            "id": "internet",
            "name": "Internet",
            "product": "External Network",
            "type": "Gateway",
            "status": "Online",
            "x": 0,
            "y": 2,
        },
        {
            "id": "router",
            "name": "NetEngine Router",
            "product": "Huawei NetEngine",
            "type": "Router",
            "status": "Online",
            "x": 1.5,
            "y": 2,
        },
        {
            "id": "firewall",
            "name": "HiSecEngine Firewall",
            "product": "Huawei HiSecEngine",
            "type": "Firewall",
            "status": "Online",
            "x": 3,
            "y": 2,
        },
        {
            "id": "core",
            "name": "CloudEngine Core",
            "product": "Huawei CloudEngine",
            "type": "Core Switch",
            "status": "Online",
            "x": 4.5,
            "y": 2,
        },
        {
            "id": "access",
            "name": "Access Switch",
            "product": "Enterprise Switch",
            "type": "Access Switch",
            "status": "Online",
            "x": 6,
            "y": 2,
        },
        {
            "id": "ap",
            "name": "AirEngine Wi-Fi AP",
            "product": "Huawei AirEngine",
            "type": "Wireless Access Point",
            "status": "Online",
            "x": 7.5,
            "y": 2,
        },
        {
            "id": "users",
            "name": "User Devices",
            "product": "Client Devices",
            "type": "End Devices",
            "status": "Online",
            "x": 9,
            "y": 2,
        },
    ]

    # Simulated fault scenarios
    if selected_scenario == "Network Congestion":
        devices[3]["status"] = "Warning"
        devices[4]["status"] = "Warning"

    elif selected_scenario == "Packet Loss":
        devices[5]["status"] = "Warning"

    elif selected_scenario == "High CPU Usage":
        devices[1]["status"] = "Warning"

    elif selected_scenario == "Device Offline":
        devices[5]["status"] = "Offline"
        devices[6]["status"] = "Disconnected"

    return devices


def create_network_topology(devices):
    """
    Draw an interactive network topology using Plotly.
    """

    status_colors = {
        "Online": "#22C55E",
        "Warning": "#F59E0B",
        "Offline": "#EF4444",
        "Disconnected": "#94A3B8",
    }

    fig = go.Figure()

    # Connections between network devices
    connections = [
        ("internet", "router"),
        ("router", "firewall"),
        ("firewall", "core"),
        ("core", "access"),
        ("access", "ap"),
        ("ap", "users"),
    ]

    device_lookup = {
        device["id"]: device for device in devices
    }

    # Draw network links
    for source_id, target_id in connections:

        source = device_lookup[source_id]
        target = device_lookup[target_id]

        source_status = source["status"]
        target_status = target["status"]

        if (
            source_status == "Offline"
            or target_status == "Offline"
            or source_status == "Disconnected"
            or target_status == "Disconnected"
        ):
            line_color = "#94A3B8"
            line_dash = "dash"

        elif (
            source_status == "Warning"
            or target_status == "Warning"
        ):
            line_color = "#F59E0B"
            line_dash = "solid"

        else:
            line_color = "#22C55E"
            line_dash = "solid"

        fig.add_trace(
            go.Scatter(
                x=[source["x"], target["x"]],
                y=[source["y"], target["y"]],
                mode="lines",
                line=dict(
                    color=line_color,
                    width=4,
                    dash=line_dash,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Draw network devices
    for status in status_colors:

        group = [
            device for device in devices
            if device["status"] == status
        ]

        if not group:
            continue

        fig.add_trace(
            go.Scatter(
                x=[device["x"] for device in group],
                y=[device["y"] for device in group],
                mode="markers+text",
                name=status,
                text=[
                    device["name"] for device in group
                ],
                textposition="bottom center",
                textfont=dict(size=11),
                marker=dict(
                    size=30,
                    color=status_colors[status],
                    line=dict(
                        color="white",
                        width=2,
                    ),
                ),
                customdata=[
                    [
                        device["type"],
                        device["product"],
                        device["status"],
                    ]
                    for device in group
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Type: %{customdata[0]}<br>"
                    "Product mapping: %{customdata[1]}<br>"
                    "Status: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Enterprise Campus Network Topology",
        height=420,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            visible=False,
            range=[-0.8, 9.8],
        ),
        yaxis=dict(
            visible=False,
            range=[0.8, 3.2],
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=80, b=80),
        hovermode="closest",
    )

    return fig


def latest_value(series: pd.Series):
    valid = series.dropna()
    return None if valid.empty else float(valid.iloc[-1])


def display_value(value, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.1f}{suffix}"



df = detect_anomalies(
    generate_demo_data(scenario)
)

# Apply selected time range
time_range_points = {
    "Last 15 minutes": 15,
    "Last 30 minutes": 30,
    "Last 60 minutes": 60,
}

df = df.tail(
    time_range_points[time_range]
).copy()

latency_now = latest_value(df["Latency (ms)"])
loss_now = latest_value(df["Packet Loss (%)"])
bandwidth_now = latest_value(df["Bandwidth Utilisation (%)"])
cpu_now = latest_value(df["Device CPU (%)"])
device_status = "Offline" if scenario == "Device Offline" else "Online"

st.subheader("Network health overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Device status", device_status)
col2.metric("Latency", display_value(latency_now, " ms"))
col3.metric("Packet loss", display_value(loss_now, "%"))
col4.metric("Bandwidth", display_value(bandwidth_now, "%"))
col5.metric("Device CPU", display_value(cpu_now, "%"))

if scenario == "Normal Network":
    st.success("Scenario status: normal operating conditions (simulated).")
else:
    st.warning(f"Scenario status: {scenario} (simulated). Automated diagnosis is not implemented yet.")

st.subheader("AI anomaly detection")

anomaly_rows = df[df["AI Anomaly"]]
offline_rows = df[FEATURE_COLUMNS].isna().any(axis=1)

metric_a, metric_b, metric_c = st.columns(3)

metric_a.metric(
    "Detected anomalies",
    len(anomaly_rows),
)

metric_b.metric(
    "Analysed samples",
    int((~offline_rows).sum()),
)

metric_c.metric(
    "Missing telemetry",
    int(offline_rows.sum()),
)

if offline_rows.any():
    st.error(
        "Device telemetry is missing. "
        "The device may be offline or unreachable. "
        "This is a missing-data check, not an Isolation Forest prediction."
    )

if not anomaly_rows.empty:
    st.warning(
        f"AI detected {len(anomaly_rows)} unusual observations."
    )

    st.dataframe(
        anomaly_rows[
            ["Time"] + FEATURE_COLUMNS + ["Anomaly Score"]
        ].tail(10),
        use_container_width=True,
        hide_index=True,
    )
else:
    if not offline_rows.any():
        st.success(
            "No anomalies detected in the current sample."
        )

st.caption(
    "Isolation Forest is trained on synthetic normal baseline data. "
    "An anomaly indicates unusual behaviour, not a confirmed fault."
)


st.divider()

st.subheader("Network Fault Diagnosis")

diagnosis = diagnose_network(df)

severity = diagnosis["severity"]

if severity == "Critical":
    st.error(f"Severity: {severity}")
elif severity == "High":
    st.warning(f"Severity: {severity}")
elif severity == "Medium":
    st.info(f"Severity: {severity}")
else:
    st.success(f"Severity: {severity}")

st.markdown(f"### {diagnosis['issue']}")

st.markdown("**Possible Cause**")
st.write(diagnosis["cause"])

st.markdown("**Troubleshooting Recommendation**")
st.write(diagnosis["recommendation"])

st.caption(
    "Diagnosis is generated using predefined rules and synthetic data. "
    "Recommendations are indicative, not confirmed root causes."
)


# ==========================================
# STEP 4: INCIDENT MANAGEMENT DASHBOARD
# ==========================================

st.divider()
st.header("🚨 Incident Management")

# Initialise incident storage
if "incidents" not in st.session_state:
    st.session_state.incidents = []

# Create incident button
st.subheader("Create Network Incident")

if diagnosis["severity"] == "Normal":
    st.info(
        "No significant network issue detected. "
        "Incident creation is disabled."
    )

if st.button(
    "＋ Create Incident",
    type="primary",
    disabled=(diagnosis["severity"] == "Normal"),
    use_container_width=False,
):

    latest = df.iloc[-1]

    def format_metric(value, suffix):
        if pd.isna(value):
            return "N/A"
        return f"{value:.1f}{suffix}"

    new_incident = {
        "id": (
            f"INC-{datetime.now().strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex[:6].upper()}"
        ),
        "detected_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "scenario": scenario,
        "severity": diagnosis["severity"],
        "issue": diagnosis["issue"],
        "cause": diagnosis["cause"],
        "recommendation": diagnosis["recommendation"],
        "status": "Open",
        "anomalies": int(df["AI Anomaly"].sum()),
        "metrics": {
            "Latency": format_metric(
                latest["Latency (ms)"], " ms"
            ),
            "Packet Loss": format_metric(
                latest["Packet Loss (%)"], "%"
            ),
            "Bandwidth": format_metric(
                latest["Bandwidth Utilisation (%)"], "%"
            ),
            "CPU": format_metric(
                latest["Device CPU (%)"], "%"
            ),
        },
    }

    st.session_state.incidents.insert(
        0, new_incident
    )

    st.success(
        f"Incident {new_incident['id']} created successfully."
    )


# ==========================================
# INCIDENT SUMMARY
# ==========================================

st.subheader("Incident Overview")

incidents = st.session_state.incidents

total_incidents = len(incidents)

open_count = sum(
    item["status"] == "Open"
    for item in incidents
)

investigating_count = sum(
    item["status"] == "Investigating"
    for item in incidents
)

resolved_count = sum(
    item["status"] == "Resolved"
    for item in incidents
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Incidents", total_incidents)
c2.metric("Open", open_count)
c3.metric("Investigating", investigating_count)
c4.metric("Resolved", resolved_count)


# ==========================================
# INCIDENT FILTERS
# ==========================================

st.subheader("Incident Records")

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    severity_filter = st.selectbox(
        "Filter by Severity",
        [
            "All",
            "Critical",
            "High",
            "Medium",
            "Low",
        ],
    )

with filter_col2:
    status_filter = st.selectbox(
        "Filter by Status",
        [
            "All",
            "Open",
            "Investigating",
            "Resolved",
        ],
    )


filtered_incidents = incidents

if severity_filter != "All":
    filtered_incidents = [
        item for item in filtered_incidents
        if item["severity"] == severity_filter
    ]

if status_filter != "All":
    filtered_incidents = [
        item for item in filtered_incidents
        if item["status"] == status_filter
    ]


# ==========================================
# INCIDENT RECORD DISPLAY
# ==========================================

if not filtered_incidents:

    st.info(
        "No incidents found. "
        "Select a network scenario and create an incident."
    )

else:

    for incident in filtered_incidents:

        with st.expander(
            f"{incident['id']} | "
            f"{incident['issue']} | "
            f"{incident['severity']}"
        ):

            st.caption(
                f"Detected: {incident['detected_at']} "
                f"| Scenario: {incident['scenario']}"
            )

            st.markdown(
                f"**Severity:** {incident['severity']}"
            )

            st.markdown(
                f"**Possible Cause:** {incident['cause']}"
            )

            st.markdown(
                f"**Recommendation:** "
                f"{incident['recommendation']}"
            )

            st.markdown("**Telemetry Snapshot**")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Latency",
                incident["metrics"]["Latency"]
            )

            m2.metric(
                "Packet Loss",
                incident["metrics"]["Packet Loss"]
            )

            m3.metric(
                "Bandwidth",
                incident["metrics"]["Bandwidth"]
            )

            m4.metric(
                "CPU",
                incident["metrics"]["CPU"]
            )

            st.caption(
                f"AI anomalies in current sample: "
                f"{incident['anomalies']}"
            )

            st.markdown("**Update Incident Status**")

            status_options = [
                "Open",
                "Investigating",
                "Resolved",
            ]

            current_index = status_options.index(
                incident["status"]
            )

            updated_status = st.selectbox(
                "Incident Status",
                status_options,
                index=current_index,
                key=f"status_{incident['id']}",
            )

            incident["status"] = updated_status

st.caption(
    "Incident records are stored in Streamlit Session State. "
    "They are not permanent database records."
)



left, right = st.columns(2)
with left:
    st.subheader("Latency trend")
    latency_fig = px.line(
    df,
    x="Time",
    y="Latency (ms)",
    title="Network Latency",
    template="plotly_white",
)

    latency_fig.update_traces(
        line=dict(width=2),
    )

    latency_fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    detected = df[df["AI Anomaly"]]

    latency_fig.add_scatter(
        x=detected["Time"],
        y=detected["Latency (ms)"],
        mode="markers",
        name="AI Anomaly",
        marker=dict(
            color="red",
            size=11,
            symbol="x",
        ),
    )
    latency_fig.update_layout(height=300, margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(latency_fig, use_container_width=True)

with right:
    st.subheader("Packet loss trend")
    
    loss_fig = px.line(
        df,
        x="Time",
        y="Packet Loss (%)",
        title="Packet Loss",
        template="plotly_dark",
    )

    loss_fig.update_traces(
        line=dict(width=2),
    )

    loss_fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    loss_fig.update_layout(height=300, margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(loss_fig, use_container_width=True)

st.subheader("Illustrative campus network topology")
st.markdown(
    "`Internet`  →  `Router / Firewall`  →  `Core Switch`  →  "
    "`Access Switch`  →  `Wi-Fi Access Points`  →  `User Devices`"
)
st.caption(
    "Huawei product mapping for learning purposes: NetEngine (routing), "
    "HiSecEngine (security), CloudEngine (switching), AirEngine (wireless access). "
    "This is a conceptual mapping, not a tested Huawei configuration."
)


# ==========================================
# TELEMETRY DATA EXPORT
# ==========================================

st.subheader("Network Telemetry Data")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

csv_data = df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="⬇ Download Telemetry CSV",
    data=csv_data,
    file_name="network_telemetry.csv",
    mime="text/csv",
)


st.markdown("---")
st.caption("AI Network Observatory · Independent learning prototype · Synthetic data only")


# ==========================================
# NETWORK TOPOLOGY DISPLAY
# ==========================================

st.divider()
st.header("🖧 Network Topology")

st.caption(
    "Simulated enterprise campus network. "
    "Device health changes according to the selected scenario."
)

network_devices = get_network_devices(scenario)

topology_fig = create_network_topology(
    network_devices
)

st.plotly_chart(
    topology_fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "scrollZoom": True,
    },
)

# Device health summary
st.subheader("Device Health Status")

status_counts = {
    "Online": 0,
    "Warning": 0,
    "Offline": 0,
    "Disconnected": 0,
}

for device in network_devices:
    status_counts[device["status"]] += 1

health_cols = st.columns(4)

health_cols[0].metric(
    "Online",
    status_counts["Online"]
)

health_cols[1].metric(
    "Warning",
    status_counts["Warning"]
)

health_cols[2].metric(
    "Offline",
    status_counts["Offline"]
)

health_cols[3].metric(
    "Disconnected",
    status_counts["Disconnected"]
)

# Device details
with st.expander("View network device details"):

    device_df = pd.DataFrame(network_devices)

    st.dataframe(
        device_df[
            ["name", "product", "type", "status"]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Note: Huawei product names are used for conceptual mapping only. "
    "All topology devices and statuses are simulated."
)
