import math
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import FEATURES, train_models

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Parcel Tracker Estimator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------ Styling --------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root { color-scheme: light; }
.stApp {
    background: #f7f5f2 !important;
    color: #29233f !important;
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1120px;
    padding: 2rem 2rem 3rem;
}
header[data-testid="stHeader"] { background: transparent !important; }
h1, h2, h3, h4 { color: #29233f !important; }

.brand {
    background: #29233f;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 28px;
}
.brand-title {
    color: #ffffff !important;
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    line-height: 1.15;
}
.brand-subtitle {
    color: #e2d5b5 !important;
    font-size: 12px;
    margin-top: 7px;
    letter-spacing: .25px;
}
.section-title {
    color: #29233f !important;
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    margin: 24px 0 4px;
}
.helper {
    color: #716d7e !important;
    font-size: 13px;
    margin-bottom: 14px;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #39344e !important;
    font-weight: 600 !important;
}

/* Streamlit/BaseWeb select controls: force light surface and dark text. */
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #dcd5ca !important;
    border-radius: 9px !important;
    box-shadow: none !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #29233f !important;
    -webkit-text-fill-color: #29233f !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #29233f !important;
    fill: #29233f !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: #b49a62 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color: #b49a62 !important;
    box-shadow: 0 0 0 1px #b49a62 !important;
}

/* Open dropdown list. */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"] {
    background: #ffffff !important;
}
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="menu"] [role="option"] {
    background: #ffffff !important;
    color: #29233f !important;
    -webkit-text-fill-color: #29233f !important;
}
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="menu"] [role="option"]:hover,
div[data-baseweb="popover"] [aria-selected="true"] {
    background: #f0ecf2 !important;
    color: #29233f !important;
}

/* Numeric input. */
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #29233f !important;
    -webkit-text-fill-color: #29233f !important;
    border-color: #dcd5ca !important;
    border-radius: 9px !important;
}
div[data-testid="stNumberInput"] button {
    background: #29233f !important;
    color: #ffffff !important;
    border-color: #29233f !important;
}
div[data-testid="stNumberInput"] button svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* Main action. */
div[data-testid="stButton"] > button {
    width: 100%;
    min-height: 46px;
    background: #29233f !important;
    color: #ffffff !important;
    border: 1px solid #29233f !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button:hover {
    background: #40375f !important;
    border-color: #40375f !important;
}

/* Prediction cards and model score cards. */
.result-card, .score-card {
    background: #ffffff;
    border: 1px solid #e6e0d6;
    border-radius: 11px;
    padding: 18px 19px;
    height: 100%;
}
.result-card {
    border-left: 4px solid #b49a62;
}
.card-label {
    color: #777386 !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .8px;
    text-transform: uppercase;
}
.card-value {
    color: #29233f !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 29px;
    font-weight: 700;
    margin-top: 7px;
}
.card-note {
    color: #716d7e !important;
    font-size: 12px;
    margin-top: 5px;
}
.analysis-heading {
    color: #39344e !important;
    font-size: 13px;
    font-weight: 700;
    margin: 18px 0 8px;
}
.score-name {
    color: #777386 !important;
    font-size: 11px;
}
.score-value {
    color: #29233f !important;
    font-size: 22px;
    font-weight: 700;
    margin-top: 5px;
}
.score-note {
    color: #716d7e !important;
    font-size: 11px;
    margin-top: 4px;
}
div[data-testid="stCaptionContainer"] p {
    color: #777386 !important;
    opacity: 1 !important;
}
div[data-testid="stAlert"] {
    border-radius: 9px;
}
hr { border-color: #e8e2d9; }
footer { visibility: hidden; }

@media (max-width: 700px) {
    .block-container { padding: 1rem 1rem 2rem; }
    .brand-title { font-size: 27px; }
    .card-value { font-size: 24px; }
}
</style>
""", unsafe_allow_html=True)


# ------------------------------ Data & models --------------------------------
@st.cache_data
def load_locations():
    frame = pd.read_csv(DATA_DIR / "pincode_locations.csv", dtype={"pincode": str})
    required = {"pincode", "city", "state", "latitude", "longitude"}
    if not required.issubset(frame.columns):
        raise ValueError("PIN-code file is missing required columns.")
    frame["pincode"] = frame["pincode"].astype(str).str.zfill(6)
    frame = frame.drop_duplicates("pincode").reset_index(drop=True)
    return frame


@st.cache_resource
def load_models():
    # Train from the included synthetic dataset in this runtime.
    # This avoids incompatibility with model files made by another Python version.
    return train_models()


locations = load_locations()
with st.spinner("Training and validating the regression models..."):
    linear_model, poisson_model, metrics, _ = load_models()

location_map = locations.set_index("pincode").to_dict("index")
pin_codes = locations["pincode"].tolist()


def pin_label(pin):
    place = location_map[pin]
    return f"{place['city']}, {place['state']} — {pin}"


def estimate_route_km(origin_pin, destination_pin):
    """Approximate road distance from city-centre coordinates."""
    origin = location_map[origin_pin]
    destination = location_map[destination_pin]
    lat1, lat2 = math.radians(float(origin["latitude"])), math.radians(float(destination["latitude"]))
    lon1, lon2 = math.radians(float(origin["longitude"])), math.radians(float(destination["longitude"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    haversine = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    straight_km = 6371.0 * 2 * math.asin(math.sqrt(max(0.0, min(1.0, haversine))))
    # Training data uses a route factor between 1.12 and 1.32; use its midpoint.
    return max(5.0, straight_km) * 1.22


# -------------------------------- Header --------------------------------------
st.markdown("""
<div class="brand">
    <div class="brand-title">Parcel Tracker Estimator</div>
    <div class="brand-subtitle">E-commerce delivery time prediction</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Delivery details</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="helper">Enter shipment details to estimate delivery time.</div>',
    unsafe_allow_html=True,
)

# -------------------------------- Form ----------------------------------------
col1, col2 = st.columns(2)
with col1:
    default_origin = "301001" if "301001" in pin_codes else pin_codes[0]
    origin_pin = st.selectbox(
        "Pickup PIN code",
        pin_codes,
        index=pin_codes.index(default_origin),
        format_func=pin_label,
        key="origin_pin",
    )
with col2:
    destination_options = [pin for pin in pin_codes if pin != origin_pin]
    default_destination = "305001" if "305001" in destination_options else destination_options[0]
    destination_pin = st.selectbox(
        "Delivery PIN code",
        destination_options,
        index=destination_options.index(default_destination),
        format_func=pin_label,
        key="destination_pin",
    )

col3, col4 = st.columns(2)
with col3:
    weight = st.number_input(
        "Package weight (kg)",
        min_value=0.1,
        max_value=25.0,
        value=1.5,
        step=0.1,
        format="%.2f",
        key="package_weight",
    )
with col4:
    courier = st.selectbox(
        "Courier type",
        ["Standard", "Express", "Same-Day"],
        index=0,
        key="courier_type",
    )

distance_km = estimate_route_km(origin_pin, destination_pin)
origin_city = location_map[origin_pin]["city"]
destination_city = location_map[destination_pin]["city"]
st.caption(
    f"{origin_city} → {destination_city}  ·  "
    f"Approx. route distance: {distance_km:,.1f} km"
)

predict_clicked = st.button("Predict delivery time", use_container_width=True)

if predict_clicked:
    row = pd.DataFrame([{
        "distance_km": distance_km,
        "package_weight_kg": float(weight),
        "courier_type": courier,
    }], columns=FEATURES)

    linear_days = float(linear_model.predict(row)[0])
    poisson_expected_days = float(poisson_model.predict(row)[0])

    if not math.isfinite(linear_days) or not math.isfinite(poisson_expected_days):
        st.error("The models returned an invalid result. Please check the inputs and try again.")
        st.stop()

    # Keep continuous predictions decimal; round expected Poisson count to nearest day.
    linear_days = max(0.1, linear_days)
    poisson_expected_days = max(0.1, poisson_expected_days)
    poisson_whole_days = max(1, int(math.floor(poisson_expected_days + 0.5)))

    st.session_state["last_prediction"] = {
        "linear_days": linear_days,
        "poisson_expected_days": poisson_expected_days,
        "poisson_whole_days": poisson_whole_days,
        "origin_city": origin_city,
        "destination_city": destination_city,
        "distance_km": distance_km,
        "weight": float(weight),
        "courier": courier,
    }

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="card-label">Multiple Linear Regression</div>'
            f'<div class="card-value">{result["linear_days"]:.1f} days</div>'
            f'<div class="card-note">Continuous delivery-time estimate</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="card-label">Poisson Regression</div>'
            f'<div class="card-value">{result["poisson_whole_days"]} days</div>'
            f'<div class="card-note">Expected count: {result["poisson_expected_days"]:.2f} days</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f'{result["origin_city"]} → {result["destination_city"]}  ·  '
        f'{result["distance_km"]:.1f} km  ·  {result["weight"]:.1f} kg  ·  {result["courier"]}'
    )

# ----------------------------- Model analysis --------------------------------
st.markdown('<div class="section-title">Regression model analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="helper">Evaluation on the same unseen 20% test set. '
    'Lower MAE and RMSE are better; higher R² is better.</div>',
    unsafe_allow_html=True,
)

metric_specs = [
    ("Mean Absolute Error (MAE)", "MAE", "days", "lower"),
    ("Root Mean Squared Error (RMSE)", "RMSE", "days", "lower"),
    ("R² Score", "R2", "", "higher"),
]

for title, key, unit, direction in metric_specs:
    st.markdown(f'<div class="analysis-heading">{title}</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    for column, model_key, model_title in [
        (left, "linear", "Multiple Linear Regression"),
        (right, "poisson", "Poisson Regression"),
    ]:
        value = metrics[model_key][key]
        suffix = f" {unit}" if unit else ""
        with column:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-name">{model_title}</div>'
                f'<div class="score-value">{value:.3f}{suffix}</div>'
                f'<div class="score-note">{"Lower is better" if direction == "lower" else "Higher is better"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.caption(
    f"Model evaluation: {metrics['train_rows']:,} training records · "
    f"{metrics['test_rows']:,} unseen test records."
)
st.caption(
    f"Mean-only baseline MAE: Linear {metrics['linear_baseline_mae']:.3f} days · "
    f"Poisson {metrics['poisson_baseline_mae']:.3f} days. "
    "The count model has greater natural variation, so interpret its R² alongside MAE and RMSE."
)
st.markdown(
    '<div class="helper">This educational project uses synthetic shipment data and approximate route distances. '
    'Results demonstrate regression techniques and are not real courier delivery commitments.</div>',
    unsafe_allow_html=True,
)
