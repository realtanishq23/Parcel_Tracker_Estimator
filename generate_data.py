"""Regenerate the educational synthetic shipment dataset (fixed seed)."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 20261118
RECORDS = 6000


def generate():
    locations = pd.read_csv(DATA_DIR / "pincode_locations.csv", dtype={"pincode": str})
    rng = np.random.default_rng(SEED)

    origin_index = rng.integers(0, len(locations), size=RECORDS)
    offset = rng.integers(1, len(locations), size=RECORDS)
    destination_index = (origin_index + offset) % len(locations)

    origin = locations.iloc[origin_index].reset_index(drop=True)
    destination = locations.iloc[destination_index].reset_index(drop=True)

    lat1 = np.radians(origin["latitude"].to_numpy(float))
    lat2 = np.radians(destination["latitude"].to_numpy(float))
    lon1 = np.radians(origin["longitude"].to_numpy(float))
    lon2 = np.radians(destination["longitude"].to_numpy(float))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    haversine = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    straight_distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    route_distance = np.maximum(5, straight_distance) * rng.uniform(1.12, 1.32, size=RECORDS)

    weight = np.round(rng.uniform(0.2, 25.0, size=RECORDS), 2)
    courier = rng.choice(
        ["Standard", "Express", "Same-Day"],
        size=RECORDS,
        p=[0.55, 0.32, 0.13],
    )
    courier_effect = np.select(
        [courier == "Standard", courier == "Express", courier == "Same-Day"],
        [0.42, 0.05, -0.28],
    )

    continuous_mean = 0.62 + route_distance / 690 + weight * 0.027 + courier_effect
    continuous_days = np.maximum(
        0.35, continuous_mean + rng.normal(0, 0.20, size=RECORDS)
    )

    # Separate count-generating process for Poisson Regression.
    poisson_rate = np.maximum(
        1.05, 0.68 + route_distance / 720 + weight * 0.025 + courier_effect
    )
    discrete_days = np.maximum(1, rng.poisson(poisson_rate)).astype(int)

    result = pd.DataFrame({
        "origin_pincode": origin["pincode"].astype(str).str.zfill(6),
        "origin_city": origin["city"],
        "origin_state": origin["state"],
        "destination_pincode": destination["pincode"].astype(str).str.zfill(6),
        "destination_city": destination["city"],
        "destination_state": destination["state"],
        "distance_km": np.round(route_distance, 2),
        "package_weight_kg": weight,
        "courier_type": courier,
        "continuous_delivery_days": np.round(continuous_days, 3),
        "discrete_delivery_days": discrete_days,
    })
    result.to_csv(DATA_DIR / "delivery_data.csv", index=False)
    print(f"Generated {len(result):,} synthetic shipment records.")


if __name__ == "__main__":
    generate()
