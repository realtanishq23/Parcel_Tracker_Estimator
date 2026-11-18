# Parcel Tracker Estimator

A single-page Streamlit project for an academic regression-analysis lab.

## Models
- **Multiple Linear Regression** predicts continuous delivery duration in days.
- **Poisson Regression** predicts the expected count of delivery days. The displayed whole-day estimate uses nearest-integer rounding.

Both models use the same three inputs: route distance, package weight, and courier type. Both are evaluated on the same fixed 80/20 train/test split.

## Run locally (Windows / VS Code)

1. Extract the project ZIP.
2. Open the `Parcel_Tracker_Estimator` folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application trains both models from the included CSV dataset when it starts. To regenerate the same synthetic dataset, run `python -m src.generate_data`. It does not load pre-trained model files, so it avoids old Python/scikit-learn pickle compatibility issues.

## Run model tests

```powershell
python -m unittest discover -s tests -v
```

## Project structure

```text
Parcel_Tracker_Estimator/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── delivery_data.csv
│   └── pincode_locations.csv
├── src/
│   ├── __init__.py
│   └── model.py
└── tests/
    └── test_models.py
```

## Important data and interpretation notes

- All shipment training records are **synthetic**. They are generated for this educational demonstration and are not historical courier-company data.
- The PIN directory contains 158 curated sample locations, not every Indian PIN code.
- Coordinates are approximate city-centre coordinates. The app estimates route distance from straight-line distance using a fixed road-distance factor.
- Continuous and discrete targets are generated separately from documented synthetic relationships. Poisson Regression is trained on a positive integer count target; it is not trained by simply rounding the linear model's predictions.
- Test metrics measure performance on a held-out portion of this synthetic dataset. They do not establish real-world courier accuracy.


## Model evaluation note

The current fixed-seed test run produced:
- Multiple Linear Regression: MAE 0.157 days, RMSE 0.199 days, R² 0.962.
- Poisson Regression: MAE 1.240 days, RMSE 1.565 days, R² 0.234.

The Poisson target is a random count outcome, so its R² is lower because count outcomes vary naturally even for similar inputs. It improves on the mean-only baseline MAE (1.423 days) in this test. Do not describe these synthetic-data scores as real-world courier accuracy.
