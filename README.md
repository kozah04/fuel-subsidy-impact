# Nigerian Fuel Subsidy Removal — Economic Impact Analysis

A data science project formally analysing the macroeconomic impact of Nigeria's May 2023 fuel subsidy removal on inflation, fuel pump prices, and the USD/NGN exchange rate.

## Key Finding

The Chow test (F=44.46, p≈0) confirms a statistically significant structural break at May 2023 — the subsidy removal didn't just change price levels, it changed the fundamental relationship between fuel prices, exchange rates, and inflation. Counterfactual modelling suggests inflation would have remained around 16–17% without the fuel shock, compared to an actual peak of 35% in late 2024.

## Project Structure

```
fuel-subsidy-impact/
├── data/
│   ├── raw/              # Raw datasets from CBN, NBS, World Bank (gitignored)
│   └── processed/        # Cleaned/merged data (gitignored)
├── notebooks/
│   ├── exploration.ipynb # EDA — inflation, fuel price, exchange rate trends
│   └── analysis.ipynb    # Modelling — Chow test, counterfactual, feature importance
├── src/
│   ├── loader.py         # Data loading and merging functions
│   └── analysis.py       # Feature engineering, modelling, and Chow test
├── tests/
│   └── test_loader.py    # Unit tests for loader functions (33/33 passing)
├── outputs/              # Generated charts (gitignored)
└── report.md             # Full written report with findings and limitations
```

## Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| CBN | Monthly inflation (all items, food, core) | 2003–2026 |
| CBN | Monthly USD/NGN exchange rates | 2004–2021 |
| NBS | PMS petrol pump price reports | 2021–2025 |
| World Bank | Annual USD/NGN exchange rates | 2004–2024 |

## Setup

```bash
# Clone the repo
git clone https://github.com/kozah04/fuel-subsidy-impact.git
cd fuel-subsidy-impact

# Create and activate conda environment
conda create -n fuel-subsidy python=3.11
conda activate fuel-subsidy

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy jupyter ipykernel python-dotenv requests pytest openpyxl

# Register kernel
python -m ipykernel install --user --name fuel-subsidy --display-name "fuel-subsidy"
```

## Running the Project

```bash
# Run unit tests
pytest tests/test_loader.py -v

# Launch notebooks
jupyter notebook
```

Open `notebooks/exploration.ipynb` first for EDA, then `notebooks/analysis.ipynb` for the modelling.

## Methods

- **Structural break testing** — Chow test to formally confirm the May 2023 break point
- **Counterfactual modelling** — Pre-removal regression used to simulate what inflation would have looked like without the fuel shock
- **Feature importance** — Linear Regression, Ridge, Lasso, and Random Forest trained to identify which variables consistently drive inflation
- **Exploratory analysis** — Before/after comparisons, correlation heatmaps, and time series visualisations

## Results Summary

| Indicator | Pre-Removal avg | Post-Removal avg | Change |
|-----------|----------------|-----------------|--------|
| Inflation — All Items (%) | 16.89 | 27.28 | +61.5% |
| Inflation — Food (%) | 19.68 | 29.76 | +51.2% |
| Fuel Price (NGN/litre) | 160.72 | 800.43 | +398% |
| Exchange Rate (NGN/USD) | 418.29 | 1,293.39 | +209% |

## Author

**Gwachat Kozah** — [github.com/kozah04](https://github.com/kozah04)