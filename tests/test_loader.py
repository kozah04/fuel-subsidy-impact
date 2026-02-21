"""
test_loader.py
--------------
Unit tests for src/loader.py

Run with: pytest tests/test_loader.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loader import (
    load_inflation,
    load_exchange_rate_monthly,
    load_exchange_rate_nfem,
    load_exchange_rate_worldbank,
    load_exchange_rate,
    load_fuel_prices,
)

# ---------------------------------------------------------------------------
# Paths — adjust if your data/raw folder is in a different location
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

INFLATION_FILE = os.path.join(DATA_DIR, "cbn_inflation.xlsx")
MONTHLY_FX_FILE = os.path.join(DATA_DIR, "Monthly_Average_Exchange_Rates_Data_in_Excel.xlsx")
NFEM_FILE = os.path.join(DATA_DIR, "NFEM_Rates_Data_in_Excel.xlsx")
WORLDBANK_FX_FILE = os.path.join(DATA_DIR, "API_PA_NUS_FCRF_DS2_en_csv_v2_108.csv")
PMS_2022_FILE = os.path.join(DATA_DIR, "PMS_Fuel_JUNE_2022.xlsx")
PMS_2024_FILE = os.path.join(DATA_DIR, "PMS_NOV_2024_REPORT.xlsx")
PMS_2025_FILE = os.path.join(DATA_DIR, "PMS_Report_June_2025.xlsx")


# ---------------------------------------------------------------------------
# Inflation tests
# ---------------------------------------------------------------------------

class TestLoadInflation:

    def setup_method(self):
        self.df = load_inflation(INFLATION_FILE)

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        required = {"date", "inflation_all_items", "inflation_food", "inflation_core"}
        assert required.issubset(set(self.df.columns))

    def test_date_column_is_datetime(self):
        assert pd.api.types.is_datetime64_any_dtype(self.df["date"])

    def test_dates_are_first_of_month(self):
        assert (self.df["date"].dt.day == 1).all()

    def test_covers_2020_to_2023(self):
        dates = self.df["date"]
        assert dates.min() <= pd.Timestamp("2020-01-01")
        assert dates.max() >= pd.Timestamp("2023-12-01")

    def test_inflation_values_are_numeric(self):
        for col in ["inflation_all_items", "inflation_food", "inflation_core"]:
            assert pd.api.types.is_float_dtype(self.df[col]) or pd.api.types.is_integer_dtype(self.df[col])

    def test_inflation_values_in_plausible_range(self):
        # Nigerian inflation has been between 5% and 100% in recent years
        valid = self.df["inflation_all_items"].dropna()
        assert (valid > 0).all()
        assert (valid < 150).all()

    def test_sorted_by_date(self):
        assert self.df["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Exchange rate tests
# ---------------------------------------------------------------------------

class TestLoadExchangeRateMonthly:

    def setup_method(self):
        self.df = load_exchange_rate_monthly(MONTHLY_FX_FILE)

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        assert {"date", "usd_ngn"}.issubset(set(self.df.columns))

    def test_date_is_datetime(self):
        assert pd.api.types.is_datetime64_any_dtype(self.df["date"])

    def test_no_null_values(self):
        assert self.df[["date", "usd_ngn"]].isnull().sum().sum() == 0

    def test_rates_are_positive(self):
        assert (self.df["usd_ngn"] > 0).all()

    def test_sorted_by_date(self):
        assert self.df["date"].is_monotonic_increasing


class TestLoadExchangeRateNFEM:

    def setup_method(self):
        self.df = load_exchange_rate_nfem(NFEM_FILE)

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        assert {"date", "usd_ngn"}.issubset(set(self.df.columns))

    def test_is_monthly(self):
        # All dates should be first of month after resampling
        assert (self.df["date"].dt.day == 1).all()

    def test_rates_above_1000(self):
        # NFEM covers late 2024 when naira was well above 1000/dollar
        assert (self.df["usd_ngn"] > 1000).all()


class TestLoadExchangeRateStitched:

    def setup_method(self):
        self.df = load_exchange_rate(MONTHLY_FX_FILE, NFEM_FILE, WORLDBANK_FX_FILE)

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_starts_from_2020(self):
        assert self.df["date"].min() <= pd.Timestamp("2020-01-01")

    def test_no_null_rates(self):
        assert self.df["usd_ngn"].isnull().sum() == 0

    def test_rates_are_positive(self):
        assert (self.df["usd_ngn"] > 0).all()

    def test_sorted_by_date(self):
        assert self.df["date"].is_monotonic_increasing

    def test_2020_rate_plausible(self):
        # USD/NGN was around 300-380 in 2020
        rate_2020 = self.df[self.df["date"].dt.year == 2020]["usd_ngn"].mean()
        assert 250 < rate_2020 < 500

    def test_2024_rate_plausible(self):
        # USD/NGN was above 1000 in 2024
        rate_2024 = self.df[self.df["date"].dt.year == 2024]["usd_ngn"].mean()
        assert rate_2024 > 1000


# ---------------------------------------------------------------------------
# Fuel price tests
# ---------------------------------------------------------------------------

class TestLoadFuelPrices:

    def setup_method(self):
        self.df = load_fuel_prices(PMS_2022_FILE, PMS_2024_FILE, PMS_2025_FILE)

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        assert {"date", "fuel_price_ngn", "price_source"}.issubset(set(self.df.columns))

    def test_starts_from_2020(self):
        assert self.df["date"].min() <= pd.Timestamp("2020-01-01")

    def test_sorted_by_date(self):
        assert self.df["date"].is_monotonic_increasing

    def test_pre_subsidy_price_around_145_to_179(self):
        pre = self.df[self.df["date"] < "2023-05-01"]["fuel_price_ngn"]
        assert (pre >= 140).all()
        assert (pre <= 185).all()

    def test_post_subsidy_price_above_200(self):
        post = self.df[self.df["date"] >= "2023-06-01"]["fuel_price_ngn"].dropna()
        assert (post > 200).all()

    def test_price_jumps_at_subsidy_removal(self):
        may_2023 = self.df[self.df["date"] == "2023-05-01"]["fuel_price_ngn"].values[0]
        nov_2023 = self.df[self.df["date"] == "2023-11-01"]["fuel_price_ngn"].values[0]
        assert nov_2023 > may_2023 * 2  # price more than doubled

    def test_price_source_column_has_valid_values(self):
        valid_sources = {"fixed_govt", "nbs_pms_report", "interpolated"}
        actual = set(self.df["price_source"].dropna().unique())
        assert actual.issubset(valid_sources)