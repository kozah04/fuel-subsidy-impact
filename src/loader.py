"""
loader.py
---------
Functions to load and clean each raw dataset into a consistent monthly
time series (DataFrame with a 'date' column as the first column).

All functions return a DataFrame with:
  - date: dtype datetime64, first of each month (e.g. 2023-05-01)
  - one or more indicator columns (floats)

No merging happens here — that is handled in analysis.py.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. INFLATION — CBN monthly inflation data
# ---------------------------------------------------------------------------

def load_inflation(filepath: str) -> pd.DataFrame:
    """
    Load CBN monthly inflation data.

    Returns columns:
        date, inflation_all_items, inflation_food, inflation_core
    """
    df = pd.read_excel(filepath, header=0)

    # Rename columns for clarity
    df = df.rename(columns={
        "tyear": "year",
        "tmonth": "month",
        "allItemsYearOn": "inflation_all_items",
        "foodYearOn": "inflation_food",
        "allItemsLessFrmProdAndEnergyYearOn": "inflation_core",
    })

    # Build date column
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
    )

    df = df[["date", "inflation_all_items", "inflation_food", "inflation_core"]].copy()

    # Convert to numeric, coerce errors to NaN
    for col in ["inflation_all_items", "inflation_food", "inflation_core"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 2. EXCHANGE RATE — stitch together three sources into one monthly series
# ---------------------------------------------------------------------------

def load_exchange_rate_monthly(filepath: str) -> pd.DataFrame:
    """
    Load CBN monthly average exchange rates (2004–April 2021).

    Returns columns:
        date, usd_ngn
    """
    df = pd.read_excel(filepath, header=0)

    df = df.rename(columns={"tyear": "year", "tmonth": "month", "ifemDollar": "usd_ngn"})

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
    )

    df["usd_ngn"] = pd.to_numeric(df["usd_ngn"], errors="coerce")

    df = df[["date", "usd_ngn"]].dropna().sort_values("date").reset_index(drop=True)

    return df


def load_exchange_rate_nfem(filepath: str) -> pd.DataFrame:
    """
    Load CBN NFEM daily exchange rates and resample to monthly averages.

    Returns columns:
        date, usd_ngn
    """
    df = pd.read_excel(filepath, header=0)

    df = df.rename(columns={"ratedate": "date", "weightedAvgRate": "usd_ngn"})

    df["date"] = pd.to_datetime(df["date"], format="%B-%d-%Y", errors="coerce")
    df["usd_ngn"] = pd.to_numeric(df["usd_ngn"], errors="coerce")

    df = df[["date", "usd_ngn"]].dropna()

    # Resample to monthly average, using first of month as label
    df = df.set_index("date").resample("MS").mean().reset_index()

    return df


def load_exchange_rate_worldbank(filepath: str) -> pd.DataFrame:
    """
    Load World Bank annual average USD/NGN exchange rates.
    Expands annual values to monthly (same value for each month in year).

    Returns columns:
        date, usd_ngn
    """
    df = pd.read_csv(filepath, skiprows=4)
    ng = df[df["Country Code"] == "NGA"].copy()

    # Melt year columns into rows
    year_cols = [c for c in df.columns if c.isdigit()]
    ng = ng[year_cols].T.reset_index()
    ng.columns = ["year", "usd_ngn"]
    ng["year"] = ng["year"].astype(int)
    ng["usd_ngn"] = pd.to_numeric(ng["usd_ngn"], errors="coerce")
    ng = ng.dropna()

    # Expand each year into 12 monthly rows
    rows = []
    for _, row in ng.iterrows():
        for month in range(1, 13):
            rows.append({
                "date": pd.Timestamp(year=int(row["year"]), month=month, day=1),
                "usd_ngn": row["usd_ngn"],
            })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_exchange_rate(
    monthly_filepath: str,
    nfem_filepath: str,
    worldbank_filepath: str,
) -> pd.DataFrame:
    """
    Stitch together all three exchange rate sources into one clean monthly
    series from January 2020 to the latest available date.

    Priority order:
        1. CBN Monthly Average (most accurate for 2020–Apr 2021)
        2. World Bank annual (fills 2021–2023 gap, expanded to monthly)
        3. NFEM daily resampled (most accurate for late 2024 onwards)

    Returns columns:
        date, usd_ngn
    """
    cbm = load_exchange_rate_monthly(monthly_filepath)
    wb = load_exchange_rate_worldbank(worldbank_filepath)
    nfem = load_exchange_rate_nfem(nfem_filepath)

    # Start with World Bank as the base (covers everything annually)
    base = wb.copy()

    # Overwrite with CBN monthly where available (more accurate)
    for _, row in cbm.iterrows():
        mask = base["date"] == row["date"]
        if mask.any():
            base.loc[mask, "usd_ngn"] = row["usd_ngn"]
        else:
            base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)

    # Overwrite with NFEM monthly where available (most recent, most accurate)
    for _, row in nfem.iterrows():
        mask = base["date"] == row["date"]
        if mask.any():
            base.loc[mask, "usd_ngn"] = row["usd_ngn"]
        else:
            base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)

    base = base.sort_values("date").reset_index(drop=True)

    # Filter to 2020 onwards
    base = base[base["date"] >= "2020-01-01"].reset_index(drop=True)

    return base


# ---------------------------------------------------------------------------
# 3. FUEL PRICES — construct monthly series from PMS reports + known prices
# ---------------------------------------------------------------------------

def load_fuel_prices(
    pms_2022_filepath: str,
    pms_2024_filepath: str,
    pms_2025_filepath: str,
) -> pd.DataFrame:
    """
    Construct a national monthly average fuel price (NGN/litre) series.

    Pre-subsidy removal prices are well-known fixed government prices.
    Post-removal prices are extracted from the NBS PMS reports.

    Returns columns:
        date, fuel_price_ngn, price_source
    """

    # --- Known fixed government pump prices (pre-subsidy removal) ---
    # Source: NNPC/PPPRA official price history
    fixed_prices = [
        ("2020-01-01", "2021-02-28", 145.0),   # ₦145 fixed price
        ("2021-03-01", "2022-05-31", 162.0),   # slight increase
        ("2022-06-01", "2023-05-31", 179.0),   # price adjustment
    ]

    rows = []
    for start, end, price in fixed_prices:
        dates = pd.date_range(start=start, end=end, freq="MS")
        for d in dates:
            rows.append({"date": d, "fuel_price_ngn": price, "price_source": "fixed_govt"})

    # --- Extract national averages from NBS PMS reports ---
    def extract_national_avg(filepath, sheet_index=0):
        """Extract the national average price from an NBS PMS report."""
        xl = pd.ExcelFile(filepath)
        df = xl.parse(xl.sheet_names[sheet_index], header=None)

        # The zone summary is on the right side — average the zone averages
        # Zone average is in the last column, rows 1 onwards until NaN
        # Column structure: State | m-2 | m-1 | m | ... | Zone | ZoneAvg
        last_col = df.iloc[1:, -1]
        prices = pd.to_numeric(last_col, errors="coerce").dropna()
        return round(prices.mean(), 2)

    # PMS 2022: Jun-21, May-22, Jun-22
    try:
        xl_2022 = pd.ExcelFile(pms_2022_filepath)
        df_2022 = xl_2022.parse(xl_2022.sheet_names[0], header=None)
        # Column indices: 1=Jun21, 2=May22, 3=Jun22
        for col_idx, date_str in [(1, "2021-06-01"), (2, "2022-05-01"), (3, "2022-06-01")]:
            prices = pd.to_numeric(df_2022.iloc[1:37, col_idx], errors="coerce").dropna()
            rows.append({
                "date": pd.Timestamp(date_str),
                "fuel_price_ngn": round(prices.mean(), 2),
                "price_source": "nbs_pms_report",
            })
    except Exception as e:
        print(f"Warning: could not parse 2022 PMS file: {e}")

    # PMS 2024: Nov-23, Oct-24, Nov-24
    try:
        xl_2024 = pd.ExcelFile(pms_2024_filepath)
        df_2024 = xl_2024.parse(xl_2024.sheet_names[0], header=None)
        for col_idx, date_str in [(1, "2023-11-01"), (2, "2024-10-01"), (3, "2024-11-01")]:
            prices = pd.to_numeric(df_2024.iloc[2:38, col_idx], errors="coerce").dropna()
            rows.append({
                "date": pd.Timestamp(date_str),
                "fuel_price_ngn": round(prices.mean(), 2),
                "price_source": "nbs_pms_report",
            })
    except Exception as e:
        print(f"Warning: could not parse 2024 PMS file: {e}")

    # PMS 2025: Jun-24, May-25, Jun-25
    try:
        xl_2025 = pd.ExcelFile(pms_2025_filepath)
        df_2025 = xl_2025.parse(xl_2025.sheet_names[0], header=None)
        for col_idx, date_str in [(1, "2024-06-01"), (2, "2025-05-01"), (3, "2025-06-01")]:
            prices = pd.to_numeric(df_2025.iloc[1:37, col_idx], errors="coerce").dropna()
            rows.append({
                "date": pd.Timestamp(date_str),
                "fuel_price_ngn": round(prices.mean(), 2),
                "price_source": "nbs_pms_report",
            })
    except Exception as e:
        print(f"Warning: could not parse 2025 PMS file: {e}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    # Interpolate missing months between known data points
    full_range = pd.DataFrame({"date": pd.date_range("2020-01-01", df["date"].max(), freq="MS")})
    df = full_range.merge(df, on="date", how="left")
    df["fuel_price_ngn"] = df["fuel_price_ngn"].interpolate(method="linear")
    df["price_source"] = df["price_source"].fillna("interpolated")

    return df


# ---------------------------------------------------------------------------
# 4. MASTER LOADER — load and merge all datasets
# ---------------------------------------------------------------------------

def load_all(data_dir: str) -> pd.DataFrame:
    """
    Load and merge all datasets into a single monthly master DataFrame.

    Args:
        data_dir: path to the data/raw/ folder

    Returns:
        Merged DataFrame indexed by date (2020-01 to latest available)
    """
    import os

    inflation = load_inflation(os.path.join(data_dir, "cbn_inflation.xlsx"))

    exchange = load_exchange_rate(
        monthly_filepath=os.path.join(data_dir, "Monthly_Average_Exchange_Rates_Data_in_Excel.xlsx"),
        nfem_filepath=os.path.join(data_dir, "NFEM_Rates_Data_in_Excel.xlsx"),
        worldbank_filepath=os.path.join(data_dir, "API_PA_NUS_FCRF_DS2_en_csv_v2_108.csv"),
    )

    fuel = load_fuel_prices(
        pms_2022_filepath=os.path.join(data_dir, "PMS_Fuel_JUNE_2022.xlsx"),
        pms_2024_filepath=os.path.join(data_dir, "PMS_NOV_2024_REPORT.xlsx"),
        pms_2025_filepath=os.path.join(data_dir, "PMS_Report_June_2025.xlsx"),
    )

    # Merge all on date
    df = inflation.merge(exchange, on="date", how="outer")
    df = df.merge(fuel[["date", "fuel_price_ngn", "price_source"]], on="date", how="outer")

    df = df.sort_values("date").reset_index(drop=True)

    # Filter to 2020 onwards
    df = df[df["date"] >= "2020-01-01"].reset_index(drop=True)

    # Add structural break indicator (1 = post subsidy removal, May 2023 onwards)
    df["post_subsidy_removal"] = (df["date"] >= "2023-05-01").astype(int)

    return df