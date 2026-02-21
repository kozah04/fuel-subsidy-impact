"""
analysis.py
-----------
Feature engineering and modelling functions for the Nigerian fuel subsidy
economic impact analysis.

Functions are grouped into:
    1. Feature engineering — lagged variables, rolling averages, pct changes
    2. Model training — linear regression, Ridge, Lasso, Random Forest
    3. Model evaluation — metrics, cross-validation scores
    4. Structural break — Chow test utility
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged variables, rolling averages, and percentage changes to the
    master DataFrame.

    New columns added:
        - fuel_pct_change        : month-on-month % change in fuel price
        - usd_ngn_pct_change     : month-on-month % change in exchange rate
        - inflation_lag1         : inflation_all_items lagged 1 month
        - inflation_lag2         : inflation_all_items lagged 2 months
        - inflation_lag3         : inflation_all_items lagged 3 months
        - fuel_lag1              : fuel_price_ngn lagged 1 month
        - fuel_lag2              : fuel_price_ngn lagged 2 months
        - usd_ngn_lag1           : usd_ngn lagged 1 month
        - fuel_roll3             : 3-month rolling average of fuel price
        - usd_ngn_roll3          : 3-month rolling average of exchange rate
        - post_subsidy_removal   : structural break dummy (already in df)

    Args:
        df: master DataFrame from load_all()

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Percentage changes
    df["fuel_pct_change"] = df["fuel_price_ngn"].pct_change() * 100
    df["usd_ngn_pct_change"] = df["usd_ngn"].pct_change() * 100

    # Lagged variables
    df["inflation_lag1"] = df["inflation_all_items"].shift(1)
    df["inflation_lag2"] = df["inflation_all_items"].shift(2)
    df["inflation_lag3"] = df["inflation_all_items"].shift(3)
    df["fuel_lag1"] = df["fuel_price_ngn"].shift(1)
    df["fuel_lag2"] = df["fuel_price_ngn"].shift(2)
    df["usd_ngn_lag1"] = df["usd_ngn"].shift(1)

    # Rolling averages
    df["fuel_roll3"] = df["fuel_price_ngn"].rolling(window=3).mean()
    df["usd_ngn_roll3"] = df["usd_ngn"].rolling(window=3).mean()

    # Month-on-month change in inflation (more stationary target)
    df["inflation_change"] = df["inflation_all_items"].diff()

    return df


def get_model_data(
    df: pd.DataFrame,
    target: str = "inflation_all_items",
    features: list = None,
    dropna: bool = True,
) -> tuple:
    """
    Prepare feature matrix X and target vector y for modelling.

    Args:
        df: DataFrame with engineered features
        target: column to predict (default: inflation_all_items)
        features: list of feature column names. If None, uses default set.
        dropna: whether to drop rows with NaN values

    Returns:
        X (DataFrame), y (Series), feature_names (list)
    """
    if features is None:
        features = [
            "fuel_price_ngn",
            "usd_ngn",
            "fuel_lag1",
            "fuel_lag2",
            "usd_ngn_lag1",
            "fuel_pct_change",
            "usd_ngn_pct_change",
            "fuel_roll3",
            "usd_ngn_roll3",
            "post_subsidy_removal",
        ]

    # Only use features that exist in the dataframe
    features = [f for f in features if f in df.columns]

    data = df[features + [target]].copy()

    if dropna:
        data = data.dropna()

    X = data[features]
    y = data[target]

    return X, y, features


# ---------------------------------------------------------------------------
# 2. MODEL TRAINING
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Train a simple OLS linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train, alpha: float = 1.0) -> Ridge:
    """Train a Ridge regression model."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_lasso(X_train, y_train, alpha: float = 0.1) -> Lasso:
    """Train a Lasso regression model."""
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train, y_train, n_estimators: int = 100, random_state: int = 42
) -> RandomForestRegressor:
    """Train a Random Forest regression model."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        max_features="sqrt",
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train) -> dict:
    """
    Train all four models and return them in a dictionary.

    Returns:
        dict with keys: 'linear', 'ridge', 'lasso', 'random_forest'
    """
    return {
        "linear": train_linear_regression(X_train, y_train),
        "ridge": train_ridge(X_train, y_train),
        "lasso": train_lasso(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
    }


# ---------------------------------------------------------------------------
# 3. MODEL EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model on test data.

    Returns:
        dict with R², RMSE, MAE
    """
    y_pred = model.predict(X_test)
    return {
        "R2": round(r2_score(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
    }


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a comparison DataFrame.

    Args:
        models: dict of trained models from train_all_models()
        X_test: test feature matrix
        y_test: test target vector

    Returns:
        DataFrame with R², RMSE, MAE for each model
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)

    return pd.DataFrame(results).T.sort_values("R2", ascending=False)


def cross_validate_models(models: dict, X, y, n_splits: int = 5) -> pd.DataFrame:
    """
    Time-series cross-validation for all models.
    Uses TimeSeriesSplit to respect temporal ordering.

    Args:
        models: dict of trained models
        X: full feature matrix
        y: full target vector
        n_splits: number of CV folds

    Returns:
        DataFrame with mean and std of R² across folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
        results[name] = {
            "CV R² (mean)": round(scores.mean(), 4),
            "CV R² (std)": round(scores.std(), 4),
        }

    return pd.DataFrame(results).T.sort_values("CV R² (mean)", ascending=False)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from a trained model.
    Works for LinearRegression, Ridge, Lasso (coefficients) and
    RandomForest (feature_importances_).

    Returns:
        DataFrame sorted by absolute importance descending
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        col = "Importance"
    elif hasattr(model, "coef_"):
        importances = model.coef_
        col = "Coefficient"
    else:
        raise ValueError("Model has neither feature_importances_ nor coef_")

    df = pd.DataFrame({
        "Feature": feature_names,
        col: importances,
    })
    df["Abs"] = df[col].abs()
    df = df.sort_values("Abs", ascending=False).drop(columns="Abs")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. STRUCTURAL BREAK — CHOW TEST
# ---------------------------------------------------------------------------

def chow_test(df: pd.DataFrame, break_date: str, features: list, target: str) -> dict:
    """
    Perform a Chow test to determine whether a structural break exists
    at the given date.

    The Chow test compares:
        - RSS from a single model on the full dataset
        - RSS from separate models on pre and post break subsets

    A significant F-statistic indicates a structural break.

    Args:
        df: DataFrame with features and target
        break_date: date string of the break point (e.g. '2023-05-01')
        features: list of feature column names
        target: target column name

    Returns:
        dict with F-statistic, p-value, and interpretation
    """
    from scipy import stats

    data = df[features + [target]].dropna()
    pre = data[df.loc[data.index, "date"] < break_date]
    post = data[df.loc[data.index, "date"] >= break_date]

    def get_rss(subset):
        X = subset[features].values
        y = subset[target].values
        model = LinearRegression().fit(X, y)
        return np.sum((y - model.predict(X)) ** 2), len(y)

    rss_full, n = get_rss(data)
    rss_pre, n1 = get_rss(pre)
    rss_post, n2 = get_rss(post)

    k = len(features)
    f_stat = ((rss_full - (rss_pre + rss_post)) / k) / ((rss_pre + rss_post) / (n - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)

    return {
        "F-statistic": round(f_stat, 4),
        "p-value": round(p_value, 6),
        "Structural break detected": p_value < 0.05,
        "Interpretation": (
            "Strong evidence of a structural break at the subsidy removal date."
            if p_value < 0.05
            else "No statistically significant structural break detected."
        ),
    }