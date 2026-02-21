# The Economic Impact of Nigeria's 2023 Fuel Subsidy Removal
### A Data Science Analysis of Inflation, Exchange Rates, and Fuel Prices

**Author:** Gwachat Kozah  
**Date:** February 2026  
**Data Sources:** Central Bank of Nigeria (CBN), Nigerian Bureau of Statistics (NBS), World Bank  
**Repository:** [github.com/kozah04/fuel-subsidy-impact](https://github.com/kozah04/fuel-subsidy-impact)

---

## Executive Summary

On 29 May 2023, Nigeria's President Bola Tinubu announced the immediate removal of the country's long-standing fuel subsidy, ending decades of government-controlled petrol prices. The policy change was widely anticipated but the speed and scale of its impact was not. Within months, petrol prices had tripled, the naira had collapsed, and headline inflation had surged to levels not seen in a generation.

This report uses 74 months of macroeconomic data (January 2020 – February 2026) to formally analyse the impact of the subsidy removal on three key indicators: inflation, fuel pump prices, and the USD/NGN exchange rate. The analysis combines exploratory data analysis, structural break testing, counterfactual modelling, and machine learning feature importance to answer three research questions:

1. Did a statistically significant structural break occur at May 2023?
2. How much of the post-removal inflation surge can be attributed to fuel prices vs exchange rates?
3. Which factors consistently matter across different modelling approaches?

**Key findings:** A structural break is confirmed with very high statistical confidence (F=44.46, p≈0). Fuel prices were the dominant driver of the inflation surge, with the counterfactual analysis suggesting inflation would have remained around 16–17% without the fuel shock — compared to an actual peak of 35% in late 2024. Exchange rate depreciation amplified the shock but played a secondary role in month-to-month inflation dynamics.

---

## 1. Background

Nigeria's fuel subsidy was one of the most expensive government interventions in sub-Saharan Africa. For decades, the government sold petrol at fixed prices well below market rates, subsidising the difference from oil revenues. By 2022, subsidy costs had ballooned to over ₦4 trillion annually — consuming a significant share of government revenue and crowding out spending on infrastructure and social services.

The argument for removal was fiscal: Nigeria could not afford to keep subsidising fuel while running large budget deficits. The argument against was distributional: petrol prices in Nigeria affect almost everything — transport, food logistics, power generation — and removing the subsidy would disproportionately hurt low-income households who depend on cheap fuel and cheap food.

The removal happened with almost no transition period. On 29 May 2023, pump prices went from ₦179 per litre to over ₦500 overnight, eventually rising above ₦1,200 per litre by late 2024.

---

## 2. Data

Data was sourced from three institutions:

- **CBN (Central Bank of Nigeria):** Monthly inflation rates (all items, food, and core) from 2003–2026; monthly USD/NGN exchange rates from 2004–2021
- **NBS (Nigerian Bureau of Statistics):** PMS petrol pump price reports from 2021–2025
- **World Bank:** Annual USD/NGN exchange rates used to fill gaps in the 2021–2023 period

The final dataset covers 74 months from January 2020 to February 2026 with 8 core variables. Fuel prices for months between NBS reports were linearly interpolated. Exchange rates for 2021–2023 were derived from World Bank annual averages expanded to monthly, which understates intra-year volatility — a limitation noted where relevant.

---

## 3. Exploratory Analysis

### 3.1 Inflation Trends

![Inflation Trends](outputs/inflation_trends.png)

Headline inflation was already rising before the subsidy removal, driven by COVID-era supply disruptions and pre-existing naira pressure. However, the rate of increase accelerated sharply after May 2023. All-items inflation peaked at **34.8%** in late 2024, up from **22.4%** at the time of removal. Food inflation was the most severely affected, peaking above **40%**.

Notably, inflation has been declining since late 2024, reaching **15.1%** by February 2026 — though this remains significantly above pre-removal levels.

### 3.2 Fuel Pump Prices

![Fuel Prices](outputs/fuel_prices.png)

The fuel price chart shows one of the most dramatic policy-driven price shocks in recent Nigerian economic history. Prices were essentially flat at ₦145–179 per litre from 2020 to May 2023. After the removal they rose continuously, peaking at approximately **₦1,214 per litre** in late 2024 before declining to around ₦1,040 by mid-2025 — still **6x** the pre-removal price.

### 3.3 Exchange Rate

![Exchange Rate](outputs/exchange_rate.png)

The naira had already weakened significantly in late 2022 when the CBN began unifying its multiple exchange rate windows. After the subsidy removal, the devaluation accelerated sharply — the official rate went from approximately ₦465 per USD in May 2023 to over ₦1,595 by early 2025, before recovering slightly to around ₦1,370 by February 2026.

### 3.4 Before vs After — Summary Statistics

| Indicator | Pre-Removal avg | Post-Removal avg | Change |
|-----------|----------------|-----------------|--------|
| Inflation — All Items (%) | 16.89 | 27.28 | +61.5% |
| Inflation — Food (%) | 19.68 | 29.76 | +51.2% |
| Fuel Price (NGN/litre) | 160.72 | 800.43 | +398% |
| Exchange Rate (NGN/USD) | 418.29 | 1,293.39 | +209% |

![Combined Indicators](outputs/combined_indicators.png)

---

## 4. Structural Break Analysis — Chow Test

Before modelling, we formally test whether the relationship between fuel prices, exchange rates, and inflation changed at May 2023 — or whether the data is consistent with a single stable relationship across the full period.

The **Chow test** compares how well a single regression model fits the full dataset versus two separate models fitted on the pre and post periods. A significant F-statistic means the relationships genuinely shifted.

| | Result |
|---|---|
| F-statistic | **44.46** |
| p-value | **≈ 0.000** |
| Structural break detected | **Yes** |

An F-statistic of 44.46 is extremely large. The p-value of essentially zero means we can reject the null hypothesis of parameter stability with very high confidence. This is the headline statistical finding of the analysis: **the subsidy removal did not just change price levels — it changed how fuel prices and exchange rates relate to inflation.**

---

## 5. Counterfactual Analysis

To quantify the fuel shock's contribution to inflation, we use a counterfactual approach:

1. Fit a regression model on the pre-removal period using fuel price and exchange rate changes as predictors of inflation
2. Use that model to simulate what inflation would have looked like post-removal if fuel prices had stayed flat
3. Compare simulated vs actual inflation

![Counterfactual](outputs/counterfactual.png)

The chart shows three lines in the post-removal period:

- **Red (actual):** Inflation surged from 22% to a peak of 35%
- **Green dotted (counterfactual):** If fuel prices had stayed flat, the model suggests inflation would have remained around **16–17%** — close to its pre-removal trajectory
- **Blue dashed (model fitted):** What the pre-removal model predicts using actual post-removal fuel and FX changes — it tracks the direction but undershoots the actual surge

The gap between the green and red lines — roughly **10–18 percentage points** at peak — represents the estimated contribution of the fuel price shock to inflation. This is a conservative estimate since it uses the pre-removal regression relationships, which the Chow test shows had already weakened.

The model's undershoot of actual inflation is itself informative: even accounting for fuel and exchange rate changes, actual inflation rose higher than the pre-removal relationships would predict. This suggests the subsidy removal had **second-order effects** — on inflation expectations, supply chain costs, and business pricing behaviour — that are not captured by the variables in this dataset.

---

## 6. Feature Importance Across Models

![Feature Importance](outputs/feature_importance.png)

We trained four models (Linear Regression, Ridge, Lasso, Random Forest) using 10 features including fuel price levels, lagged fuel prices, exchange rate variables, and the post-subsidy removal dummy. The models are not good predictors of inflation (negative R² on the test set, as expected for a complex macroeconomic outcome), but their feature importance rankings reveal consistent patterns:

**Findings consistent across all models:**
- **`post_subsidy_removal`** is the dominant feature in Linear and Ridge regression — the structural break itself carries more explanatory weight than any continuous variable
- **`fuel_lag1` and `fuel_lag2`** consistently rank highly — inflation responds to fuel price changes with a 1–2 month delay, not immediately
- **`fuel_pct_change`** ranks in the top 3 across all models — the rate of change matters more than the level
- **Exchange rate variables rank consistently low** — despite the naira's dramatic depreciation, its direct month-to-month contribution to inflation was smaller than fuel prices in all four models

**Why the models don't predict well:**  
Inflation is determined by expectations, monetary policy decisions, global commodity prices, and supply chain dynamics — none of which are in this dataset. The poor predictive performance is expected and honest. The value of the models here is in understanding relative feature importance, not forecasting.

---

## 7. Key Findings

1. **A structural break is statistically confirmed.** The Chow test (F=44.46, p≈0) formally proves that the relationship between fuel prices, exchange rates, and inflation changed fundamentally at May 2023. This is not just a level shift — the transmission mechanism itself changed.

2. **Fuel prices were the primary driver of the inflation surge.** Monthly fuel price volatility increased 13x post-removal (0.57% vs 7.54% per month). The counterfactual analysis suggests inflation would have remained around 16–17% without the fuel shock, versus an actual peak of 35%.

3. **Inflation responds to fuel prices with a lag.** The 1–2 month lagged fuel price variables consistently outrank contemporaneous fuel prices across all models, suggesting businesses and households adjust prices with a delay after fuel cost increases.

4. **Exchange rate depreciation amplified but did not primarily drive inflation.** The naira lost over 200% of its value post-removal, but its direct month-to-month contribution to inflation was smaller than fuel prices in all modelling approaches.

5. **The shock had second-order effects.** Actual inflation exceeded what fuel and exchange rate changes alone would predict, suggesting the subsidy removal affected inflation expectations and business pricing behaviour beyond the direct input cost channel.

6. **Inflation has been declining since late 2024** but remains well above pre-removal levels, indicating the economy is adjusting but the adjustment is slow.

---

## 8. Limitations

- **Exchange rate data quality:** The 2021–2023 exchange rate series uses World Bank annual averages expanded to monthly, understating intra-year volatility
- **Fuel price interpolation:** NBS PMS reports are sparse for some months; linear interpolation was used to fill gaps
- **Missing confounders:** Global oil prices, CBN monetary policy decisions, food supply chain disruptions, and pre-existing naira pressure are not in the dataset
- **Small sample:** 74 months is a relatively small sample for time series modelling, limiting the statistical power of the regression analysis
- **Counterfactual assumptions:** The counterfactual assumes the pre-removal relationships would have held, which the Chow test itself shows is not true — the estimates should be interpreted as illustrative, not precise

---

## 9. Conclusion

Nigeria's May 2023 fuel subsidy removal was one of the most significant economic policy shocks in the country's recent history. This analysis formally confirms, using structural break testing, that the event changed not just price levels but the fundamental relationships between macroeconomic variables.

The fuel price shock was the dominant driver of the subsequent inflation surge, with the counterfactual analysis suggesting a 10–18 percentage point contribution to the inflation peak. Exchange rate depreciation played an amplifying role but was secondary in month-to-month dynamics. The shock also appears to have had second-order effects on inflation expectations and pricing behaviour that go beyond direct input cost transmission.

The Nigerian economy is now adjusting — inflation has been falling since late 2024 — but the adjustment is gradual and the price level remains far above where it was before the removal. Whether the long-run fiscal benefits of removing the subsidy outweigh the short-run distributional costs remains an open and important policy question.

---

*Data, code, and notebooks for this analysis are available at [github.com/kozah04/fuel-subsidy-impact](https://github.com/kozah04/fuel-subsidy-impact)*