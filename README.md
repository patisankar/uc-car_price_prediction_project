# Used-Car Price Analysis

**Notebook:** [used_car_price_analysis.ipynb](used_car_price_analysis.ipynb)
**Data:** [data/](data/README.md) (~427,000 Craigslist listings; 356,126 after cleaning)

## Summary

A used-car marketplace wants to know what drives listing price. The notebook cleans the listings, compares Linear, Ridge and Lasso regression on log price, and segments the market with K-Means.

## Method

- Cleaning: dropped columns with >60% missing, removed price = 0 and rows missing price/year/odometer, mode-imputed other categoricals, trimmed price/year/odometer outliers with an IQR rule.
- Features: vehicle age, log odometer, mileage per year, ordered condition and title status, one-hot fuel/transmission/drive/type/colour.
- Manufacturer and state are target-encoded using the **training split only**, so test prices do not leak into the features.
- Models: Linear, Ridge, Lasso with 5-fold cross-validation and GridSearchCV over alpha.
- Metrics: R² to compare models (share of log-price variance explained); RMSE and MAE to judge accuracy, converted to dollars.

## Results (20% held-out test set)

| Model | CV R² | Test R² | Test RMSE (log) | Test MAE (log) |
|---|---|---|---|---|
| Linear Regression | 0.6018 | 0.6035 | 0.5483 | 0.3518 |
| Ridge (α≈13.7) | 0.6018 | 0.6035 | 0.5483 | 0.3518 |
| Lasso (α≈0.00015) | 0.6018 | 0.6035 | 0.5483 | 0.3518 |

In dollars, the typical error is about **$8,060 (RMSE)** and **$5,332 (MAE)**, against a median listing price of $15,920. The 0.80 R² target was **not met**.

## Findings

- Vehicle age is the strongest driver (permutation importance 0.78 R² drop), then fuel type (0.15) and mileage per year (0.09). Importance shows what the model relies on, not what causes price.
- Cars aged 0–5 years have a median price of $32,990, against $22,995 for 6–10 years (30% lower).
- All three models perform the same, so features, not model type, limit accuracy.
- K-Means found two segments: newer, low-mileage cars (avg $28,347, 8 yrs, ~36,700 mi, 148,329 listings) and older, high-mileage cars (avg $11,385, 14 yrs, ~133,200 mi, 207,797 listings).

## Recommendations

- Price and buy mainly by age and mileage; adjust for fuel type and brand.
- Use the model as a guide, not a quote: it explains about 60% of price differences.
- Next steps: add listing dates, location and sale outcomes; try tree-based models; refit the encoder inside each CV fold.

## Files

```
used_car_price_analysis.ipynb   analysis notebook with saved outputs
data/                           dataset and data README
images/                         plots saved by the notebook (created in code)
```

## Requirements

pandas, numpy, matplotlib, seaborn, scikit-learn
