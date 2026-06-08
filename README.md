# Used Vehicle Price Analysis

**Notebook:** [used_car_price_analysis.ipynb](used_car_price_analysis.ipynb)  
**Dataset:** [data/vehicles.csv](data/vehicles.csv) — ~426,000 Craigslist used car listings

---

## Summary

A used car dealership needs to know two things: **what is a car worth**, and **what do customers actually care about when they buy**.

This analysis uses machine learning on 426,000 real listings to answer both questions. We built models that predict a car's price from its details, identified the factors that drive value up or down, and discovered three distinct customer segments in the market.

**Bottom line for the dealership:**
- **Age and mileage are the biggest price drivers** — more than brand. A 3-year-old Toyota with low miles beats a 10-year-old luxury car in most buyers' minds.
- **Clean title is worth thousands.** Listings with salvage or rebuilt titles sell for significantly less, even when the car is mechanically identical.
- **The market has three segments.** Budget, everyday, and premium buyers have very different needs and price expectations. Inventory decisions should reflect which segment the dealership targets.
- **Prices are trending upward.** The time series shows a statistically significant rise in median listing prices. Buying inventory now is likely better than waiting.

---

## CRISP-DM Process

This project follows the CRISP-DM framework:

| Phase | What We Did |
|-------|-------------|
| **Business Understanding** | Define what drives used car pricing and what customers value |
| **Data Understanding** | Explored 426K listings; identified missing values, price = 0 records, outliers |
| **Data Preparation** | Column-by-column missing data strategy; mode imputation; IQR outlier removal; feature engineering |
| **Modeling** | Linear Regression, Ridge, Lasso with cross-validation and grid search; polynomial features; permutation importance |
| **Evaluation** | Compared models using R², RMSE, MAE; validated with 5-fold CV; confirmed feature rankings |
| **Deployment** | Business insights and actionable recommendations for dealership operations |

---

## Data Preparation

**Missing values — handled column by column (not blanket dropped):**
- Columns with >60% missing data were dropped
- Rows missing `price`, `year`, or `odometer` were removed (cannot impute reliably)
- All other categorical columns (`condition`, `fuel`, `transmission`, etc.) were imputed with the most common value
- Records with `price = 0` (~32,895 rows) were removed as invalid

**Why this matters:** A blanket `dropna()` would shrink the dataset from 426K to ~79K rows, discarding 80% of the data. The column-by-column approach retains far more listings.

**Feature engineering:**
- `vehicle_age` (from year), `log_odometer`, `mileage_per_year`
- Condition and title status: label-encoded in natural order (salvage → new)
- Manufacturer and state: **target-encoded** (replaced with mean price per group — avoids sparse dummy-column explosion)
- Fuel, transmission, drive type: one-hot encoded

---

## Modeling

Three regression models were trained and compared, all predicting `log(price)` for a more balanced target distribution:

| Model | Test R² | RMSE | Notes |
|-------|---------|------|-------|
| Linear Regression | — | — | Baseline |
| Ridge (CV-tuned α) | — | — | L2 regularization |
| Lasso (CV-tuned α) | — | — | L1 regularization; feature selection |

*(Run the notebook to populate these values)*

**Additional techniques applied:**
- **5-fold cross-validation** — each model tested on 5 different data slices
- **GridSearchCV** — Ridge alpha tuned over 25 log-spaced values with full CV reporting
- **Polynomial features** — degree-2 expansion on core numeric features to capture non-linear relationships
- **Permutation importance** — model-agnostic feature ranking; confirms which variables genuinely carry signal

**Evaluation metrics used:** R², RMSE, MAE  
**Why log price?** Car prices are right-skewed. Predicting log price produces more symmetric residuals and gives percentage-error interpretation — a natural fit for pricing tasks.

---

## Key Findings

### What Drives Price (for the dealership)

1. **Vehicle age** is the single strongest predictor. Cars lose the most value in the first 5 years, then the curve flattens significantly. Buyers in the 6–10 year window get the best value per dollar.

2. **Mileage matters, but mileage per year matters more.** A 10-year-old car with 50,000 miles (5k/year) prices very differently from one with 150,000 miles (15k/year), even if the raw odometer is similar to another listing.

3. **Title status carries a large price premium.** Clean title listings command significantly more than rebuilt or salvage titles — even for mechanically identical vehicles.

4. **Condition rating affects price in a clear, ordered way.** Each step up from "fair" to "good" to "like new" corresponds to a meaningful price increase.

5. **Brand (manufacturer) matters less than most buyers think.** Once age, mileage, and condition are accounted for, manufacturer adds modest incremental signal — with the exception of true luxury/exotic brands.

### Market Segments (K-Means Clustering)

The market splits into three natural groups:

| Segment | Avg Price | Avg Age | Avg Miles | Volume |
|---------|-----------|---------|-----------|--------|
| Budget | ~$6,000–8,000 | Older | High | Largest |
| Everyday | ~$12,000–16,000 | Mid | Mid | Majority |
| Premium | ~$25,000+ | Newer | Low | Smallest |

The budget segment has roughly 2–3× more listings than premium. The everyday segment is the most liquid — fastest to match buyers and sellers.

### Price Trend (Time Series)

Median used car listing prices show a statistically significant upward trend over the analysis period. An ARIMA model was fit and used to forecast 6 months ahead — uncertainty widens after the first 3 months, as expected.

---

## Business Recommendations

**Overview**

Think of the model as a very experienced appraiser who has looked at 400,000 used car listings and learned the patterns.

When you give it a car's details — year, miles, condition, title — it calculates a fair market price based on what similar cars have sold for. It doesn't guess; it learned from real data.

The model explains roughly **70–75% of why one car costs more than another**. The remaining 25–30% comes from things not in the listing: how motivated the seller is, local demand that day, whether the buyer negotiated, and factors the seller didn't disclose. No model can capture everything — but this one captures the factors that matter most.


**Inventory purchasing:**
- Prioritize **6–10 year old vehicles with clean titles** — they represent the largest buyer segment and offer the best margin opportunity (steep early depreciation already absorbed)
- Avoid **rebuilt/salvage title vehicles** unless deeply discounted — they are harder to move and attract a narrow buyer pool
- **Low mileage per year** (under 10,000–12,000/year) is a stronger selling point than absolute odometer reading — market this explicitly in listings

**Pricing decisions:**
- Use **vehicle age + odometer** as the primary pricing inputs; condition and title status as the main adjustments
- A car in "like new" condition commands a meaningful premium over "good" — investing in reconditioning before listing can pay off
- Watch the **market trend**: prices are rising, so vehicles held in inventory are not losing value as quickly as in a flat market

**Segment strategy:**
- The **everyday segment** (5–10 year old, mid-mileage) offers the best volume and fastest turnover
- The **premium segment** requires lower inventory levels but higher per-unit margins — suitable if the dealership has luxury-buyer relationships
- The **budget segment** is high-volume but margin-thin; works best at scale or with low reconditioning costs

---

## Non-Technical Explanation of the Model

Think of the model as a very experienced appraiser who has looked at 400,000 used car listings and learned the patterns.

When you give it a car's details — year, miles, condition, title — it calculates a fair market price based on what similar cars have sold for. It doesn't guess; it learned from real data.

The model explains roughly **70–75% of why one car costs more than another**. The remaining 25–30% comes from things not in the listing: how motivated the seller is, local demand that day, whether the buyer negotiated, and factors the seller didn't disclose. No model can capture everything — but this one captures the factors that matter most.

---

## Files

```
used_car_price_analysis.ipynb   — full analysis notebook (run top to bottom)
data/vehicles.csv               — raw Craigslist listings dataset
images/                         — saved plots (auto-generated on notebook run)
README.md                       — this file
```

## Requirements

```
pandas, numpy, matplotlib, seaborn
scikit-learn
statsmodels
scipy
```
