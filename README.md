# Statistical & Machine Learning Analysis of Airbnb Listings in NYC (2024)

MATH 895 course project analyzing **20,770 Airbnb listings** in New York City. The work combines data cleaning, exploratory analysis, hypothesis testing, regression, classification, and clustering to explain how location, property type, and host strategy shape nightly prices and guest demand.

**Author:** Harsh Bajpai

---

## Question

How do property characteristics, host behavior, and neighborhood factors influence pricing and guest demand in New York City’s short-term rental market?

The analysis is organized around six research questions:

1. Do nightly prices differ across boroughs and room types?
2. Which property and location features are the strongest predictors of price?
3. Do multi-listing hosts set different prices than single-listing hosts?
4. How does host strategy influence guest engagement (reviews)?
5. Can listings be classified into high-demand vs. low-demand segments?
6. What natural market clusters exist among listings?

---

## Repository contents

| File | Description |
| --- | --- |
| `Project Math 895/Airbnb_Project_Math_895.ipynb` | **Main notebook** — full analysis with executed outputs |
| `Math_895_Airbnb_Project.ipynb` | Earlier draft of the notebook |
| `airbnb_dataset.csv` | NYC Airbnb listings used in the analysis (~20.8k rows) |
| `Project Math 895/airbnb_dataset.csv` | Same dataset, kept with the final notebook |
| `895 Project description.pdf` | Course project prompt |
| `Project Math 895/895_Project_Report_HB.pdf` | Written project report |
| `Airbnb_Project_Math_895.pdf` | Exported notebook / report PDF |
| `895_Project_Presentation_HB.pptx` | Project presentation |

---

## Dataset

Each row is an Airbnb listing with fields such as:

- Location: `neighbourhood_group` (borough), `neighbourhood`, `latitude`, `longitude`
- Property: `room_type`, `price`, `bedrooms`, `beds`, `baths`, `minimum_nights`
- Host: `host_id`, `host_name`, `calculated_host_listings_count`
- Demand / quality: `number_of_reviews`, `reviews_per_month`, `rating`, `availability_365`, `license`

Hosts are coded as **Single** (exactly one listing) vs **Multi** (two or more listings).

### Cleaning notes

- Critical missing values (`price`, location, `room_type`) were rare (~0.03%) and those rows were dropped.
- The original `id` column was not a reliable primary key (thousands of colliding IDs). It was replaced with a synthetic `unique_id`.
- Unrealistic outliers were trimmed or capped:
  - Price kept in **$10–$1,000** (preserves ~99% of listings; raw max was $100,000)
  - `minimum_nights` capped at **365**
  - `reviews_per_month` capped at **10**
- Types were fixed for dates, ratings, bedrooms, and baths.

---

## Methods

| Stage | Approach |
| --- | --- |
| EDA | Univariate, bivariate, and multivariate views of price, room type, borough, and host type |
| Inferential tests | ANOVA / Welch ANOVA on log-price, Kruskal–Wallis, Mann–Whitney U |
| Price modeling | OLS / linear regression on `log(price)`; HistGradientBoostingRegressor with permutation importance |
| Engagement | Mann–Whitney U plus OLS on log reviews and availability-adjusted engagement |
| Demand classification | Logistic regression, Random Forest, HistGradientBoostingClassifier |
| Segmentation | K-Means (`k = 4`) after standardizing price, rating, reviews, availability, and host type; PCA for visualization |

---

## Key results

### Pricing

- Prices differ significantly by **borough** and **room type** (ANOVA, Welch ANOVA, and Kruskal–Wallis all `p < 0.001`).
- Manhattan is the premium borough (median about **$150**), then Brooklyn (**$124**). Queens, the Bronx, and Staten Island are cheaper (medians under **$100**).
- Hotel rooms and entire homes cost the most; private and shared rooms sit at the low end.
- Most expensive slice: **Manhattan hotel rooms** (median ~$235). Cheapest: **shared rooms in the Bronx / Queens** (~$50).

### What predicts price

Test-set performance on log-price:

| Model | R² | RMSE (log scale) |
| --- | --- | --- |
| Linear regression | 0.447 | 0.488 |
| HistGradientBoosting | **0.546** | **0.442** |

Permutation importance (tree model) ranks **room type** first, then **borough**, then property size (`beds` / `bedrooms` / `baths`). Ratings and host type matter less; reviews and availability barely move price. Pricing in this market is mostly supply-side: what the listing is and where it is, not how busy it looks.

### Host strategy

- **56%** of listings are run by multi-listing hosts; **44%** by single-listing hosts. Most *hosts* still have one listing, but a large share of *supply* is professional.
- Single hosts charge more (median **$145** vs **$110**) and get more reviews per month (**0.81** vs **0.54**). Both gaps are significant (Mann–Whitney U, `p < 0.001`).
- After controlling for room type, borough, size, and rating, single hosts still charge about **17%** more (`OLS R² = 0.435`, host-type coefficient `+0.159`, `p < 0.001`).
- Raw review volume is slightly higher for single hosts (~**+1.4%**), but that gap **disappears** once availability and listing traits are controlled. Multi-hosts run more inventory at lower per-listing engagement; they are not less efficient once occupancy opportunity is accounted for.

### Demand classification

Listings are labeled high/low demand at the median of `reviews_per_month` (balanced ~50/50 split).

| Model | Accuracy | ROC-AUC |
| --- | --- | --- |
| Logistic regression | 0.593 | 0.634 |
| Random Forest | 0.714 | 0.786 |
| HistGradientBoosting | **0.755** | **0.825** |

Demand is nonlinear. High-demand listings tend to be lower-priced, higher-rated, often private/shared rooms in Brooklyn or Queens. Low-demand listings are more often expensive entire homes or hotel units in Manhattan, frequently run by multi-hosts.

### Market clusters (K-Means, k = 4)

| Cluster | Profile | Size (approx.) |
| --- | --- | --- |
| 0 | Mid-priced, high-rated **single-host** listings | 7,345 |
| 1 | High-availability **multi-host** operators | 4,501 |
| 2 | Low-rated, low-demand listings | 1,297 |
| 3 | Budget, high-turnover, high-review listings | 3,638 |

---

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open `Project Math 895/Airbnb_Project_Math_895.ipynb` in Jupyter or VS Code / Cursor. The notebook loads `airbnb_dataset.csv` from the working directory — run it from `Project Math 895/` or update the read path to `../airbnb_dataset.csv`.

---

## Stack

Python, pandas, NumPy, matplotlib, seaborn, SciPy, statsmodels, scikit-learn.

---

## Disclaimer

This is an academic analysis of publicly available Airbnb listing data. It is not affiliated with Airbnb and is not investment, pricing, or policy advice.
