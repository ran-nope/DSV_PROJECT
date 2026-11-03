<div align="center">

# The Perfect Product Launch
### Google Play Store · Exploratory Analysis · Feature Engineering · PCA

From messy app records to a reproducible view of app performance patterns.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pPh43M59JUzmIp6kPgybzvLeL_mE0lCg)
![Python](https://img.shields.io/badge/Python-3-3776AB?style=flat-square&logo=python&logoColor=white)
![Records](https://img.shields.io/badge/Clean_records-10%2C357-2E86AB?style=flat-square)
![PCA](https://img.shields.io/badge/PCA-95.08%25_variance-E4572E?style=flat-square)

[Explore the notebook](https://colab.research.google.com/drive/1pPh43M59JUzmIp6kPgybzvLeL_mE0lCg) · [Results](#results-at-a-glance) · [Visual gallery](#visual-gallery) · [Run locally](#run-the-analysis) · [Validation](#verified-execution)

</div>

---

## What this project explores

How do ratings, review counts, install buckets, prices, app size and update recency vary across the Google Play Store sample? This project cleans the source records, explores relationships, engineers features and uses principal component analysis to summarize the feature space.

**Domain:** Business, Retail & E-Commerce · **SDG 9:** Industry, Innovation & Infrastructure

## Results at a glance

| Measure | Verified result |
|:--|--:|
| Raw records | 10,841 |
| Exact duplicates removed | 483 |
| Malformed record quarantined | 1 |
| Clean records | **10,357** |
| Engineered dataset | **51 columns** |
| PCA candidates → selected features | 8 → 7 |
| Components retaining at least 95% variance | **5** |
| Actual variance retained | **95.08%** |
| Variance represented by 2D / 3D views | 59.11% / 72.72% |

The highly correlated `log_Installs` feature is removed at the `|r| > 0.85` threshold. PCA then uses rating, update age, app type, log reviews, log size, log price and log update age.

## Analysis workflow

~~~mermaid
flowchart TD
    A["Raw app records"] --> B["Validate, quarantine and clean"]
    B --> C["EDA and visual comparisons"]
    C --> D["Transform, scale and encode"]
    D --> E["Select features and fit PCA"]
    E --> F["Inspect charts and validate exports"]
~~~

## Explore interactively

Open the **Colab notebook**, connect to a Python runtime and choose **Runtime → Run all**. Each weekly stage has its own code cell and explanatory text. Expand outputs, edit parameters and rerun a cell to investigate the data.

<details>
<summary><b>Try three experiments</b></summary>

1. In Week 7, change `THRESHOLD = 0.85` to `0.90` or `0.80`. Rerun Week 7 and compare the selected features and variance curve.
2. In Week 6, compare before/after histograms and reported skewness. A log transform does not make every distribution symmetric.
3. In Week 7, change the top-category count or 3D sample size. The plot changes while PCA remains fitted on the full selected dataset.

Restore the original settings before comparing with the verified results above. Existing charts are static Matplotlib/Seaborn figures; the notebook is the place to run these experiments.

</details>

## Weekly source files

| Sprint | Focus | Source | Main output |
|:--|:--|:--|:--|
| Week 1 | Dataset and Pandas/NumPy basics | [Data hunting](week1_data_hunting.py) | Initial inspection |
| Week 2 | Schema, nulls and duplicate checks | [Know your data](week2_know_your_data.py) | `first_impression_report.txt` |
| Week 3 | Parsing, quarantine and imputation | [Cleaning sprint](week3_cleaning_sprint.py) | `googleplaystore_cleaned.csv` |
| Week 4 | EDA and correlations | [EDA deep dive](week4_eda_deep_dive.py) | EDA figures and insights |
| Week 6 | Transforms, scaling and encoding | [Feature engineering](week6_feature_engineering.py) | `googleplaystore_engineered.csv` |
| Week 7 | Feature selection and PCA | [Dimensionality reduction](week7_dimensionality_reduction.py) | `googleplaystore_pca.csv` |

Sprint numbering follows the uploaded files; the current repository has no separate Week 5 script.

## Run the analysis

Keep `googleplaystore.csv` in the **repository root**, alongside the scripts. Generated CSVs, reports and PNGs are saved to the current working directory.

<details>
<summary><b>Installation and execution commands</b></summary>

~~~bash
git clone https://github.com/ran-nope/DSV_PROJECT.git
cd DSV_PROJECT
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn missingno

python week1_data_hunting.py
python week2_know_your_data.py
python week3_cleaning_sprint.py
python week4_eda_deep_dive.py
python week6_feature_engineering.py
python week7_dimensionality_reduction.py
~~~

Week 6 requires Week 3's cleaned CSV. Week 7 requires Week 6's engineered CSV. Run all commands from the same directory.

</details>

## Verified execution

**Checked on 3 November 2026 in a Colab Python 3 runtime.** All six actual GitHub scripts were executed in two separate run directories, with a fresh namespace for each script. Both runs completed and produced identical engineered and PCA datasets.

| Check | Outcome |
|:--|:--|
| Six scripts compile and execute | Passed |
| Clean → engineered → PCA row counts | Passed: 10,357 |
| Rating range, numeric completeness and nonnegative values | Passed |
| Malformed rating-19 record excluded | Passed |
| StandardScaler means and MinMax bounds | Passed |
| Each one-hot group sums to one per record | Passed |
| Encoding report contains the Free/Paid mapping | Passed |
| PCA variance totals, cumulative curve and full reconstruction | Passed |
| Reduced export reaches 95% variance | Passed |
| Seven Week 6/7 chart files exist and are nonempty | Passed |
| Two runs produce identical exported CSV data | Passed |

The notebook validation cell saves logs, source SHA-256 fingerprints and `validation_runs/validation_summary.json`. Checks cover the supplied dataset and runtime, not arbitrary inputs.

## Methodology and interpretation

<details>
<summary><b>Cleaning, feature engineering and PCA decisions</b></summary>

- Remove exact duplicate records. Repeated app names can represent different observations and are retained.
- Quarantine malformed records before imputation; a stray `Free` install value is not converted to zero.
- Parse numeric fields and convert size to MB (KB divided by 1,024).
- Median-impute rating and size; mode-impute app type in the weekly pipeline.
- Preserve original reviews; store 99th-percentile capped values in a separate plot-only column.
- Measure update age against the latest date in the historical dataset, not today's date.
- Apply `log1p` to highly skewed non-rating columns; compare StandardScaler and MinMaxScaler.
- Label-encode app type; one-hot-encode content rating and ten largest categories plus OTHER.
- Exclude one-hot columns from PCA, filter highly correlated numeric features and standardize the selected inputs.

</details>

**Interpretation limits.** This is a historical Kaggle sample, not a current Play Store census. Installs are bucket lower bounds. Free apps do not establish a freemium business model. Imputed ratings are not observed user scores. A missingness chart alone cannot establish MCAR. PCA weights and correlations describe associations, not causal launch-success factors or revenue forecasts.

## Data and attribution

Dataset: [Google Play Store Apps — lava18 on Kaggle](https://www.kaggle.com/datasets/lava18/google-play-store-apps) · [Source CSV](googleplaystore.csv)

Project: **The Perfect Product Launch** · [Kavin M](https://github.com/ran-nope) · B.Tech Artificial Intelligence & Data Science, KCT

## Visual gallery

Figures and screenshots below come from the verified execution, rather than illustrative mockups.

### PCA explained variance

<img width="1650" height="750" alt="PCA individual and cumulative explained variance" src="https://github.com/user-attachments/assets/efd24094-3747-43cf-83d8-d974f0f597a2" />


### App profiles in two principal components

Colour compares app types and the five largest categories. This 2D view represents 59.11% of selected-feature variance.

<img width="2400" height="900" alt="PCA projections by app type and category" src="https://github.com/user-attachments/assets/fac13570-a630-4d36-a0cf-03f2d71fce77" />


