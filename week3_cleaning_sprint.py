import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

DATA_PATH = "googleplaystore.csv"
df = pd.read_csv(DATA_PATH)

print("Shape before cleaning:", df.shape)
# Reject malformed or shifted records before any imputation.
valid_rating = pd.to_numeric(df["Rating"], errors="coerce")
valid_installs = pd.to_numeric(df["Installs"].astype(str).str.replace(r"[,+]", "", regex=True), errors="coerce")
valid_price = pd.to_numeric(df["Price"].astype(str).str.replace("$", "", regex=False), errors="coerce")
valid_reviews = pd.to_numeric(df["Reviews"], errors="coerce")
valid = (valid_rating.isna() | valid_rating.between(1, 5)) & valid_installs.ge(0) & valid_price.ge(0) & valid_reviews.ge(0)
df.loc[~valid].to_csv("week3_quarantined_records.csv", index=False)
df = df.loc[valid].copy()

# 1. REMOVE DUPLICATES

before = df.shape[0]
df = df.drop_duplicates()
print(f"Removed {before - df.shape[0]} duplicate rows.")


# 2. CLEANING "Installs" -> numeric
#    e.g. "10,000+"  ->  10000

df["Installs"] = (
    df["Installs"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("+", "", regex=False)

)
df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")


# 3. CLEANING "Price" -> numeric
#    e.g. "$4.99" -> 4.99 , "0" -> 0.0

df["Price"] = df["Price"].astype(str).str.replace("$", "", regex=False)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# 4. CLEANING "Size" -> numeric (in MB)
#    e.g. "19M" -> 19.0 , "Varies with device" -> NaN (imputed later)

def clean_size(value):
    value = str(value)
    if "Varies with device" in value:
        return np.nan
    if value.endswith("M"):
        return float(value.replace("M", ""))
    if value.endswith("k"):
        return float(value.replace("k", "")) / 1024   # convert KB to MB
    return np.nan

df["Size"] = df["Size"].apply(clean_size)

# 5. CLEANING "Reviews" -> numeric

df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

print("\nData types after cleaning:")
print(df[["Installs", "Price", "Size", "Reviews", "Rating"]].dtypes)

# 6. DESCRIPTIVE STATISTICS: CENTRAL TENDENCY & DISPERSION
#    Mean/Median/Mode, Range/Variance/Std Dev/IQR,Skewness & Kurtosis


numeric_cols = ["Rating", "Reviews", "Size", "Installs", "Price"]

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS (before imputation)")
print("=" * 60)
desc_rows = []
for col in numeric_cols:
    series = df[col].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    desc_rows.append({
        "Column": col,
        "Mean": round(series.mean(), 2),
        "Median": round(series.median(), 2),
        "Mode": round(series.mode()[0], 2) if not series.mode().empty else np.nan,
        "Range": round(series.max() - series.min(), 2),
        "Variance": round(series.var(), 2),
        "Std Dev": round(series.std(), 2),
        "IQR": round(q3 - q1, 2),
        "Skewness": round(series.skew(), 2),
        "Kurtosis": round(series.kurt(), 2),
    })
desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))
print(
    "\nReading the skewness column: values near 0 = roughly symmetric, "
    "positive = right-skewed (long tail of high values), negative = "
    "left-skewed. This is exactly why we use the MEDIAN (not the mean) "
    "to impute skewed columns like Rating and Size below."
)


# 7. MISSING DATA VISUALIZATION (before imputation)
#    Missingness Matrix / Heatmap

print("\nMissing values BEFORE imputation:")
print(df[["Rating", "Size", "Type"]].isnull().sum())

plt.figure(figsize=(10, 5))
try:
    import missingno as msno
    msno.matrix(df[["Rating", "Reviews", "Size", "Installs", "Price", "Type"]])
    plt.title("Missingness Matrix (missingno)")
except ImportError:
    # Fallback if missingno isn't installed: a boolean heatmap does the
    # same job - yellow/light cells mark a missing value in that column.
    sns.heatmap(
        df[["Rating", "Reviews", "Size", "Installs", "Price", "Type"]].isnull(),
        cbar=False, cmap="viridis"
    )
    plt.title("Missing Data Heatmap (True = missing)")
plt.tight_layout()
plt.savefig("week3_missing_data_matrix.png", dpi=150)
plt.show()
plt.close()
print("Saved missingness visualization to week3_missing_data_matrix.png")
print(
    "Missingness visualization is descriptive only. "
    "It cannot establish MCAR. "
    "Median/mode imputation is a baseline; final-review summaries preserve observed ratings."
)


# 8. HANDLING MISSING VALUES (Imputation)

# Rating -> numeric, skewed -> use MEDIAN (robust to outliers/skew)
df["Rating"] = df["Rating"].fillna(df["Rating"].median())

# Size -> numeric -> use MEDIAN as well ("Varies with device" left many NaNs)
df["Size"] = df["Size"].fillna(df["Size"].median())

# Type -> categorical -> use MODE (most frequent category)
df["Type"] = df["Type"].fillna(df["Type"].mode()[0])

print("\nMissing values AFTER imputation:")
print(df[["Rating", "Size", "Type"]].isnull().sum())


# 9. OUTLIER DETECTION - BOXPLOTS (visual identification, Unit 2)

fig, axes = plt.subplots(1, len(numeric_cols), figsize=(20, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax, color="skyblue")
    ax.set_title(f"Boxplot: {col}")
plt.tight_layout()
plt.savefig("week3_boxplots.png", dpi=150)
plt.show()
plt.close()
print("\nSaved boxplots to week3_boxplots.png")


# 10. OUTLIER DETECTION - IQR METHOD
#     outliers are below Q1 - 1.5*IQR or above Q3 + 1.5*IQR

print("\nOutlier detection using the IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR):")
iqr_summary = {}
for col in numeric_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    iqr_summary[col] = n_outliers
    print(f"  {col}: bounds=({lower_bound:.2f}, {upper_bound:.2f}) -> {n_outliers} outliers")

# 11. OUTLIER DETECTION - Z-SCORE METHOD (cross-check against IQR)

print("\nOutlier detection using Z-score (|z| > 3):")
z_summary = {}
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    n_outliers = (z_scores > 3).sum()
    z_summary[col] = n_outliers
    print(f"  {col}: {n_outliers} potential outliers")

print(
    "\nNote: IQR usually flags more borderline points than Z-score on "
    "skewed columns like Installs/Price - this is expected and worth "
    "mentioning in the report as a comparison of the two methods."
)

# Example: capping Reviews outliers at the 99th percentile
# a common strategy so that a few viral apps don't distort later analysis
upper_cap = df["Reviews"].quantile(0.99)
df["Reviews_capped_for_plot"] = np.minimum(df["Reviews"], upper_cap)
print(f"\nCreated a separate plot-only Reviews column capped at the 99th percentile ({upper_cap:.0f}).")

# 9. SAVE CLEANED DATASET

df.to_csv("googleplaystore_cleaned.csv", index=False)
print("\nCleaned dataset saved as googleplaystore_cleaned.csv")
print("Final shape:", df.shape)
