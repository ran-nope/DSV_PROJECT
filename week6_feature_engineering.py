"""
WEEK 6 : FEATURE ENGINEERING
Syllabus Mapping : Unit 3
----------------------------------------------------------------
Goals for this week:
1. Measure skewness and fix it with log transformations
2. Apply feature scaling (StandardScaler and MinMaxScaler)
3. Encode categorical variables (Label + One-Hot)
4. Show visual proof (Before vs After histograms/KDEs)
5. Save an engineered dataset for Week 7 (PCA)

Input  : googleplaystore_cleaned.csv   (created by week3_cleaning_sprint.py)
Output : googleplaystore_engineered.csv, week6_*.png, week6_feature_report.txt
----------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

sns.set_style("whitegrid")

DATA_PATH = "googleplaystore_cleaned.csv"
df = pd.read_csv(DATA_PATH)
print("Loaded cleaned data:", df.shape)

# ----------------------------------------------------------------
# 0. SAFETY CHECKS
# ----------------------------------------------------------------
# The raw Play Store file has one shifted row (Rating = 19, Category = "1.9").
# Ratings must lie between 1 and 5, so anything else is a data-entry error.
bad_rows = (~df["Rating"].between(1, 5)).sum()
df = df[df["Rating"].between(1, 5)].copy()
print(f"Dropped {bad_rows} row(s) with an impossible Rating value.")

# ----------------------------------------------------------------
# 1. DATE FEATURE : how stale is each app?
# ----------------------------------------------------------------
df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")
reference_date = df["Last Updated"].max()
df["days_since_update"] = (reference_date - df["Last Updated"]).dt.days
df["days_since_update"] = df["days_since_update"].fillna(df["days_since_update"].median())

# ----------------------------------------------------------------
# 2. SKEWNESS ANALYSIS (decides which columns need a transformation)
# ----------------------------------------------------------------
numeric_cols = ["Rating", "Reviews", "Size", "Installs", "Price", "days_since_update"]
skew_before = df[numeric_cols].skew().round(2)
print("\nSkewness BEFORE transformation:")
print(skew_before.to_string())

# Rule of thumb: |skew| > 1 is highly skewed
skewed_cols = skew_before[skew_before.abs() > 1].index.tolist()
skewed_cols = [c for c in skewed_cols if c != "Rating"]  # Rating is bounded 1-5
print("\nColumns selected for log transformation:", skewed_cols)

# ----------------------------------------------------------------
# 3. FIXING SKEWNESS : log(1 + x)
#    log1p is used instead of log because Price and Reviews contain zeros
# ----------------------------------------------------------------
for col in skewed_cols:
    df[f"log_{col}"] = np.log1p(df[col])

log_cols = [f"log_{c}" for c in skewed_cols]
skew_after = df[log_cols].skew().round(2)
print("\nSkewness AFTER transformation:")
print(skew_after.to_string())

# ----------------------------------------------------------------
# 4. VISUAL PROOF : Before vs After histograms with KDE
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, len(skewed_cols), figsize=(5 * len(skewed_cols), 8))
axes = np.atleast_2d(axes)
for i, col in enumerate(skewed_cols):
    sns.histplot(df[col], kde=True, ax=axes[0, i], color="#E4572E", bins=40)
    axes[0, i].set_title(f"{col} (before)\nskew = {skew_before[col]}")
    sns.histplot(df[f"log_{col}"], kde=True, ax=axes[1, i], color="#17BEBB", bins=40)
    axes[1, i].set_title(f"log1p({col}) (after)\nskew = {skew_after[f'log_{col}']}")
plt.suptitle("Fixing Skewness: Before vs After Log Transformation", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("week6_skewness_before_after.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------------------------------------------
# 5. FEATURE SCALING : StandardScaler vs MinMaxScaler
# ----------------------------------------------------------------
scale_cols = ["Rating"] + log_cols + ["days_since_update"]
scale_cols = [c for c in scale_cols if c in df.columns]

std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()

std_vals = std_scaler.fit_transform(df[scale_cols])
mm_vals = mm_scaler.fit_transform(df[scale_cols])

for j, col in enumerate(scale_cols):
    df[f"{col}_std"] = std_vals[:, j]
    df[f"{col}_mm"] = mm_vals[:, j]

print("\nScaled feature summary (StandardScaler: mean ~ 0, std ~ 1):")
print(pd.DataFrame(std_vals, columns=scale_cols).describe().loc[["mean", "std"]].round(3).to_string())
print("\nScaled feature summary (MinMaxScaler: range 0 to 1):")
print(pd.DataFrame(mm_vals, columns=scale_cols).describe().loc[["min", "max"]].round(3).to_string())

# Compare one column across the three versions
demo = log_cols[0] if log_cols else "Rating"
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(df[demo], kde=True, ax=axes[0], color="#6C757D")
axes[0].set_title(f"{demo} (original scale)")
sns.histplot(df[f"{demo}_std"], kde=True, ax=axes[1], color="#2E86AB")
axes[1].set_title(f"{demo} (StandardScaler)")
sns.histplot(df[f"{demo}_mm"], kde=True, ax=axes[2], color="#F18F01")
axes[2].set_title(f"{demo} (MinMaxScaler)")
plt.tight_layout()
plt.savefig("week6_scaling_comparison.png", dpi=150)
plt.show()
plt.close()

# ----------------------------------------------------------------
# 6. ENCODING CATEGORICAL VARIABLES
# ----------------------------------------------------------------
# 6a. Label encoding : Type is binary (Free / Paid)
le = LabelEncoder()
df["Type_enc"] = le.fit_transform(df["Type"])
print("\nLabel encoding for Type:", {c: int(i) for i, c in enumerate(le.classes_)})

# 6b. One-hot encoding : Content Rating has a handful of unordered categories
df = pd.get_dummies(df, columns=["Content Rating"], prefix="CR", dtype=int)

# 6c. One-hot on Category : 33 categories are too many, so keep the top 10
#     and group the rest as "Other" to avoid a huge sparse matrix
top_categories = df["Category"].value_counts().head(10).index
df["Category_grouped"] = np.where(df["Category"].isin(top_categories), df["Category"], "OTHER")
df = pd.get_dummies(df, columns=["Category_grouped"], prefix="Cat", dtype=int)

print("Top 10 categories kept for one-hot encoding:")
print(list(top_categories))

# ----------------------------------------------------------------
# 7. SAVE FEATURE REPORT + ENGINEERED DATASET
# ----------------------------------------------------------------
with open("week6_feature_report.txt", "w", encoding="utf-8") as f:
    f.write("WEEK 6 FEATURE ENGINEERING REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write("Skewness before transformation:\n" + skew_before.to_string() + "\n\n")
    f.write("Skewness after log1p transformation:\n" + skew_after.to_string() + "\n\n")
    f.write(f"Scaled columns: {scale_cols}\n\n")
    f.write(f"Label encoded: Type -> {{c: int(i) for i, c in enumerate(le.classes_)}}\n")
    f.write("One-hot encoded: Content Rating, Category (top 10 + OTHER)\n")

df.to_csv("googleplaystore_engineered.csv", index=False)
print("\nSaved: googleplaystore_engineered.csv, week6_feature_report.txt")
print("Final shape:", df.shape)
