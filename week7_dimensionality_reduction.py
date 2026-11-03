"""
WEEK 7 : DIMENSIONALITY REDUCTION
Syllabus Mapping : Unit 4
----------------------------------------------------------------
Goals for this week:
1. Feature selection: correlation matrix + removal of multicollinear features
2. PCA on the standardised feature space
3. Visualise the Explained Variance Ratio (scree + cumulative)
4. 2D and 3D scatter plots of the Principal Components
5. Interpret components with a loadings heatmap

Input  : googleplaystore_engineered.csv   (created by week6_feature_engineering.py)
Output : googleplaystore_pca.csv, week7_*.png, week7_pca_report.txt
----------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

DATA_PATH = "googleplaystore_engineered.csv"
df = pd.read_csv(DATA_PATH)
print("Loaded engineered data:", df.shape)

# ----------------------------------------------------------------
# 1. CANDIDATE FEATURES
#    PCA works on continuous variance, so we use the transformed numeric
#    features plus the binary Type flag. One-hot columns (Category, Content
#    Rating) are kept out of PCA because they would swamp the variance with
#    sparse 0/1 dummies; they are used later to colour the scatter plots.
# ----------------------------------------------------------------
numeric_features = [c for c in df.columns if c.startswith("log_") and not c.endswith(("_std", "_mm"))]
candidate_features = ["Rating", "days_since_update", "Type_enc"] + numeric_features
candidate_features = [c for c in candidate_features if c in df.columns]
print(f"\nCandidate features ({len(candidate_features)}):")
print(candidate_features)

# ----------------------------------------------------------------
# 2. CORRELATION MATRIX
# ----------------------------------------------------------------
corr = df[candidate_features].corr()

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, cbar_kws={"label": "Pearson r"})
plt.title("Correlation Matrix (before feature selection)")
plt.tight_layout()
plt.savefig("week7_correlation_matrix.png", dpi=150)
plt.show()
plt.close()

# ----------------------------------------------------------------
# 3. FEATURE SELECTION : drop one feature from each highly correlated pair
# ----------------------------------------------------------------
THRESHOLD = 0.85
corr_all = df[candidate_features].corr().abs()
upper = corr_all.where(np.triu(np.ones(corr_all.shape), k=1).astype(bool))

to_drop = []
print(f"\nPairs with |r| > {THRESHOLD} (multicollinearity):")
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and val > THRESHOLD and col not in to_drop:
            print(f"  {row}  <->  {col}   r = {val:.2f}  -> dropping '{col}'")
            to_drop.append(col)

selected_features = [c for c in candidate_features if c not in to_drop]
print(f"\nDropped {len(to_drop)} feature(s): {to_drop}")
print(f"Kept {len(selected_features)} feature(s) for PCA.")

# ----------------------------------------------------------------
# 4. STANDARDISE (PCA is variance-based, so scale matters)
# ----------------------------------------------------------------
X = df[selected_features].astype(float)
X_scaled = StandardScaler().fit_transform(X)

# ----------------------------------------------------------------
# 5. PCA : full decomposition first, to study explained variance
# ----------------------------------------------------------------
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

evr = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr)
n_95 = int(np.argmax(cum_evr >= 0.95) + 1)
n_80 = int(np.argmax(cum_evr >= 0.80) + 1)
print(f"\nComponents needed for 80% variance: {n_80}")
print(f"Components needed for 95% variance: {n_95}  (from {len(selected_features)} features)")

# ----------------------------------------------------------------
# 6. EXPLAINED VARIANCE RATIO PLOT (scree + cumulative)
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(1, len(evr) + 1)
ax.bar(x, evr, color="#2E86AB", alpha=0.85, label="Individual")
ax.plot(x, cum_evr, color="#E4572E", marker="o", linewidth=2, label="Cumulative")
ax.axhline(0.95, color="grey", linestyle="--", linewidth=1)
ax.axvline(n_95, color="grey", linestyle="--", linewidth=1)
ax.text(len(evr) * 0.55, 0.96, f"95% variance at PC{n_95}", color="grey")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("PCA Explained Variance Ratio")
ax.set_xticks(x)
ax.legend(loc="center right")
plt.tight_layout()
plt.savefig("week7_explained_variance.png", dpi=150)
plt.show()
plt.close()

# ----------------------------------------------------------------
# 7. REDUCED PCA (3 components for plotting, n_95 for the reduced dataset)
# ----------------------------------------------------------------
pca3 = PCA(n_components=3, random_state=42)
pcs = pca3.fit_transform(X_scaled)
df["PC1"], df["PC2"], df["PC3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]
print("\nVariance captured by PC1, PC2, PC3:",
      [f"{v:.1%}" for v in pca3.explained_variance_ratio_],
      f"(total {pca3.explained_variance_ratio_.sum():.1%})")

# ----------------------------------------------------------------
# 8. 2D SCATTER OF PRINCIPAL COMPONENTS
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x="PC1", y="PC2", hue="Type", alpha=0.5, s=15,
                palette={"Free": "#2E86AB", "Paid": "#E4572E"}, ax=axes[0])
axes[0].set_title("PC1 vs PC2 by App Type")
axes[0].set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]:.1%})")

top_cats = df["Category"].value_counts().head(5).index
sub = df[df["Category"].isin(top_cats)]
sns.scatterplot(data=sub, x="PC1", y="PC2", hue="Category", alpha=0.5, s=15,
                palette="Set2", ax=axes[1])
axes[1].set_title("PC1 vs PC2 for the Top 5 Categories")
axes[1].set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]:.1%})")

plt.tight_layout()
plt.savefig("week7_pca_2d_scatter.png", dpi=150)
plt.show()
plt.close()

# ----------------------------------------------------------------
# 9. 3D SCATTER OF PRINCIPAL COMPONENTS
# ----------------------------------------------------------------
sample = df.sample(n=min(3000, len(df)), random_state=42)
colors = sample["Type"].map({"Free": "#2E86AB", "Paid": "#E4572E"})

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(sample["PC1"], sample["PC2"], sample["PC3"], c=colors, alpha=0.5, s=10)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("First Three Principal Components (3,000 sampled apps)")
plt.tight_layout()
plt.savefig("week7_pca_3d_scatter.png", dpi=150)
plt.show()
plt.close()

# ----------------------------------------------------------------
# 10. LOADINGS HEATMAP : what does each component actually mean?
# ----------------------------------------------------------------
loadings = pd.DataFrame(
    pca3.components_.T,
    index=selected_features,
    columns=["PC1", "PC2", "PC3"],
)
# keep the strongest contributors so the heatmap stays readable
top_idx = loadings.abs().max(axis=1).sort_values(ascending=False).head(10).index

plt.figure(figsize=(8, 8))
sns.heatmap(loadings.loc[top_idx], annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("PCA Loadings (top 10 features)")
plt.tight_layout()
plt.savefig("week7_pca_loadings.png", dpi=150)
plt.show()
plt.close()

print("\nTop drivers of each component:")
for pc in ["PC1", "PC2", "PC3"]:
    top = loadings[pc].abs().sort_values(ascending=False).head(3).index.tolist()
    print(f"  {pc}: {top}")

# ----------------------------------------------------------------
# 11. SAVE REDUCED DATASET + REPORT
# ----------------------------------------------------------------
pca_n = PCA(n_components=n_95, random_state=42)
reduced = pca_n.fit_transform(X_scaled)
reduced_df = pd.DataFrame(reduced, columns=[f"PC{i + 1}" for i in range(n_95)])
reduced_df.insert(0, "App", df["App"].values)
reduced_df.insert(1, "Category", df["Category"].values)
reduced_df.insert(2, "Type", df["Type"].values)
reduced_df.to_csv("googleplaystore_pca.csv", index=False)

with open("week7_pca_report.txt", "w", encoding="utf-8") as f:
    f.write("WEEK 7 DIMENSIONALITY REDUCTION REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Correlation threshold: {THRESHOLD}\n")
    f.write(f"Features dropped for multicollinearity: {to_drop}\n")
    f.write(f"Features kept: {len(selected_features)}\n\n")
    f.write("Explained variance ratio per component:\n")
    for i, (e, c) in enumerate(zip(evr, cum_evr), start=1):
        f.write(f"  PC{i}: {e:.4f}   cumulative: {c:.4f}\n")
    f.write(f"\nComponents for 95% variance: {n_95}\n")

print(f"\nSaved: googleplaystore_pca.csv ({n_95} components), week7_pca_report.txt")
