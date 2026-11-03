# FINAL REVIEW: corrected cleaning + Weeks 5-7 (run independently)
# Dependencies: pandas numpy scikit-learn plotly ipywidgets
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output

SOURCE = "https://raw.githubusercontent.com/ran-nope/DSV_PROJECT/main/googleplaystore.csv"
OUT = Path("final_review_outputs")
OUT.mkdir(exist_ok=True)
raw = pd.read_csv(SOURCE)
required = {"App", "Category", "Rating", "Reviews", "Size", "Installs", "Price", "Type"}
assert required.issubset(raw.columns), "Unexpected source schema"
d = raw.drop_duplicates().copy()
exact_duplicates = len(raw) - len(d)
for c in ["Rating", "Reviews"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
for c, pattern in [("Installs", r"[,+]"), ("Price", r"\$")]:
    d[c] = pd.to_numeric(d[c].astype(str).str.replace(pattern, "", regex=True), errors="coerce")
# Quarantine shifted/malformed records; never turn 'Free' into zero installs.
valid = (d["Rating"].isna() | d["Rating"].between(1, 5))
valid &= d[["Reviews", "Installs", "Price"]].notna().all(axis=1)
valid &= d[["Reviews", "Installs", "Price"]].ge(0).all(axis=1)
valid &= d["App"].notna() & d["Category"].notna()
quarantine = d.loc[~valid].copy()
d = d.loc[valid].copy()

def size_mb(value):
    text = str(value).strip()
    if text.endswith("M"):
        return pd.to_numeric(text[:-1], errors="coerce")
    if text.endswith("k"):
        return pd.to_numeric(text[:-1], errors="coerce") / 1024
    return np.nan

d["Size_MB"] = d["Size"].map(size_mb)
d["Rating_missing"] = d["Rating"].isna()
d["Size_missing"] = d["Size_MB"].isna()
# Preserve observed ratings for summaries; impute only the PCA input.
# Missingness patterns alone cannot establish MCAR.
d["Type"] = np.where(d["Price"].eq(0), "Free", "Paid")
d["Last Updated"] = pd.to_datetime(d["Last Updated"], errors="coerce")
# Repeated app names may represent different observations: retain them and audit.
d["Repeated_app_name"] = d.duplicated("App", keep=False)

# WEEK 5: engineered features; installs are bucket lower bounds, not exact sales.
for c in ["Reviews", "Installs", "Price"]:
    d["Log1p_" + c] = np.log1p(d[c])
d["Reviews_per_1000_install_lower_bound"] = np.where(
    d["Installs"].gt(0), 1000 * d["Reviews"] / d["Installs"], np.nan)
d["Main_genre"] = d["Genres"].astype("string").str.split(";").str[0]
features = ["Rating", "Log1p_Reviews", "Size_MB", "Log1p_Installs", "Log1p_Price"]
X = d[features].copy()
medians = X.median()
assert medians.notna().all(), "A PCA feature is entirely missing"
X = X.fillna(medians)
assert np.isfinite(X.to_numpy()).all()
Z = StandardScaler().fit_transform(X)

# WEEK 6: full PCA, variance, weights, scores and reconstruction validation.
pca = PCA(svd_solver="full")
scores = pca.fit_transform(Z)
d["PC1"], d["PC2"] = scores[:, 0], scores[:, 1]
variance = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(len(features))],
                         "Explained_variance": pca.explained_variance_ratio_,
                         "Cumulative_variance": np.cumsum(pca.explained_variance_ratio_)})
weights = pd.DataFrame(pca.components_.T, index=features, columns=variance["Component"])
assert np.allclose(pca.inverse_transform(scores), Z)
assert np.isclose(variance["Explained_variance"].sum(), 1)
assert len(raw) == exact_duplicates + len(quarantine) + len(d)
assert d["Rating"].dropna().between(1, 5).all()
assert d[["Reviews", "Installs", "Price"]].ge(0).all().all()
# Primitive conversion checks guard the original dirty-value cases.
assert np.isclose(size_mb("1024k"), 1) and size_mb("19M") == 19
assert np.isnan(size_mb("Varies with device"))
assert np.isfinite(scores).all() and len(scores) == len(d)

audit = {"raw_rows": len(raw), "exact_duplicates_removed": exact_duplicates,
         "quarantined_rows": len(quarantine), "clean_records": len(d),
         "unique_app_names": int(d["App"].nunique()),
         "missing_ratings_preserved": int(d["Rating_missing"].sum()),
         "missing_sizes_preserved": int(d["Size_missing"].sum()),
         "pca_medians": medians.to_dict(),
         "two_pc_variance_pct": float(100 * pca.explained_variance_ratio_[:2].sum()),
         "source": SOURCE,
         "limitations": "Historical sample; repeated app names retained; install buckets are lower bounds; associations are not causal; no revenue estimate."}
d.to_csv(OUT / "apps_features_pca.csv", index=False)
quarantine.to_csv(OUT / "quarantined_records.csv", index=False)
variance.to_csv(OUT / "pca_variance.csv", index=False)
weights.to_csv(OUT / "pca_component_weights.csv")
(OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("VALIDATION PASSED", json.dumps(audit, indent=2))
display(variance, weights)
px.line(variance, x="Component", y="Cumulative_variance", markers=True,
        title="PCA cumulative explained variance").show()
px.imshow(weights, text_auto=".2f", color_continuous_scale="RdBu_r",
          title="PCA component weights (not causal effects)").show()

# WEEK 7: interactive category/type filters, observed-rating KPI and chart selector.
category = widgets.Dropdown(options=["All"] + sorted(d["Category"].unique()), description="Category:")
app_type = widgets.Dropdown(options=["All", "Free", "Paid"], description="Type:")
min_reviews = widgets.IntSlider(value=0, min=0, max=100000, step=1000, description="Min reviews:")
chart = widgets.Dropdown(options=["PCA", "Rating vs installs", "Category summary"], description="Chart:")
output = widgets.Output()

def draw_dashboard(change=None):
    view = d.copy()
    if category.value != "All":
        view = view.loc[view["Category"].eq(category.value)]
    if app_type.value != "All":
        view = view.loc[view["Type"].eq(app_type.value)]
    view = view.loc[view["Reviews"].ge(min_reviews.value)]
    with output:
        clear_output(wait=True)
        if view.empty:
            print("No records match these filters.")
            return
        print(f"Records: {len(view):,} | Unique app names: {view['App'].nunique():,} | Observed mean rating: {view['Rating'].mean():.2f}")
        print("Historical sample. Install values are lower bounds. Ratings exclude missing values.")
        if chart.value == "PCA":
            fig = px.scatter(view, x="PC1", y="PC2", color="Type", hover_name="App",
                             hover_data=["Category", "Rating"], opacity=0.5,
                             title=f"PCA projection: first two PCs explain {audit['two_pc_variance_pct']:.2f}%")
        elif chart.value == "Rating vs installs":
            fig = px.scatter(view, x="Log1p_Installs", y="Rating", color="Type", hover_name="App",
                             opacity=0.5, title="Observed rating vs log1p(install lower bound)")
        else:
            summary = view.groupby("Category", as_index=False).agg(
                Records=("App", "size"), Observed_mean_rating=("Rating", "mean"),
                Median_install_lower_bound=("Installs", "median"))
            display(summary.sort_values("Records", ascending=False))
            fig = px.bar(summary.sort_values("Records", ascending=False), x="Category", y="Records",
                         title="Record counts by category in the filtered historical sample")
        fig.show()

for control in [category, app_type, min_reviews, chart]:
    control.observe(draw_dashboard, names="value")
display(widgets.VBox([widgets.HBox([category, app_type]), min_reviews, chart, output]))
draw_dashboard()
assert pd.read_csv(OUT / "apps_features_pca.csv").shape[0] == len(d)
assert json.loads((OUT / "audit.json").read_text())["clean_records"] == len(d)
print("EXPORT ROUND-TRIP CHECKS PASSED")
