"""ablation.py

Feature-group ablation for the Jagawana AI fire-risk model.

Questions answered:
  1. How much do fire-history features and weather features each contribute?
  2. How well can weather, season and location alone (no fire history)
     separate dangerous from safe conditions?
  3. How does each model do in "quiet" cells that have not burned recently
     (no hotspot in the last 30 or 90 days)?
  4. Do dryness anomalies (this period's rain, humidity and temperature
     compared with the normal for the same cell and month) help?

The "normal" values (climatology) are computed from TRAINING years only.
Same date split and model settings as train_models.py, which must be in the
same folder (src) and unchanged.

Run from the project folder:
    python src\\ablation.py
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from train_models import (
    DATA_FILE, FEATURES, FIG_DIR, MODEL_DIR, RESULTS_DIR,
    compute_metrics, make_xgb, pick_threshold, split_data,
)

SEED = 42
FIRE_HISTORY = ["fires_7d", "fires_30d", "fires_90d", "days_since_fire",
                "nb_fires_7d", "nb_fires_30d"]
WEATHER = ["rain_7d", "rain_14d", "rain_30d", "rain_60d", "dry_streak",
           "tmax_7d", "rh_7d", "wind_7d"]
SEASON = ["month_sin", "month_cos"]
LOCATION = ["lat", "lon"]
ANOMALY_SOURCES = ["rain_30d", "rain_60d", "rh_7d", "tmax_7d"]
ANOMALY = [f"{c}_anom" for c in ANOMALY_SOURCES]

# (key, short label, features)
VARIANTS = [
    ("A", "Full model", FEATURES),
    ("B", "Fire history only", FIRE_HISTORY),
    ("C", "No fire history (weather+season+location)", WEATHER + SEASON + LOCATION),
    ("D", "Weather + season only", WEATHER + SEASON),
    ("E", "No fire history + dryness anomalies", WEATHER + SEASON + LOCATION + ANOMALY),
    ("F", "Full model + dryness anomalies", FEATURES + ANOMALY),
]


def add_dryness_anomalies(df, train):
    """Anomaly = value minus the training-years average for the same cell and month."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    reference = train.assign(month=train["date"].dt.month)
    climatology = (
        reference.groupby(["cell_id", "month"])[ANOMALY_SOURCES]
        .mean().add_suffix("_clim").reset_index()
    )
    df = df.merge(climatology, on=["cell_id", "month"], how="left")
    for col in ANOMALY_SOURCES:
        df[f"{col}_anom"] = df[col] - df[f"{col}_clim"]
    return df, climatology


def top_precision(y, score, fraction=0.10):
    """Share of real fire rows among the riskiest `fraction` of rows."""
    k = max(int(len(y) * fraction), 1)
    order = np.argsort(-score, kind="stable")[:k]
    return float(y[order].mean())


def evaluate(variant, subset, y, score, threshold=None):
    positive_rate = float(y.mean())
    row = {
        "variant": variant,
        "subset": subset,
        "n_rows": int(len(y)),
        "positive_rate": positive_rate,
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan,
    }
    row["top10_precision"] = top_precision(y, score, 0.10)
    row["top10_lift"] = row["top10_precision"] / positive_rate if positive_rate > 0 else np.nan
    if threshold is not None:
        metrics = compute_metrics(y, score, threshold)
        row.update(precision=metrics["precision"], recall=metrics["recall"], f1=metrics["f1"])
    return row


def plot_ablation(results, path):
    labels = {key: f"{key}. {label}" for key, label, _ in VARIANTS}
    panels = [("all test rows", "All test rows"),
              ("no fire in last 30 days", "Quiet areas (no fire in last 30 days)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (subset, title) in zip(axes, panels):
        part = results[(results["subset"] == subset) & results["variant"].isin(labels.values())]
        part = part.set_index("variant").loc[[labels[k] for k, _, _ in VARIANTS]]
        ax.barh(part.index[::-1], part["pr_auc"].values[::-1], color="#2a6f97")
        ax.axvline(part["positive_rate"].iloc[0], ls="--", color="gray", label="Random guess")
        ax.set_title(title)
        ax.set_xlabel("PR-AUC (test 2024-2025)")
        ax.set_xlim(0, 1)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    for folder in (RESULTS_DIR, FIG_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    train_only, _, _ = split_data(df)
    df, climatology = add_dryness_anomalies(df, train_only)
    train, valid, test = split_data(df)
    y_tr, y_va, y_te = train["label"].values, valid["label"].values, test["label"].values

    subsets = {
        "all test rows": np.ones(len(test), dtype=bool),
        "no fire in last 30 days": (test["fires_30d"] == 0).values,
        "no fire in last 90 days": (test["fires_90d"] == 0).values,
    }
    print("Test subsets:")
    for name, mask in subsets.items():
        print(f"  {name:26s} {int(mask.sum()):>8,} rows, positive rate {y_te[mask].mean():.3f}")

    print("\nVariants:")
    for key, label, feats in VARIANTS:
        print(f"  {key}. {label} ({len(feats)} features)")

    rows = []
    # Persistence baseline: fire in the last 7 days
    pers_va = valid["fires_7d"].values.astype(float)
    pers_te = test["fires_7d"].values.astype(float)
    pers_thr = pick_threshold(y_va, pers_va)
    rows.append(evaluate("Persistence baseline", "all test rows", y_te, pers_te, pers_thr))

    models, thresholds = {}, {}
    for key, label, feats in VARIANTS:
        name = f"{key}. {label}"
        print(f"\nTraining {name}...")
        model = make_xgb()
        model.fit(train[feats], y_tr, eval_set=[(valid[feats], y_va)], verbose=False)
        threshold = pick_threshold(y_va, model.predict_proba(valid[feats])[:, 1])
        score = model.predict_proba(test[feats])[:, 1]
        for subset, mask in subsets.items():
            rows.append(evaluate(name, subset, y_te[mask], score[mask],
                                 threshold if subset == "all test rows" else None))
        models[key], thresholds[key] = model, threshold

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "ablation_metrics.csv", index=False)

    columns = ["variant", "n_rows", "positive_rate", "pr_auc", "roc_auc",
               "top10_precision", "top10_lift"]
    for subset in subsets:
        print(f"\n=== TEST: {subset.upper()} ===")
        part = results[results["subset"] == subset][columns]
        print(part.round(3).to_string(index=False))

    plot_ablation(results, FIG_DIR / "ablation_pr_auc.png")

    # What drives risk when fire history is hidden? (variant E)
    feats_e = dict((k, f) for k, _, f in VARIANTS)["E"]
    sample = test[feats_e].sample(n=min(5000, len(test)), random_state=SEED)
    contrib = models["E"].get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)
    importance = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=feats_e)
    importance = importance.sort_values(ascending=False)
    importance.to_csv(RESULTS_DIR / "shap_importance_conditions_only.csv",
                      header=["mean_abs_shap"])
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.sort_values().plot.barh(ax=ax, color="#2a6f97")
    ax.set_xlabel("Mean |SHAP value| (impact on fire-risk log-odds)")
    ax.set_title("Drivers of risk without fire history (variant E)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_conditions_only.png", dpi=150)
    plt.close(fig)
    print("\n=== DRIVERS WITHOUT FIRE HISTORY (variant E, mean |SHAP|) ===")
    print(importance.head(10).round(3).to_string())

    # Save the conditions-only model for the demo
    joblib.dump(models["E"], MODEL_DIR / "xgboost_conditions_only.joblib")
    climatology.to_csv(MODEL_DIR / "climatology.csv", index=False)
    (MODEL_DIR / "model_info_conditions_only.json").write_text(json.dumps(
        {"features": feats_e, "threshold": thresholds["E"],
         "anomaly_sources": ANOMALY_SOURCES}, indent=2))

    print("\nDone. Created: results/ablation_metrics.csv, "
          "results/shap_importance_conditions_only.csv, "
          "results/figures/ablation_pr_auc.png, results/figures/shap_conditions_only.png, "
          "models/xgboost_conditions_only.joblib, models/climatology.csv")


if __name__ == "__main__":
    main()