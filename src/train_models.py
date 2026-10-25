"""train_models.py

Trains and evaluates the Jagawana AI fire-risk models.

Task: for a 0.5-degree grid cell on day t, predict whether at least one
hotspot appears in the next 7 days (label = 1).

Split by date (a 7-day gap is removed before each boundary so that no label
window overlaps the next period):
    train      2019-04 ... 2022-12-24
    validation 2023-01-01 ... 2023-12-24
    test       2024-01-01 ... end of data

Models: persistence baseline, Logistic Regression, Random Forest, XGBoost
(XGBoost is the headline model, chosen in advance).
Decision thresholds are tuned on the validation set only.

Extra analyses: top-k operational table, per-year results, SHAP explanations,
and a spatial holdout (train on one island, test on the other).

Run from the project folder:
    python src\\train_models.py
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "dataset.parquet"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"
PRED_FILE = ROOT / "data" / "processed" / "predictions_valid_test.parquet"

SEED = 42
HORIZON = 7
FEATURES = [
    "fires_7d", "fires_30d", "fires_90d", "days_since_fire",
    "nb_fires_7d", "nb_fires_30d",
    "rain_7d", "rain_14d", "rain_30d", "rain_60d", "dry_streak",
    "tmax_7d", "rh_7d", "wind_7d",
    "month_sin", "month_cos", "lat", "lon",
]
PERSIST = "Persistence baseline"
LOGREG = "Logistic Regression"
FOREST = "Random Forest"
XGBOOST = "XGBoost"


# ------------------------------------------------------------------ helpers
def split_data(df):
    """Date-based split with a 7-day purge gap before each boundary."""
    gap = pd.Timedelta(days=HORIZON)
    train = df[df["date"] <= pd.Timestamp("2022-12-31") - gap]
    valid = df[(df["date"] >= "2023-01-01") & (df["date"] <= pd.Timestamp("2023-12-31") - gap)]
    test = df[df["date"] >= "2024-01-01"]
    return train, valid, test


def pick_threshold(y, score):
    """Threshold that maximises F1 (used on validation data only)."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def compute_metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "threshold": threshold,
        "positive_rate": float(np.mean(y)),
    }


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
        tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50,
        n_jobs=-1, random_state=SEED,
    )


def topk_table(y, scores, fractions=(0.05, 0.10, 0.20), seed=SEED):
    """If responders check only the top share of riskiest cell-days, what do they catch?"""
    rng = np.random.default_rng(seed)
    rows = []
    n = len(y)
    for name, score in scores.items():
        s = score + rng.random(n) * 1e-3 if name == PERSIST else score  # random tie-breaks
        order = np.argsort(-s, kind="stable")
        y_sorted = y[order]
        for frac in fractions:
            k = int(n * frac)
            caught = y_sorted[:k].sum()
            rows.append({
                "model": name,
                "top_share_checked": frac,
                "precision_in_top": caught / k,
                "share_of_fire_cell_days_captured": caught / y.sum(),
                "lift_vs_random": (caught / k) / y.mean(),
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- plots
def plot_curves(y, scores, kind, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, score in scores.items():
        if kind == "pr":
            p, r, _ = precision_recall_curve(y, score)
            ax.plot(r, p, label=f"{name} (AP = {average_precision_score(y, score):.3f})")
        else:
            fpr, tpr, _ = roc_curve(y, score)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, score):.3f})")
    if kind == "pr":
        ax.axhline(np.mean(y), ls="--", color="gray", label=f"Random (rate = {np.mean(y):.2f})")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    else:
        ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Random")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title="ROC curves (test set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y, score, threshold, path):
    pred = (score >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["No fire", "Fire"], cmap="Blues",
        values_format="d", ax=ax)
    ax.set_title(f"XGBoost confusion matrix (test, threshold {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results, path):
    test = results[results["split"] == "test"].set_index("model")[["pr_auc", "roc_auc"]]
    ax = test.plot.bar(figsize=(7, 4.5), rot=15)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the test set (2024-2025)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["PR-AUC", "ROC-AUC"])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")


def shap_outputs(model, x_test):
    """SHAP values from XGBoost's built-in TreeSHAP (log-odds units)."""
    sample = x_test.sample(n=min(5000, len(x_test)), random_state=SEED)
    contrib = model.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)
    values = contrib[:, :-1]  # last column is the bias term
    importance = pd.Series(np.abs(values).mean(axis=0), index=sample.columns)
    importance = importance.sort_values(ascending=False)
    importance.to_csv(RESULTS_DIR / "shap_importance.csv", header=["mean_abs_shap"])

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.sort_values().plot.barh(ax=ax, color="#2a6f97")
    ax.set_xlabel("Mean |SHAP value| (impact on fire-risk log-odds)")
    ax.set_title("What drives the XGBoost fire-risk predictions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)

    try:
        import shap
        shap.summary_plot(values, sample, show=False, max_display=18)
        plt.gcf().savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    except Exception as error:  # the bar chart above still exists
        print("  (SHAP beeswarm plot skipped:", repr(error)[:150], ")")
    finally:
        plt.close("all")
    return importance


# ---------------------------------------------------------- spatial holdout
def spatial_holdout(train, valid, test):
    """Train on one island, test on each island (latitude/longitude not used)."""
    feats = [f for f in FEATURES if f not in ("lat", "lon")]
    rows = []
    for source in ("Sumatra", "Kalimantan"):
        tr = train[train["island"] == source]
        va = valid[valid["island"] == source]
        model = make_xgb()
        model.fit(tr[feats], tr["label"], eval_set=[(va[feats], va["label"])], verbose=False)
        threshold = pick_threshold(va["label"].values, model.predict_proba(va[feats])[:, 1])
        for target in ("Sumatra", "Kalimantan"):
            te = test[test["island"] == target]
            score = model.predict_proba(te[feats])[:, 1]
            row = compute_metrics(te["label"].values, score, threshold)
            row.update(trained_on=source, tested_on=target,
                       kind="same island" if source == target else "other island")
            rows.append(row)
            print(f"  trained on {source:10s} -> tested on {target:10s}: "
                  f"PR-AUC {row['pr_auc']:.3f}, ROC-AUC {row['roc_auc']:.3f}")
    cols = ["trained_on", "tested_on", "kind", "pr_auc", "roc_auc",
            "precision", "recall", "f1", "threshold", "positive_rate"]
    return pd.DataFrame(rows)[cols]


# ----------------------------------------------------------------------- main
def main():
    started = time.time()
    for folder in (RESULTS_DIR, FIG_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    missing = [f for f in FEATURES + ["label", "date", "cell_id"] if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in dataset: {missing}")
    df["island"] = np.where(df["lon"] < 107.0, "Sumatra", "Kalimantan")

    train, valid, test = split_data(df)
    print("Split sizes and share of 'fire in next 7 days' rows:")
    for name, part in (("train", train), ("validation", valid), ("test", test)):
        print(f"  {name:10s} {len(part):>9,} rows  "
              f"{part['date'].min().date()} to {part['date'].max().date()}  "
              f"positive rate {part['label'].mean():.3f}")
    print("Cells per island:", df.groupby("island")["cell_id"].nunique().to_dict())

    x_tr, y_tr = train[FEATURES], train["label"].values
    x_va, y_va = valid[FEATURES], valid["label"].values
    x_te, y_te = test[FEATURES], test["label"].values

    scores_va = {PERSIST: x_va["fires_7d"].values.astype(float)}
    scores_te = {PERSIST: x_te["fires_7d"].values.astype(float)}

    print("\nTraining Logistic Regression...")
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(x_tr, y_tr)
    scores_va[LOGREG] = logreg.predict_proba(x_va)[:, 1]
    scores_te[LOGREG] = logreg.predict_proba(x_te)[:, 1]

    print("Training Random Forest (a few minutes)...")
    forest = RandomForestClassifier(
        n_estimators=150, max_depth=14, min_samples_leaf=50, max_samples=0.5,
        n_jobs=-1, random_state=SEED)
    forest.fit(x_tr, y_tr)
    scores_va[FOREST] = forest.predict_proba(x_va)[:, 1]
    scores_te[FOREST] = forest.predict_proba(x_te)[:, 1]

    print("Training XGBoost (a few minutes)...")
    booster = make_xgb()
    booster.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    scores_va[XGBOOST] = booster.predict_proba(x_va)[:, 1]
    scores_te[XGBOOST] = booster.predict_proba(x_te)[:, 1]
    print(f"  XGBoost used {booster.best_iteration + 1} trees (early stopping on validation)")

    # --- metrics for every model, thresholds tuned on validation only
    thresholds = {name: pick_threshold(y_va, scores_va[name]) for name in scores_va}
    rows = []
    for name in scores_va:
        rows.append({"model": name, "split": "validation",
                     **compute_metrics(y_va, scores_va[name], thresholds[name])})
        rows.append({"model": name, "split": "test",
                     **compute_metrics(y_te, scores_te[name], thresholds[name])})
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n=== TEST SET RESULTS (thresholds tuned on validation) ===")
    show = results[results["split"] == "test"].drop(columns="split").set_index("model")
    print(show.round(3).to_string())

    # --- plots
    plot_curves(y_te, scores_te, "pr", FIG_DIR / "pr_curves.png")
    plot_curves(y_te, scores_te, "roc", FIG_DIR / "roc_curves.png")
    plot_confusion(y_te, scores_te[XGBOOST], thresholds[XGBOOST], FIG_DIR / "confusion_matrix.png")
    plot_comparison(results, FIG_DIR / "model_comparison.png")

    # --- operational view: what if responders only check the riskiest cell-days?
    topk = topk_table(y_te, scores_te)
    topk.to_csv(RESULTS_DIR / "topk_operational.csv", index=False)
    print("\n=== IF RESPONDERS CHECK ONLY THE RISKIEST CELL-DAYS (test set) ===")
    print(topk.round(3).to_string(index=False))

    # --- per-year test results for XGBoost
    year_rows = []
    for year in sorted(test["date"].dt.year.unique()):
        mask = (test["date"].dt.year == year).values
        row = compute_metrics(y_te[mask], scores_te[XGBOOST][mask], thresholds[XGBOOST])
        row["year"] = int(year)
        year_rows.append(row)
    by_year = pd.DataFrame(year_rows).set_index("year")
    by_year.to_csv(RESULTS_DIR / "test_by_year.csv")
    print("\n=== XGBOOST, TEST RESULTS BY YEAR ===")
    print(by_year.round(3).to_string())

    # --- explanations
    print("\nComputing SHAP explanations...")
    importance = shap_outputs(booster, x_te)
    print("Top drivers (mean |SHAP|):")
    print(importance.head(8).round(3).to_string())

    # --- spatial holdout
    print("\n=== SPATIAL HOLDOUT (XGBoost, no lat/lon, test years only) ===")
    spatial = spatial_holdout(train, valid, test)
    spatial.to_csv(RESULTS_DIR / "spatial_holdout.csv", index=False)

    # --- save model, settings, and predictions for the demo
    joblib.dump(booster, MODEL_DIR / "xgboost_model.joblib")
    info = {"features": FEATURES, "threshold": thresholds[XGBOOST], "horizon_days": HORIZON,
            "train_end": "2022-12-24", "validation": "2023", "test_start": "2024-01-01"}
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    preds = pd.concat([
        valid.assign(split="validation", risk=scores_va[XGBOOST]),
        test.assign(split="test", risk=scores_te[XGBOOST]),
    ])[["date", "cell_id", "lat", "lon", "island", "split", "label", "risk"]]
    preds.to_parquet(PRED_FILE, index=False)

    print(f"\nDone in {(time.time() - started) / 60:.1f} minutes.")
    print("Created: results/metrics.csv, topk_operational.csv, test_by_year.csv, "
          "shap_importance.csv, spatial_holdout.csv")
    print("         results/figures/*.png, models/xgboost_model.joblib, "
          "data/processed/predictions_valid_test.parquet")


if __name__ == "__main__":
    main()