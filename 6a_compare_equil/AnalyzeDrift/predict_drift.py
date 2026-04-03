"""Predict x0_drift label from bond parameter distribution statistics.

Defines x0_drift (bool) as True when the force field equilibrium bond length
(ff_equilibrium) lies outside a z-score of 3 relative to the training data
distribution:

    x0_drift = |ff_equilibrium - Training_mean| / Training_std > 3

Two analyses are performed, each with 5-fold stratified cross-validation
using Logistic Regression and Random Forest classifiers:

    Analysis A — Training features only (all 799 parameters):
        Uses MD off-equilibrium distribution statistics from the training
        set (snapshots at a range of temperatures): mean, std, IQR, full
        min/max range, whisker range, skewness proxy, upper/lower tail
        widths, tail asymmetry ratio, and sample count.

    Analysis B — Training + Sage 2.3.0 features (~550 parameters with
        Sage 2.3.0 coverage):
        Adds Sage 2.3.0 minimized equilibrium statistics (mean, std, IQR,
        full range, whisker range, skew proxy, tail widths) and derived
        comparison features: MD-vs-minimized mean offset (temperature-
        induced shift), std ratio, IQR ratio, and full-range ratio between
        the two datasets. Parameters with no Sage 2.3.0 coverage are
        excluded. ff_equilibrium is intentionally excluded from all features
        — the goal is to predict drift from data properties alone.

Note: Training_mean and Training_std appear as raw features AND directly
define the target. The classifiers will primarily leverage those two
columns; the remaining features reveal which distributional properties
additionally separate drifted from non-drifted parameters.

Features
--------
All features are derived from bond length distributions across a set of
molecular geometries; ff_equilibrium is never used as a feature.

Training data features (MD snapshots at a range of temperatures):

    training_n_log10
        log10 of the number of training bond instances for this parameter
        type. Low-count parameters are more likely to be poorly constrained.

    training_mean
        Mean bond length across all training geometries. Together with
        training_std, this directly defines the x0_drift target.

    training_std
        Standard deviation of the bond length distribution. Reflects the
        total spread due to both thermal fluctuations and chemical diversity
        within the SMIRKS pattern.

    training_iqr
        Interquartile range (Q3 − Q1). A robust spread measure less
        sensitive to outliers than std.

    training_full_range
        Full observed range (max − min). Captures the extremes of the
        distribution, including outlier geometries.

    training_whisker_range
        Range between the Tukey whisker endpoints (upper whisker − lower
        whisker), i.e., the non-outlier span. Complements full_range by
        separating typical spread from extreme tails.

    training_skew_proxy
        Mean minus median. Positive values indicate a right-skewed
        distribution; negative values indicate left-skew.

    training_upper_tail
        Distance from Q3 to the maximum (max − Q3). Measures how far the
        upper extreme extends beyond the box.

    training_lower_tail
        Distance from the minimum to Q1 (Q1 − min). Measures the downward
        extent of the lower extreme.

    training_tail_ratio
        Ratio of upper tail to lower tail (training_upper_tail /
        training_lower_tail). Values far from 1 indicate asymmetric
        outlier behaviour.

Sage 2.3.0 features (minimized equilibrium geometries, Analysis B only):

    sage_n_log10
        log10 of the number of Sage 2.3.0 minimized geometries for this
        parameter type. Parameters with few minimized examples may be
        poorly represented in the minimized reference.

    sage_mean
        Mean bond length across Sage 2.3.0 minimized geometries. Provides
        a temperature-independent reference for the equilibrium geometry.

    sage_std
        Standard deviation of the minimized bond length distribution.
        Near-zero for parameter types with only one or two minimized
        geometries.

    sage_iqr
        Interquartile range of the minimized distribution.

    sage_full_range
        Full range of the minimized distribution (max − min).

    sage_whisker_range
        Whisker span of the minimized distribution.

    sage_skew_proxy
        Mean minus median of the minimized distribution.

    sage_upper_tail
        Upper tail width of the minimized distribution (max − Q3).

    sage_lower_tail
        Lower tail width of the minimized distribution (Q1 − min).

Derived cross-dataset features (Analysis B only):

    train_mean_minus_sage_mean
        Offset between the MD training mean and the Sage minimized mean.
        Captures the temperature-induced elongation of bonds in MD relative
        to their minimized equilibrium geometry.

    std_ratio_train_over_sage
        Ratio of training std to Sage std. Quantifies how much broader the
        MD distribution is compared to the minimized distribution; large
        values indicate strong thermal broadening.

    iqr_ratio_train_over_sage
        Ratio of training IQR to Sage IQR. A robust version of the
        broadening ratio, less influenced by outlier geometries.

    range_ratio_train_over_sage
        Ratio of full training range to full Sage range. Captures the
        relative extent of extreme conformations in MD vs. minimization.

Input
-----
--csv : str
    Path to bond_statistics.csv. Defaults to the file produced by
    check_equil_values.py in the c_bond_general analysis directory.
--output-dir : str, optional
    Directory for output plots. Defaults to the same directory as the CSV.

Output
------
Prints class balance, CV classification metrics, and feature importances
to stdout. Saves drift_feature_importances_training_only.pdf and
drift_feature_importances_with_sage.pdf in output-dir.

Examples
--------
# Default CSV path (relative to this script's location):
$ python predict_drift.py

# Explicit path:
$ python predict_drift.py \\
    --csv ../c_bond_general/equilibrium_value_analysis/plots/bond_statistics.csv \\
    --output-dir ./outputs
"""

import argparse
import pathlib

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


Z_SCORE_THRESHOLD = 2.0
SAGE_PREFIX = "Sage 2.3.0 Train"


def load_and_label(csv_path: pathlib.Path) -> pl.DataFrame:
    """Load bond_statistics.csv and add the x0_drift boolean column.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to bond_statistics.csv.

    Returns
    -------
    pl.DataFrame
        Original columns plus x0_drift (bool).
    """
    df = pl.read_csv(csv_path)
    df = df.with_columns(
        (
            (pl.col("ff_equilibrium") - pl.col("Training_mean")).abs()
            / pl.col("Training_std")
            > Z_SCORE_THRESHOLD
        ).alias("x0_drift")
    )
    return df


def build_training_features(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Feature matrix using only Training dataset statistics.

    Includes mean, std, IQR, full min/max range, whisker range, skewness
    proxy, upper/lower tail widths, and tail asymmetry ratio. ff_equilibrium
    is excluded — we predict drift from data properties alone.

    Suitable for all rows regardless of Sage 2.3.0 coverage.

    Parameters
    ----------
    df : pl.DataFrame
        Full labeled DataFrame.

    Returns
    -------
    X : np.ndarray, shape (n_params, n_features)
    feature_names : list[str]
    """
    t_min = df["Training_min"].to_numpy()
    t_max = df["Training_max"].to_numpy()
    t_q1 = df["Training_q1"].to_numpy()
    t_q3 = df["Training_q3"].to_numpy()
    eps = 1e-9

    raw = {
        "training_n_log10": np.log10(df["Training_n"].to_numpy()),
        "training_mean": df["Training_mean"].to_numpy(),
        "training_std": df["Training_std"].to_numpy(),
        # Spread measures
        "training_iqr": t_q3 - t_q1,
        "training_full_range": t_max - t_min,
        "training_whisker_range": (
            df["Training_whisker_high"].to_numpy()
            - df["Training_whisker_low"].to_numpy()
        ),
        # Asymmetry / shape
        "training_skew_proxy": (
            df["Training_mean"].to_numpy() - df["Training_median"].to_numpy()
        ),
        "training_upper_tail": t_max - t_q3,
        "training_lower_tail": t_q1 - t_min,
        "training_tail_ratio": ((t_max - t_q3) / (t_q1 - t_min + eps)),
    }
    feature_names = list(raw.keys())
    X = np.column_stack(list(raw.values()))
    return X, feature_names


def build_training_and_sage_features(
    df: pl.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix using Training and Sage 2.3.0 statistics.

    Includes all Training features plus Sage 2.3.0 minimized equilibrium
    statistics (mean, std, IQR, full range, whisker range, skew proxy, tail
    widths) and derived cross-dataset features (mean offset, std/IQR/range
    ratios). ff_equilibrium is intentionally excluded from all features.

    Rows with null Sage 2.3.0 data must be excluded before calling this
    function. Uses a small epsilon to guard against zero std/iqr in Sage
    2.3.0 data (occurs for parameters with only one minimized geometry).

    Parameters
    ----------
    df : pl.DataFrame
        Labeled DataFrame pre-filtered to rows with Sage 2.3.0 coverage.

    Returns
    -------
    X : np.ndarray, shape (n_params, n_features)
    feature_names : list[str]
    """
    sp = SAGE_PREFIX
    eps = 1e-9  # guard against zero sage_std / sage_iqr (single-geometry rows)

    sage_std = df[f"{sp}_std"].fill_null(0.0).to_numpy()
    sage_iqr = df[f"{sp}_iqr"].fill_null(0.0).to_numpy()

    t_min = df["Training_min"].to_numpy()
    t_max = df["Training_max"].to_numpy()
    t_q1 = df["Training_q1"].to_numpy()
    t_q3 = df["Training_q3"].to_numpy()

    s_min = df[f"{sp}_min"].fill_null(0.0).to_numpy()
    s_max = df[f"{sp}_max"].fill_null(0.0).to_numpy()
    s_q1 = df[f"{sp}_q1"].fill_null(0.0).to_numpy()
    s_q3 = df[f"{sp}_q3"].fill_null(0.0).to_numpy()
    s_mean = df[f"{sp}_mean"].fill_null(0.0).to_numpy()

    raw = {
        # Training distribution
        "training_n_log10": np.log10(df["Training_n"].to_numpy()),
        "training_mean": df["Training_mean"].to_numpy(),
        "training_std": df["Training_std"].to_numpy(),
        # Spread measures
        "training_iqr": t_q3 - t_q1,
        "training_full_range": t_max - t_min,
        "training_whisker_range": (
            df["Training_whisker_high"].to_numpy()
            - df["Training_whisker_low"].to_numpy()
        ),
        # Asymmetry / shape
        "training_skew_proxy": (
            df["Training_mean"].to_numpy() - df["Training_median"].to_numpy()
        ),
        "training_upper_tail": t_max - t_q3,
        "training_lower_tail": t_q1 - t_min,
        "training_tail_ratio": ((t_max - t_q3) / (t_q1 - t_min + eps)),
        # Sage 2.3.0 minimized equilibrium statistics
        "sage_n_log10": np.log10(
            df[f"{sp}_n"].fill_null(1).cast(pl.Float64).to_numpy()
        ),
        "sage_mean": s_mean,
        "sage_std": sage_std,
        "sage_iqr": s_q3 - s_q1,
        "sage_full_range": s_max - s_min,
        "sage_whisker_range": (
            df[f"{sp}_whisker_high"].fill_null(0.0).to_numpy()
            - df[f"{sp}_whisker_low"].fill_null(0.0).to_numpy()
        ),
        "sage_skew_proxy": (s_mean - df[f"{sp}_median"].fill_null(0.0).to_numpy()),
        "sage_upper_tail": s_max - s_q3,
        "sage_lower_tail": s_q1 - s_min,
        # Derived: temperature-induced shift (MD mean vs minimized mean)
        # NOTE: ff_equilibrium is intentionally excluded — we are predicting
        # whether a bond type will drift during fitting, not explaining why
        # a particular FF value is wrong.
        "train_mean_minus_sage_mean": (df["Training_mean"].to_numpy() - s_mean),
        # Derived: temperature broadening relative to minimized spread
        "std_ratio_train_over_sage": (df["Training_std"].to_numpy() / (sage_std + eps)),
        "iqr_ratio_train_over_sage": ((t_q3 - t_q1) / (sage_iqr + eps)),
        "range_ratio_train_over_sage": ((t_max - t_min) / (s_max - s_min + eps)),
    }
    feature_names = list(raw.keys())
    X = np.column_stack(list(raw.values()))
    return X, feature_names


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label: str = "",
    n_splits: int = 5,
) -> tuple[np.ndarray, set[str]]:
    """Run stratified cross-validated classification with regularization.

    Step 1 — L1 Logistic Regression with CV-tuned C (nested CV):
        Uses LogisticRegressionCV with L1 penalty and a log-spaced grid of
        20 C values. C is selected by 3-fold inner CV within each outer fold.
        L1 drives irrelevant feature coefficients to exactly zero, producing
        a sparse, interpretable selection. Reports the tuned C per fold,
        features selected in all outer folds, and features selected in any
        outer fold.

    Step 2 — Random Forest with SelectFromModel:
        RF is evaluated by outer CV for performance metrics. A separate RF is
        then fit on the full dataset and SelectFromModel identifies features
        with mean importance >= the mean across all features. This provides
        a complementary non-linear feature subset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary target array (1 = x0_drift, 0 = no drift).
    feature_names : list[str]
        Feature names for reporting.
    label : str
        Short label used in section headers.
    n_splits : int
        Number of stratified outer CV folds.

    Returns
    -------
    rf_importances : np.ndarray
        Mean RF feature importances averaged over outer CV folds.
    selected_features : set[str]
        Union of features selected by L1-LR (any fold) and RF SelectFromModel.
    """
    cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scoring = ["accuracy", "roc_auc", "f1"]
    hdr = f"  {label} — " if label else "  "

    # ------------------------------------------------------------------
    # Step 1: L1 Logistic Regression, C auto-tuned by 3-fold inner CV
    # ------------------------------------------------------------------
    l1_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=np.logspace(-3, 2, 20),
                    penalty="l1",
                    solver="liblinear",
                    cv=cv_inner,
                    class_weight="balanced",
                    max_iter=2000,
                    refit=True,
                ),
            ),
        ]
    )

    print(f"\n{'=' * 64}")
    print(
        f"{hdr}L1 Logistic Regression"
        f"  (C tuned by 3-fold inner CV, {n_splits}-fold outer CV)"
    )
    print(f"{'=' * 64}")
    l1_cv = cross_validate(
        l1_pipe, X, y, cv=cv_outer, scoring=scoring, return_estimator=True
    )
    _print_cv_scores(l1_cv)

    coef_matrix = np.array(
        [est.named_steps["clf"].coef_[0] for est in l1_cv["estimator"]]
    )
    avg_abs_coef = np.abs(coef_matrix).mean(axis=0)
    best_Cs = [float(est.named_steps["clf"].C_[0]) for est in l1_cv["estimator"]]
    print(f"\n  Tuned C per fold: {[f'{c:.4g}' for c in best_Cs]}")
    print(f"  Median C: {np.median(best_Cs):.4g}")

    nonzero_all = np.all(coef_matrix != 0.0, axis=0)
    nonzero_any = np.any(coef_matrix != 0.0, axis=0)

    print(f"\n  Features with non-zero coefficient in ALL {n_splits} folds:")
    selected_l1_all = [n for n, s in zip(feature_names, nonzero_all) if s]
    if selected_l1_all:
        for name in selected_l1_all:
            idx = list(feature_names).index(name)
            print(f"    {name:42s}  mean|coef|={avg_abs_coef[idx]:.4f}")
    else:
        print("    (none — try a larger C range)")

    print("\n  Features with non-zero coefficient in ANY fold (ranked by mean |coef|):")
    selected_l1_any = [
        (n, c)
        for n, s, c in sorted(
            zip(feature_names, nonzero_any, avg_abs_coef), key=lambda x: -x[2]
        )
        if s
    ]
    for name, coef in selected_l1_any:
        print(f"    {name:42s}  mean|coef|={coef:.4f}")

    # ------------------------------------------------------------------
    # Step 2: Random Forest
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print(f"\n{'=' * 64}")
    print(f"{hdr}Random Forest  ({n_splits}-fold stratified CV)")
    print(f"{'=' * 64}")
    rf_cv = cross_validate(
        rf, X, y, cv=cv_outer, scoring=scoring, return_estimator=True
    )
    _print_cv_scores(rf_cv)
    rf_importances = np.array(
        [est.feature_importances_ for est in rf_cv["estimator"]]
    ).mean(axis=0)
    print("\nRF feature importances (mean over folds):")
    _print_ranked(feature_names, rf_importances)

    # SelectFromModel on full data — identifies features above mean importance
    rf_full = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    selector = SelectFromModel(rf_full, threshold="mean")
    selector.fit(X, y)
    selected_rf = {n for n, s in zip(feature_names, selector.get_support()) if s}
    print(
        f"\n  RF SelectFromModel (threshold=mean importance):"
        f" {len(selected_rf)} features selected"
    )
    for name in sorted(
        selected_rf, key=lambda n: -rf_importances[list(feature_names).index(n)]
    ):
        idx = list(feature_names).index(name)
        print(f"    {name:42s}  importance={rf_importances[idx]:.4f}")

    # Union of L1-any and RF-selected features
    selected_l1_names = {name for name, _ in selected_l1_any}
    selected_features = selected_l1_names | selected_rf

    print(
        f"\n  Combined selected features (L1-any ∪ RF-SelectFromModel): {len(selected_features)}"
    )
    for name in sorted(
        selected_features, key=lambda n: -rf_importances[list(feature_names).index(n)]
    ):
        in_l1 = "L1" if name in selected_l1_names else "  "
        in_rf = "RF" if name in selected_rf else "  "
        idx = list(feature_names).index(name)
        print(f"    [{in_l1}+{in_rf}]  {name:42s}  rf_imp={rf_importances[idx]:.4f}")

    return rf_importances, selected_features


def _print_cv_scores(cv_result: dict) -> None:
    for metric in ("accuracy", "roc_auc", "f1"):
        key = f"test_{metric}"
        vals = cv_result[key]
        print(f"  {metric.upper():10s}: {vals.mean():.3f} ± {vals.std():.3f}")


def _print_ranked(names: list[str], values: np.ndarray) -> None:
    for name, val in sorted(zip(names, values), key=lambda x: -x[1]):
        print(f"  {name:42s}  {val:.4f}")


def plot_importances(
    importances: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: pathlib.Path,
    selected_features: set[str] | None = None,
) -> None:
    """Save a horizontal bar chart of Random Forest feature importances.

    Bars for features in selected_features are coloured orange; all others
    are steelblue. A legend indicates selected vs. not selected.

    Parameters
    ----------
    importances : np.ndarray
        Mean feature importances from cross-validation.
    feature_names : list[str]
        Feature names.
    title : str
        Plot title.
    out_path : pathlib.Path
        Output PDF path.
    selected_features : set[str], optional
        Features identified by L1-LR or RF SelectFromModel. Highlighted
        in orange. If None, all bars are steelblue.
    """
    sorted_idx = np.argsort(importances)
    selected_features = selected_features or set()
    colors = [
        "darkorange" if feature_names[i] in selected_features else "steelblue"
        for i in sorted_idx
    ]
    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_names) * 0.5)))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=colors,
        edgecolor="white",
    )
    ax.set_xlabel("Mean feature importance (Random Forest, 5-fold CV)")
    ax.set_title(title)
    if selected_features:
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="darkorange", label="selected (L1 ∪ RF)"),
                Patch(color="steelblue", label="not selected"),
            ],
            loc="lower right",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")


def main() -> None:
    default_csv = (
        pathlib.Path(__file__).parent.parent
        / "c_bond_general"
        / "equilibrium_value_analysis"
        / "plots"
        / "bond_statistics.csv"
    )

    parser = argparse.ArgumentParser(
        description="Predict x0_drift from bond parameter distribution statistics."
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=default_csv,
        help="Path to bond_statistics.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for output plots (default: same directory as CSV)",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    output_dir = (args.output_dir or csv_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and add x0_drift label
    df = load_and_label(csv_path)
    n_total = len(df)
    n_drift = int(df["x0_drift"].sum())
    print(f"Loaded {n_total} bond parameters from {csv_path.name}")
    print(
        f"x0_drift: {n_drift} True ({100 * n_drift / n_total:.1f}%)  |  "
        f"{n_total - n_drift} False ({100 * (n_total - n_drift) / n_total:.1f}%)"
    )

    # ------------------------------------------------------------------ #
    # Analysis A: Training features only (all rows)
    # ------------------------------------------------------------------ #
    print(f"\n{'#' * 64}")
    print(f"# Analysis A: Training features only  (n={n_total})")
    print(f"{'#' * 64}")

    X_train, feat_train = build_training_features(df)
    y_all = df["x0_drift"].cast(pl.Int32).to_numpy()
    assert np.isfinite(X_train).all(), "NaN/Inf in Training feature matrix"

    imp_train, sel_train = run_cv(X_train, y_all, feat_train, label="Training-only")
    plot_importances(
        imp_train,
        feat_train,
        title="Predictors of x0_drift — Training features only\n"
        r"$|$ff_equilibrium $-$ Training_mean$|$ / Training_std $>$ 2",
        out_path=output_dir / "drift_feature_importances_training_only.pdf",
        selected_features=sel_train,
    )

    # ------------------------------------------------------------------ #
    # Analysis B: Training + Sage 2.3.0 features (rows with Sage coverage)
    # ------------------------------------------------------------------ #
    df_sage = df.filter(pl.col(f"{SAGE_PREFIX}_n").is_not_null())
    n_sage = len(df_sage)
    n_drift_sage = int(df_sage["x0_drift"].sum())
    print(f"\n{'#' * 64}")
    print(f"# Analysis B: Training + Sage 2.3.0 features  (n={n_sage})")
    print(f"#   ({n_total - n_sage} parameters excluded — no Sage 2.3.0 coverage)")
    print(f"#   x0_drift: {n_drift_sage} True ({100 * n_drift_sage / n_sage:.1f}%)")
    print(f"{'#' * 64}")

    X_sage, feat_sage = build_training_and_sage_features(df_sage)
    y_sage = df_sage["x0_drift"].cast(pl.Int32).to_numpy()
    assert np.isfinite(X_sage).all(), "NaN/Inf in Training+Sage feature matrix"

    imp_sage, sel_sage = run_cv(X_sage, y_sage, feat_sage, label="Training+Sage")
    plot_importances(
        imp_sage,
        feat_sage,
        title="Predictors of x0_drift — Training + Sage 2.3.0 features\n"
        r"$|$ff_equilibrium $-$ Training_mean$|$ / Training_std $>$ 2",
        out_path=output_dir / "drift_feature_importances_with_sage.pdf",
        selected_features=sel_sage,
    )


if __name__ == "__main__":
    main()
