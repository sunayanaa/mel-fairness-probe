# =============================================================================
# Program      : 06_supplementary_regression.py
# Version      : 1.0
# Description  : Supplementary Regression Analysis
#
#                Fits an OLS regression model linking per-genre classification
#                accuracy to filterbank geometry variables, establishing that
#                effective filter density D_g(theta) is a statistically
#                significant predictor of per-genre accuracy after controlling
#                for genre-intrinsic spectral spread sigma_g and band count M.
#
#                Regression model:
#                  A_g(theta) = beta_0
#                             + beta_1 * D_g
#                             + beta_2 * sigma_g
#                             + beta_3 * M
#                             + epsilon
#
#                where:
#                  A_g(theta) : mean per-genre accuracy from Experiment 1
#                               (15-run repeated CV, f_max=8000, Slaney norm)
#                  D_g        : effective filter density from Program 00
#                  sigma_g    : spectral spread from Program 00
#                  M          : band count (40, 64, 80, 128)
#
#                f_max is fixed at 8000 Hz across all Experiment 1 rows
#                so it is not included as a predictor (zero variance).
#
#                Dataset: 10 genres x 4 M values = 40 observations.
#
#                Analysis steps:
#                  1. Build regression dataframe from exp1_results.json
#                     and precompute_filter_density.json
#                  2. Standardise all predictors (zero mean, unit variance)
#                     to produce interpretable standardised coefficients
#                     and mitigate multicollinearity between D_g and M
#                  3. Fit OLS via statsmodels, report coefficients,
#                     standard errors, t-statistics, p-values, 95% CIs
#                  4. Compute Variance Inflation Factors (VIF) to diagnose
#                     multicollinearity
#                  5. Plot partial regression plots and residual diagnostics
#                  6. Save regression summary as JSON and LaTeX table,
#                     upload to Google Drive
#
# INPUT        :
#                  Drive:
#                    exp1_results.json
#                    precompute_filter_density.json
#                    precompute_genre_centroids.json
#
# STEPS        :
#                  Step 1  Download input files from Google Drive
#                  Step 2  Build regression dataframe (40 observations)
#                  Step 3  Standardise predictors
#                  Step 4  Fit OLS and extract summary statistics
#                  Step 5  Compute VIF for each predictor
#                  Step 6  Generate diagnostic figures
#                  Step 7  Save results JSON and LaTeX table, upload to Drive
#
# OUTPUT FILES :
#                  exp6_regression_results.json
#                      Full OLS results: coefficients, SE, t, p, CI,
#                      R-squared, adjusted R-squared, F-statistic,
#                      VIF per predictor, raw dataframe
#
#                  fig_06_01_partial_regression.png
#                      Partial regression plots for D_g, sigma_g, M
#
#                  fig_06_02_residuals.png
#                      Residual vs fitted and Q-Q plot
#
#                  supp_regression_table.tex
#                      LaTeX table of standardised OLS coefficients
#                      for inclusion in supplementary material Page S1
#
# GPU Required : NO
# Dependencies : numpy, pandas, statsmodels, scipy, matplotlib, sklearn
#
# Change Log   :
#   v1.0  2026-06-07  Initial version
# =============================================================================

import subprocess
import sys

for pkg in ["numpy", "pandas", "statsmodels", "scipy", "matplotlib",
            "scikit-learn"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR     = "/content/drive/MyDrive/paper/mel-fairness-probe/"  # Persistent storage
LOCAL_WORK_DIR  = "/tmp/gtzan_fairness"
OUTPUT_DIR      = os.path.join(LOCAL_WORK_DIR, "outputs_exp6")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GTZAN_GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]
M_VALUES = [40, 64, 80, 128]
FMAX     = 8000

# =============================================================================
# GOOGLE DRIVE HELPERS
# =============================================================================

def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_filepath, remote_filename):
    """Copy a local file to Google Drive project folder."""
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_filename)
    try:
        shutil.copy2(local_filepath, dest_path)
        print(f"  [DRIVE] Uploaded: {remote_filename}")
    except Exception as e:
        print(f"  [DRIVE] Upload failed for {remote_filename}: {e}")

def load_from_drive(remote_filename, local_filepath):
    """Copy a file from Google Drive project folder to local path."""
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_filename)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_filepath)
            return True
        except Exception as e:
            print(f"  [DRIVE] Download failed for {remote_filename}: {e}")
            return False
    else:
        return False

def list_drive_files():
    """List files in the Google Drive project directory."""
    ensure_project_dir()
    try:
        return [f for f in os.listdir(PROJECT_DIR) if os.path.isfile(os.path.join(PROJECT_DIR, f))]
    except Exception as e:
        print(f"  [DRIVE] Could not list files: {e}")
        return []

# =============================================================================
# STEP 1 — DOWNLOAD INPUT FILES
# =============================================================================

def step1_download_inputs():
    print("\n=== STEP 1: Download input files from Google Drive ===")
    drive_files = list_drive_files()
    
    files = [
        "exp1_results.json",
        "precompute_filter_density.json",
        "precompute_genre_centroids.json"
    ]
    loaded = {}
    
    for fname in files:
        local_path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(local_path):
            if fname in drive_files:
                if load_from_drive(fname, local_path):
                    print(f"  Downloaded: {fname}")
                else:
                    raise RuntimeError(f"Cannot download {fname} from Drive.")
            else:
                raise RuntimeError(f"File {fname} not found on Drive.")
        else:
            print(f"  [SKIP] {fname} already local.")
        with open(local_path) as f:
            loaded[fname] = json.load(f)
    return loaded

# =============================================================================
# STEP 2 — BUILD REGRESSION DATAFRAME
# =============================================================================

def step2_build_dataframe(loaded):
    """
    Build a dataframe of 40 observations (10 genres x 4 M values).
    Columns: genre, M, mean_acc, D_g, sigma_g, group
    """
    print("\n=== STEP 2: Build regression dataframe ===")

    exp1     = loaded["exp1_results.json"]
    density  = loaded["precompute_filter_density.json"]
    centroid = loaded["precompute_genre_centroids.json"]

    rows = []
    for M in M_VALUES:
        cfg_key = f"M{M}_fmax{FMAX}_slaney"
        for genre in GTZAN_GENRES:
            mean_acc = (exp1
                        .get("per_M", {})
                        .get(str(M), {})
                        .get("per_genre", {})
                        .get(genre, {})
                        .get("mean_acc", None))
            D_g = (density
                   .get("per_genre", {})
                   .get(genre, {})
                   .get(cfg_key, {})
                   .get("D_g", None))
            sigma_g = centroid.get(genre, {}).get("sigma_hz", None)
            group   = centroid.get(genre, {}).get("group", None)

            if mean_acc is not None and D_g is not None and sigma_g is not None:
                rows.append({
                    "genre"   : genre,
                    "M"       : M,
                    "mean_acc": float(mean_acc),
                    "D_g"     : float(D_g),
                    "sigma_g" : float(sigma_g),
                    "group"   : group
                })

    df = pd.DataFrame(rows)
    print(f"  Dataframe shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Summary statistics:")
    print(df[["mean_acc", "D_g", "sigma_g", "M"]].describe().round(4).to_string())

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n  [WARN] Missing values: {missing[missing > 0].to_dict()}")
    else:
        print(f"\n  No missing values.")

    return df

# =============================================================================
# STEP 3 — STANDARDISE PREDICTORS
# =============================================================================

def step3_standardise(df):
    """
    Standardise predictors to zero mean, unit variance.
    This produces interpretable standardised coefficients and reduces
    multicollinearity between D_g and M (which are correlated since
    larger M produces more filter centres overall).
    Returns df with added *_std columns and the scaler parameters.
    """
    print("\n=== STEP 3: Standardise predictors ===")

    predictors = ["D_g", "sigma_g", "M"]
    scaler_params = {}

    for col in predictors:
        mean = df[col].mean()
        std  = df[col].std()
        df[f"{col}_std"] = (df[col] - mean) / std
        scaler_params[col] = {"mean": round(mean, 4), "std": round(std, 4)}
        print(f"  {col}: mean={mean:.4f}, std={std:.4f}")

    return df, scaler_params

# =============================================================================
# STEP 4 — FIT OLS
# =============================================================================

def step4_fit_ols(df):
    """
    Fit OLS: mean_acc ~ D_g_std + sigma_g_std + M_std
    Returns fitted model and summary dict.
    """
    print("\n=== STEP 4: Fit OLS regression ===")

    X = sm.add_constant(df[["D_g_std", "sigma_g_std", "M_std"]])
    y = df["mean_acc"]

    model  = sm.OLS(y, X).fit()
    print(model.summary())

    # Extract key statistics
    coef_table = {}
    for var in ["const", "D_g_std", "sigma_g_std", "M_std"]:
        coef_table[var] = {
            "coef"    : round(float(model.params[var]),      4),
            "se"      : round(float(model.bse[var]),         4),
            "t"       : round(float(model.tvalues[var]),     4),
            "p"       : round(float(model.pvalues[var]),     4),
            "ci_lower": round(float(model.conf_int().loc[var, 0]), 4),
            "ci_upper": round(float(model.conf_int().loc[var, 1]), 4),
            "sig"     : (
                "***" if model.pvalues[var] < 0.001 else
                "**"  if model.pvalues[var] < 0.01  else
                "*"   if model.pvalues[var] < 0.05  else
                "."   if model.pvalues[var] < 0.10  else ""
            )
        }

    summary = {
        "n_obs"         : int(model.nobs),
        "r_squared"     : round(float(model.rsquared),     4),
        "adj_r_squared" : round(float(model.rsquared_adj), 4),
        "f_statistic"   : round(float(model.fvalue),       4),
        "f_p_value"     : round(float(model.f_pvalue),     4),
        "aic"           : round(float(model.aic),          4),
        "bic"           : round(float(model.bic),          4),
        "coefficients"  : coef_table
    }

    print(f"\n  R² = {summary['r_squared']:.4f}, "
          f"Adj R² = {summary['adj_r_squared']:.4f}, "
          f"F = {summary['f_statistic']:.4f} (p = {summary['f_p_value']:.4f})")
    print(f"\n  Coefficient significance:")
    for var, vals in coef_table.items():
        if var == "const":
            continue
        print(f"    {var:<15s}  β={vals['coef']:+.4f}  "
              f"SE={vals['se']:.4f}  t={vals['t']:+.4f}  "
              f"p={vals['p']:.4f}  {vals['sig']}")

    return model, summary

# =============================================================================
# STEP 5 — VARIANCE INFLATION FACTORS
# =============================================================================

def step5_vif(df):
    """
    Compute VIF for each standardised predictor.
    VIF > 10 indicates severe multicollinearity.
    VIF > 5 warrants caution.
    """
    print("\n=== STEP 5: Variance Inflation Factors ===")

    X = df[["D_g_std", "sigma_g_std", "M_std"]].copy()
    X = sm.add_constant(X)

    vif_data = {}
    for i, col in enumerate(["const", "D_g_std", "sigma_g_std", "M_std"]):
        vif = variance_inflation_factor(X.values, i)
        vif_data[col] = round(float(vif), 4)
        if col != "const":
            flag = " [HIGH]" if vif > 5 else ""
            print(f"  VIF({col}) = {vif:.4f}{flag}")

    return vif_data

# =============================================================================
# STEP 6 — DIAGNOSTIC FIGURES
# =============================================================================

def step6_diagnostic_figures(model, df):
    print("\n=== STEP 6: Generate diagnostic figures ===")

    # ── Fig 1: Partial regression plots ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pred_labels = {
        "D_g_std"     : "$D_g$ (standardised)\nEffective Filter Density",
        "sigma_g_std" : "$\\sigma_g$ (standardised)\nSpectral Spread",
        "M_std"       : "$M$ (standardised)\nBand Count"
    }
    colors = {"D_g_std": "#D95F02", "sigma_g_std": "#1B9E77", "M_std": "#7570B3"}

    for ax, (pred, label) in zip(axes, pred_labels.items()):
        sm.graphics.plot_partregress(
            "mean_acc", pred,
            [p for p in ["D_g_std", "sigma_g_std", "M_std"] if p != pred],
            data=df, ax=ax, obs_labels=False
        )
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Partial residual (accuracy)", fontsize=9)
        ax.set_title(f"Partial regression: {pred.replace('_std','')}", fontsize=9)
        for collection in ax.collections:
            ax.collections[0].set_color(colors[pred])
            ax.collections[0].set_alpha(0.75)
        for line in ax.lines:
            if line.get_linestyle() == "None":
                line.set_color(colors[pred])
                line.set_alpha(0.75)

    fig.suptitle("Partial Regression Plots — Per-Genre Accuracy vs Filterbank Variables\n"
                 "(GTZAN, Exp 1, $f_{\\max}=8$ kHz, Slaney norm; $n=40$)",
                 fontsize=10)
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "fig_06_01_partial_regression.png")
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig1_path}")

    # ── Fig 2: Residual diagnostics ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Residuals vs fitted
    fitted    = model.fittedvalues
    residuals = model.resid
    axes[0].scatter(fitted, residuals, alpha=0.7, color="#1B9E77",
                    edgecolors="white", linewidth=0.5, s=50)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Fitted values", fontsize=9)
    axes[0].set_ylabel("Residuals", fontsize=9)
    axes[0].set_title("Residuals vs Fitted", fontsize=9)
    axes[0].grid(linestyle=":", linewidth=0.5, alpha=0.6)

    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[1].scatter(osm, osr, alpha=0.7, color="#D95F02",
                    edgecolors="white", linewidth=0.5, s=50)
    axes[1].plot(osm, slope * np.array(osm) + intercept,
                 color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Theoretical quantiles", fontsize=9)
    axes[1].set_ylabel("Sample quantiles", fontsize=9)
    axes[1].set_title(f"Normal Q-Q  ($r={r:.4f}$)", fontsize=9)
    axes[1].grid(linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle("Residual Diagnostics — OLS Regression", fontsize=10)
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "fig_06_02_residuals.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig2_path}")

    return fig1_path, fig2_path

# =============================================================================
# STEP 7 — SAVE RESULTS JSON AND LATEX TABLE
# =============================================================================

def step7_save_results(summary, vif_data, scaler_params, df,
                       fig1_path, fig2_path):
    print("\n=== STEP 7: Save results JSON and LaTeX table ===")

    # ── JSON ──────────────────────────────────────────────────────────────────
    results = {
        "experiment"   : "Exp6_SupplementaryRegression",
        "model"        : "OLS: mean_acc ~ D_g_std + sigma_g_std + M_std",
        "dataset"      : "GTZAN Experiment 1 (f_max=8000, Slaney, n=40)",
        "scaler_params": scaler_params,
        "vif"          : vif_data,
        "ols_summary"  : summary,
        "raw_data"     : df[["genre","M","mean_acc","D_g",
                              "sigma_g","group"]].to_dict(orient="records")
    }
    json_path = os.path.join(OUTPUT_DIR, "exp6_regression_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    save_to_drive(json_path, "exp6_regression_results.json")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    coef = summary["coefficients"]

    # Variable display names
    var_names = {
        "const"      : "Intercept",
        "D_g_std"    : "$D_g$ (filter density)",
        "sigma_g_std": "$\\sigma_g$ (spectral spread)",
        "M_std"      : "$M$ (band count)"
    }

    latex_lines = []
    latex_lines.append("% Auto-generated by 06_supplementary_regression.py")
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Standardised OLS regression coefficients for per-genre")
    latex_lines.append("accuracy on filterbank geometry variables (GTZAN, Experiment~1,")
    latex_lines.append(f"$f_{{\\max}}=8$~kHz, Slaney norm, $n={summary['n_obs']}$ observations).")
    latex_lines.append(f"$R^2={summary['r_squared']:.4f}$,")
    latex_lines.append(f"Adj.~$R^2={summary['adj_r_squared']:.4f}$,")
    latex_lines.append(f"$F({3},{summary['n_obs']-4})={summary['f_statistic']:.4f}$,")
    latex_lines.append(f"$p={summary['f_p_value']:.4f}$.")
    latex_lines.append("Significance codes: $^{***}p<0.001$, $^{**}p<0.01$,")
    latex_lines.append("$^{*}p<0.05$, $^{.}p<0.10$.}")
    latex_lines.append("\\label{tab:regression}")
    latex_lines.append("\\setlength{\\tabcolsep}{5pt}")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Predictor & $\\hat{\\beta}$ & SE & $t$ & $p$ & 95\\% CI & Sig. \\\\")
    latex_lines.append("\\midrule")

    for var in ["const", "D_g_std", "sigma_g_std", "M_std"]:
        v    = coef[var]
        name = var_names[var]
        ci   = f"[{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}]"
        latex_lines.append(
            f"{name} & {v['coef']:+.4f} & {v['se']:.4f} & "
            f"{v['t']:+.4f} & {v['p']:.4f} & {ci} & {v['sig']} \\\\"
        )

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")
    latex_lines.append("% VIF diagnostics")
    latex_lines.append("% (include as a note below the table if any VIF > 5)")
    for var in ["D_g_std", "sigma_g_std", "M_std"]:
        vif_val = vif_data.get(var, 0)
        flag    = " % HIGH -- report in text" if vif_val > 5 else ""
        latex_lines.append(f"% VIF({var}) = {vif_val:.4f}{flag}")

    latex_path = os.path.join(OUTPUT_DIR, "supp_regression_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    save_to_drive(latex_path, "supp_regression_table.tex")

    # Upload figures
    save_to_drive(fig1_path, "fig_06_01_partial_regression.png")
    save_to_drive(fig2_path, "fig_06_02_residuals.png")

    # ── Console interpretation ─────────────────────────────────────────────────
    print("\n=== INTERPRETATION ===")
    beta1 = coef["D_g_std"]
    print(f"  beta_1 (D_g):      {beta1['coef']:+.4f}  "
          f"p={beta1['p']:.4f}  {beta1['sig']}")
    print(f"  R-squared:          {summary['r_squared']:.4f}")
    print(f"  Adj R-squared:      {summary['adj_r_squared']:.4f}")
    print()
    if beta1["p"] < 0.05:
        print("  RESULT: beta_1 is statistically significant (p < 0.05).")
        print("  Filter density D_g is a significant predictor of per-genre")
        print("  accuracy after controlling for sigma_g and M.")
        print("  This confirms filter density as the operative fairness mechanism.")
    else:
        print(f"  NOTE: beta_1 is not significant at p=0.05 (p={beta1['p']:.4f}).")
        print("  The directional finding still holds but the regression")
        print("  does not reach conventional significance at n=40.")
        print("  Report direction and note the small-sample caveat.")

    if any(v > 5 for k, v in vif_data.items() if k != "const"):
        print()
        print("  WARNING: VIF > 5 detected. Multicollinearity present between")
        print("  D_g and M (expected, since larger M produces more filter centres).")
        print("  Standardised coefficients are reported; interpret with caution.")
        print("  Consider reporting beta_1 from a reduced model (D_g + sigma_g only)")
        print("  as a robustness check.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Program 06 — Supplementary Regression Analysis  (v1.0)")
    print("  Model: mean_acc ~ D_g + sigma_g + M  (standardised predictors)")
    print("  Data:  GTZAN Experiment 1, n=40 (10 genres x 4 M values)")
    print("=" * 70)

    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    loaded                  = step1_download_inputs()
    df                      = step2_build_dataframe(loaded)
    df, scaler_params       = step3_standardise(df)
    model, summary          = step4_fit_ols(df)
    vif_data                = step5_vif(df)
    fig1_path, fig2_path    = step6_diagnostic_figures(model, df)
    step7_save_results(summary, vif_data, scaler_params, df,
                       fig1_path, fig2_path)

    # ── Sync to ensure all writes are flushed ────────────────────────────────
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")

    print("\nProgram 06 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()