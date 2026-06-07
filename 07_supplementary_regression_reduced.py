# =============================================================================
# Program      : 07_supplementary_regression_reduced.py
# Version      : 1.0
# Description  : Supplementary Regression — Reduced Model Robustness Check
#
#                Fits a reduced OLS regression model dropping M as a predictor
#                to address the severe multicollinearity between D_g and M
#                identified in Program 06 (VIF(D_g)=65.4, VIF(M)=62.9).
#
#                Full model (Program 06):
#                  A_g = beta_0 + beta_1*D_g + beta_2*sigma_g + beta_3*M + e
#                  VIF(D_g) = 65.4, VIF(M) = 62.9  [severe]
#
#                Reduced model (this program):
#                  A_g = beta_0 + beta_1*D_g + beta_2*sigma_g + e
#
#                Rationale: D_g and M are nearly collinear by construction
#                (larger M mechanically produces more filter centres, so D_g
#                scales with M). Dropping M isolates the effect of filter
#                density independent of band count, which is the theoretically
#                motivated test: does D_g predict accuracy after controlling
#                for genre-intrinsic spread sigma_g alone?
#
#                Both models are reported in the supplementary material.
#                Consistency of beta_1 sign and significance across full and
#                reduced models constitutes a robustness check.
#
# INPUT        :
#                  Drive:
#                    exp6_regression_results.json
#                      (contains raw_data with D_g_std, sigma_g_std, M_std,
#                       mean_acc per genre per M — built by Program 06)
#
# STEPS        :
#                  Step 1  Download exp6_regression_results.json from Drive
#                  Step 2  Reconstruct dataframe from raw_data
#                  Step 3  Standardise predictors (same scaler as Program 06)
#                  Step 4  Fit reduced OLS (D_g_std + sigma_g_std only)
#                  Step 5  Compute VIF for reduced model predictors
#                  Step 6  Compare reduced vs full model coefficients
#                  Step 7  Save results JSON and LaTeX table, upload to Drive
#
# OUTPUT FILES :
#                  exp7_reduced_regression_results.json
#                      Reduced model OLS results: coefficients, SE, t, p, CI,
#                      R-squared, F-statistic, VIF, comparison with full model
#
#                  supp_reduced_regression_table.tex
#                      LaTeX table comparing full and reduced model coefficients
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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR     = "/content/drive/MyDrive/paper/mel-fairness-probe/"  # Persistent storage
LOCAL_WORK_DIR  = "/tmp/gtzan_fairness"
OUTPUT_DIR      = os.path.join(LOCAL_WORK_DIR, "outputs_exp7")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# STEP 1 — DOWNLOAD INPUT FILE
# =============================================================================

def step1_download_inputs():
    print("\n=== STEP 1: Download exp6_regression_results.json from Drive ===")
    local_path = os.path.join(OUTPUT_DIR, "exp6_regression_results.json")
    drive_files = list_drive_files()
    
    if not os.path.exists(local_path):
        if "exp6_regression_results.json" in drive_files:
            if load_from_drive("exp6_regression_results.json", local_path):
                print("  Downloaded: exp6_regression_results.json")
            else:
                raise RuntimeError(
                    "Cannot download exp6_regression_results.json from Drive. "
                    "Run Program 06 first.")
        else:
            raise RuntimeError(
                "exp6_regression_results.json not found on Drive. "
                "Run Program 06 first.")
    else:
        print("  [SKIP] Already local.")
    with open(local_path) as f:
        return json.load(f)

# =============================================================================
# STEP 2 — RECONSTRUCT DATAFRAME
# =============================================================================

def step2_build_dataframe(exp6_data):
    print("\n=== STEP 2: Reconstruct dataframe from Program 06 raw data ===")
    df = pd.DataFrame(exp6_data["raw_data"])
    print(f"  Observations: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    return df, exp6_data["scaler_params"], exp6_data["ols_summary"]

# =============================================================================
# STEP 3 — STANDARDISE PREDICTORS (SAME SCALER AS PROGRAM 06)
# =============================================================================

def step3_standardise(df, scaler_params):
    """
    Apply identical standardisation as Program 06 so coefficients are
    directly comparable across full and reduced models.
    """
    print("\n=== STEP 3: Standardise predictors (same scaler as Program 06) ===")
    for col in ["D_g", "sigma_g", "M"]:
        mean = scaler_params[col]["mean"]
        std  = scaler_params[col]["std"]
        df[f"{col}_std"] = (df[col] - mean) / std
        print(f"  {col}: mean={mean}, std={std} (from Program 06)")
    return df

# =============================================================================
# STEP 4 — FIT REDUCED OLS
# =============================================================================

def step4_fit_reduced_ols(df):
    print("\n=== STEP 4: Fit reduced OLS (D_g_std + sigma_g_std only) ===")

    X = sm.add_constant(df[["D_g_std", "sigma_g_std"]])
    y = df["mean_acc"]

    model = sm.OLS(y, X).fit()
    print(model.summary())

    coef_table = {}
    for var in ["const", "D_g_std", "sigma_g_std"]:
        coef_table[var] = {
            "coef"    : round(float(model.params[var]),           4),
            "se"      : round(float(model.bse[var]),              4),
            "t"       : round(float(model.tvalues[var]),          4),
            "p"       : round(float(model.pvalues[var]),          4),
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
        "n_obs"        : int(model.nobs),
        "r_squared"    : round(float(model.rsquared),     4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_statistic"  : round(float(model.fvalue),       4),
        "f_p_value"    : round(float(model.f_pvalue),     4),
        "aic"          : round(float(model.aic),          4),
        "bic"          : round(float(model.bic),          4),
        "coefficients" : coef_table
    }

    print(f"\n  R² = {summary['r_squared']:.4f}, "
          f"Adj R² = {summary['adj_r_squared']:.4f}, "
          f"F = {summary['f_statistic']:.4f} "
          f"(p = {summary['f_p_value']:.4f})")
    print(f"\n  Coefficient significance:")
    for var, vals in coef_table.items():
        if var == "const":
            continue
        print(f"    {var:<15s}  β={vals['coef']:+.4f}  "
              f"SE={vals['se']:.4f}  t={vals['t']:+.4f}  "
              f"p={vals['p']:.4f}  {vals['sig']}")

    return model, summary

# =============================================================================
# STEP 5 — VIF FOR REDUCED MODEL
# =============================================================================

def step5_vif_reduced(df):
    print("\n=== STEP 5: VIF for reduced model predictors ===")
    X = sm.add_constant(df[["D_g_std", "sigma_g_std"]])
    vif_data = {}
    for i, col in enumerate(["const", "D_g_std", "sigma_g_std"]):
        vif = variance_inflation_factor(X.values, i)
        vif_data[col] = round(float(vif), 4)
        if col != "const":
            flag = " [HIGH]" if vif > 5 else " [OK]"
            print(f"  VIF({col}) = {vif:.4f}{flag}")
    return vif_data

# =============================================================================
# STEP 6 — COMPARE FULL VS REDUCED MODEL
# =============================================================================

def step6_compare_models(full_summary, reduced_summary):
    print("\n=== STEP 6: Full vs Reduced model comparison ===")
    print(f"\n  {'Metric':<20}  {'Full model':>12}  {'Reduced model':>14}")
    print("  " + "-" * 50)

    metrics = [
        ("R²",         "r_squared"),
        ("Adj R²",     "adj_r_squared"),
        ("AIC",        "aic"),
        ("BIC",        "bic"),
        ("F p-value",  "f_p_value")
    ]
    for label, key in metrics:
        full_val    = full_summary[key]
        reduced_val = reduced_summary[key]
        print(f"  {label:<20}  {full_val:>12.4f}  {reduced_val:>14.4f}")

    print(f"\n  beta_1 (D_g) comparison:")
    full_b1    = full_summary["coefficients"]["D_g_std"]
    reduced_b1 = reduced_summary["coefficients"]["D_g_std"]
    print(f"    Full model    : β={full_b1['coef']:+.4f}  "
          f"SE={full_b1['se']:.4f}  p={full_b1['p']:.4f}  {full_b1['sig']}")
    print(f"    Reduced model : β={reduced_b1['coef']:+.4f}  "
          f"SE={reduced_b1['se']:.4f}  p={reduced_b1['p']:.4f}  "
          f"{reduced_b1['sig']}")

    same_sign = (full_b1["coef"] * reduced_b1["coef"]) > 0
    both_sig  = full_b1["p"] < 0.05 and reduced_b1["p"] < 0.05
    print(f"\n  Same sign across models: {same_sign}")
    print(f"  Significant in both:     {both_sig}")

    if same_sign and both_sig:
        print("\n  ROBUSTNESS CONFIRMED: beta_1 is significant and consistent")
        print("  in direction across both the full and reduced models.")
        print("  The effect of filter density on accuracy is robust to")
        print("  the inclusion/exclusion of M as a covariate.")
    elif same_sign and not both_sig:
        print("\n  PARTIAL ROBUSTNESS: beta_1 direction is consistent but")
        print("  significance differs across models. Report both with caution.")
    else:
        print("\n  ROBUSTNESS CONCERN: beta_1 changes sign or significance.")
        print("  Multicollinearity may be distorting the full model estimate.")

    return {"same_sign": same_sign, "both_significant": both_sig}

# =============================================================================
# STEP 7 — SAVE RESULTS JSON AND LATEX TABLE
# =============================================================================

def step7_save_results(full_summary, reduced_summary, vif_full,
                       vif_reduced, comparison):
    print("\n=== STEP 7: Save results JSON and LaTeX comparison table ===")

    # ── JSON ──────────────────────────────────────────────────────────────────
    results = {
        "experiment"      : "Exp7_ReducedRegressionRobustness",
        "full_model"      : "OLS: mean_acc ~ D_g_std + sigma_g_std + M_std",
        "reduced_model"   : "OLS: mean_acc ~ D_g_std + sigma_g_std",
        "vif_full"        : vif_full,
        "vif_reduced"     : vif_reduced,
        "full_summary"    : full_summary,
        "reduced_summary" : reduced_summary,
        "robustness_check": comparison
    }
    json_path = os.path.join(OUTPUT_DIR,
                             "exp7_reduced_regression_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    save_to_drive(json_path, "exp7_reduced_regression_results.json")

    # ── LaTeX comparison table ─────────────────────────────────────────────────
    var_names = {
        "const"      : "Intercept",
        "D_g_std"    : "$D_g$ (filter density)",
        "sigma_g_std": "$\\sigma_g$ (spectral spread)",
        "M_std"      : "$M$ (band count)"
    }

    latex_lines = []
    latex_lines.append("% Auto-generated by 07_supplementary_regression_reduced.py")
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Standardised OLS regression coefficients:")
    latex_lines.append("full model ($D_g$, $\\sigma_g$, $M$) vs reduced model")
    latex_lines.append("($D_g$, $\\sigma_g$) as a multicollinearity robustness check.")
    latex_lines.append(f"Full: $R^2={full_summary['r_squared']:.4f}$,")
    latex_lines.append(f"$F={full_summary['f_statistic']:.4f}$")
    latex_lines.append(f"($p={full_summary['f_p_value']:.4f}$),")
    latex_lines.append(f"VIF($D_g$)$={vif_full.get('D_g_std',0):.1f}$.")
    latex_lines.append(f"Reduced: $R^2={reduced_summary['r_squared']:.4f}$,")
    latex_lines.append(f"$F={reduced_summary['f_statistic']:.4f}$")
    latex_lines.append(f"($p={reduced_summary['f_p_value']:.4f}$),")
    latex_lines.append(f"VIF($D_g$)$={vif_reduced.get('D_g_std',0):.1f}$.")
    latex_lines.append("Significance: $^{***}p<0.001$, $^{**}p<0.01$,")
    latex_lines.append("$^{*}p<0.05$, $^{.}p<0.10$.}")
    latex_lines.append("\\label{tab:regression_comparison}")
    latex_lines.append("\\setlength{\\tabcolsep}{4pt}")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("& \\multicolumn{2}{c}{Full model} "
                       "& \\multicolumn{2}{c}{Reduced model} \\\\")
    latex_lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    latex_lines.append("Predictor & $\\hat{\\beta}$ & $p$ "
                       "& $\\hat{\\beta}$ & $p$ \\\\")
    latex_lines.append("\\midrule")

    full_coef    = full_summary["coefficients"]
    reduced_coef = reduced_summary["coefficients"]

    for var in ["const", "D_g_std", "sigma_g_std"]:
        name    = var_names[var]
        fb      = full_coef[var]
        rb      = reduced_coef[var]
        latex_lines.append(
            f"{name} & ${fb['coef']:+.4f}^{{{fb['sig']}}}$ & {fb['p']:.4f} "
            f"& ${rb['coef']:+.4f}^{{{rb['sig']}}}$ & {rb['p']:.4f} \\\\"
        )

    # M only in full model
    mb = full_coef["M_std"]
    latex_lines.append(
        f"{var_names['M_std']} & ${mb['coef']:+.4f}^{{{mb['sig']}}}$ "
        f"& {mb['p']:.4f} & \\multicolumn{{2}}{{c}}{{---}} \\\\"
    )

    latex_lines.append("\\midrule")
    latex_lines.append(
        f"$R^2$ & \\multicolumn{{2}}{{c}}{{{full_summary['r_squared']:.4f}}} "
        f"& \\multicolumn{{2}}{{c}}{{{reduced_summary['r_squared']:.4f}}} \\\\"
    )
    latex_lines.append(
        f"Adj.~$R^2$ & \\multicolumn{{2}}{{c}}"
        f"{{{full_summary['adj_r_squared']:.4f}}} "
        f"& \\multicolumn{{2}}{{c}}"
        f"{{{reduced_summary['adj_r_squared']:.4f}}} \\\\"
    )
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_path = os.path.join(OUTPUT_DIR,
                              "supp_reduced_regression_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    save_to_drive(latex_path, "supp_reduced_regression_table.tex")

    # ── Final interpretation ──────────────────────────────────────────────────
    print("\n=== FINAL INTERPRETATION ===")
    print(f"  Full model    R²={full_summary['r_squared']:.4f}  "
          f"VIF(D_g)={vif_full.get('D_g_std',0):.1f}")
    print(f"  Reduced model R²={reduced_summary['r_squared']:.4f}  "
          f"VIF(D_g)={vif_reduced.get('D_g_std',0):.1f}")
    print()
    b1_full    = full_summary["coefficients"]["D_g_std"]
    b1_reduced = reduced_summary["coefficients"]["D_g_std"]
    print(f"  beta_1 full    : {b1_full['coef']:+.4f}  "
          f"p={b1_full['p']:.4f}  {b1_full['sig']}")
    print(f"  beta_1 reduced : {b1_reduced['coef']:+.4f}  "
          f"p={b1_reduced['p']:.4f}  {b1_reduced['sig']}")
    print()
    if comparison["same_sign"] and comparison["both_significant"]:
        print("  CONCLUSION: Filter density D_g is a robust, statistically")
        print("  significant predictor of per-genre accuracy in both models.")
        print("  The negative beta_1 reflects that high D_g is a marker of")
        print("  high spectral spread (the disadvantaged group), not that")
        print("  more filters cause lower accuracy.")
        print("  Suitable for inclusion in supplementary Page S1.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Program 07 — Reduced Regression Robustness Check  (v1.0)")
    print("  Reduced model: mean_acc ~ D_g_std + sigma_g_std")
    print("  (M dropped to address VIF(D_g)=65.4, VIF(M)=62.9 from Prog 06)")
    print("=" * 70)

    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    exp6_data                    = step1_download_inputs()
    df, scaler_params, full_summ = step2_build_dataframe(exp6_data)
    df                           = step3_standardise(df, scaler_params)
    model_reduced, reduced_summ  = step4_fit_reduced_ols(df)
    vif_reduced                  = step5_vif_reduced(df)
    comparison                   = step6_compare_models(full_summ,
                                                         reduced_summ)
    step7_save_results(full_summ, reduced_summ,
                       exp6_data["vif"], vif_reduced, comparison)

    # ── Sync to ensure all writes are flushed ────────────────────────────────
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")

    print("\nProgram 07 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()