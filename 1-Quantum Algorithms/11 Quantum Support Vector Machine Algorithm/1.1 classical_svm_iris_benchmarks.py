#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Classical SVM on Iris
# ----------------------------------------------------------------------------------------------------------------

# Notes: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Code focuses on Performance, Scalability, Reliability, Carbon footprints
# Output folders are:
    #   Performance = runtime
    #   Scalability = growth of runtime as N increases (digits/bits).
    #   Reliability = success rate + dispersion across repetitions.
    #   Carbon footprints = derived *only* from PERFORMANCE totals
    #   (sum of successful wall-times), device power 65 Watts fixed for comparasion, PUE 1.2 fixed, 
    #   country carbon intensity (kg CO2/kWh). I did not use scalability or reliability results for CO2.

#      Carbon Data File: 
#      Uses ONLY Performance runtimes (not Scalability/Reliability), 
#      Placed on my Desktop Excel of country intensities "Filtered CO2 intensity 236 Countries.xlsx"
#      (CSV or XLSX; year column optional. I have selected --year-select latest.)
# ----------------------------------------------------------------------------------------------------------------

# ------------********------------ Imports
#These imports provide core utilities for filesystem access, timing, warnings, and memory profiling alongside numerical and data-analysis tools.
#They configure Matplotlib for headless plotting and enable parallel execution of experiments using process pools.
#Scikit-learn modules are included for dataset loading, preprocessing, dimensionality reduction, SVM training, and evaluation metrics.

import os, sys, time, argparse, warnings, pathlib, tracemalloc, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# sklearn
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hinge_loss

# ======================= Utilities =======================

def _parse_gamma(g):
    # Accept "scale", "auto", or numeric
    if isinstance(g, str):
        g = g.strip().lower()
        if g in ("scale", "auto"):
            return g
        try:
            return float(g)
        except:
            return "scale"
    return float(g)

def safe_hinge_loss(y_true, df_scores, classes):
    
    #Version-proof hinge loss:
    #1) Try sklearn.metrics.hinge_loss with labels=classes (1.7.x and many versions).
    #2) Fallback: manual One-vs-Rest hinge loss:
    #      mean_{i,c} max(0, 1 - y_ic * f_ic), where y_ic ∈ {-1, +1}
    #Works for binary or multiclass.
    
    y_true = np.asarray(y_true)
    df_scores = np.asarray(df_scores)

    
    try:
        return float(hinge_loss(y_true, df_scores, labels=classes))
    except TypeError:
        pass  

    # Manual OvR hinge loss
    if df_scores.ndim == 1:
        # Binary SVM gives (n,) scores; map to 2D with 2 classes for a uniform formula
        # Class order: classes[0], classes[1]
        df_scores = df_scores.reshape(-1, 1)
        # Create two-class scores: [ -s, +s ] so margin aligns with OvR intuition
        df_scores = np.hstack([-df_scores, df_scores])
        classes = np.array(classes[:2])

    Y_bin = label_binarize(y_true, classes=classes)  # (n_samples, n_classes) with {0,1}
    Y = 2.0 * Y_bin - 1.0                            # -> {-1, +1}
    loss = np.maximum(0.0, 1.0 - Y * df_scores)      # elementwise
    return float(np.mean(loss))

# ======================= One experiment run =======================
#def run_single executes one classical SVM experiment on the Iris dataset with a scale → PCA → SVM pipeline.
#It clamps PCA components to a valid range, trains and evaluates the model, and computes accuracy and hinge loss metrics.
#The function profiles runtime and peak memory usage, 
#   then returns all hyperparameters, metrics, and resource statistics in a result dictionary.

def run_single(n_components: int, max_iter: int, kernel: str, Cval: float, gamma, degree: int,
               test_size: float, acc_tol: float, seed: int) -> dict:
    # Clamp PCA components to [1..4] for iris
    n_comp_eff = int(max(1, min(4, n_components)))

    # Load data
    iris = datasets.load_iris()
    X_all = iris.data.astype(np.float64)   # (150,4)
    y_all = iris.target.astype(np.int32)   # {0,1,2}

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )

    # Pipeline: scale -> PCA -> SVM
    scaler = StandardScaler()
    pca = PCA(n_components=n_comp_eff, random_state=seed)
    clf = SVC(
        kernel=kernel,
        C=float(Cval),
        gamma=_parse_gamma(gamma),
        degree=int(degree),
        max_iter=int(max_iter),     # -1 means no cap
        decision_function_shape="ovr",
        probability=False,
        random_state=seed
    )

    tracemalloc.start()
    t0 = time.perf_counter()

    Xtr = scaler.fit_transform(X_train)
    Xtr = pca.fit_transform(Xtr)
    clf.fit(Xtr, y_train)

    Xte = scaler.transform(X_test)
    Xte = pca.transform(Xte)
    y_pred = clf.predict(Xte)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    df_scores = clf.decision_function(Xte)  # shape (n_samples, n_classes) for multiclass
    hloss = safe_hinge_loss(y_test, df_scores, clf.classes_)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "pca_components": int(n_components),
        "pca_components_eff": int(n_comp_eff),
        "kernel": str(kernel),
        "C": float(Cval),
        "gamma": gamma,
        "degree": int(degree),
        "max_iter": int(max_iter),
        "test_size": float(test_size),
        "seed": int(seed),
        "accuracy": float(acc),
        "hinge_loss": float(hloss),
        "success": bool(acc >= acc_tol),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }

# ======================= Carbon accounting (Performance only) =======================

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists(): return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

def load_carbon_excel(path: str, year_select: str = "latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
    if country is None: raise ValueError("No 'Country' column found in Excel.")
    year = next((v for k,v in cols.items() if "year" in k or "date" in k), None)
    intensity = next((v for k,v in cols.items()
                      if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)
    if intensity is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if year in numeric: numeric.remove(year)
        if not numeric: raise ValueError("No numeric intensity column detected.")
        intensity = numeric[0]
    keep = [country] + ([year] if year else []) + [intensity]
    df = df[keep].copy()
    df.columns = ["Country","Year","Intensity"] if len(keep)==3 else ["Country","Intensity"]
    if "Year" in df.columns and year_select.lower()=="latest":
        df = df.sort_values(["Country","Year"]).groupby("Country", as_index=False).tail(1)
    med = float(df["Intensity"].dropna().median())
    if med > 50: df["Intensity"] = df["Intensity"] / 1000.0  # g/kWh -> kg/kWh
    return df.dropna(subset=["Country","Intensity"]).reset_index(drop=True)

def compute_carbon(perf_df: pd.DataFrame, intensity_df: pd.DataFrame,
                   power_watts: float, pue: float):
    total_runtime_s = float(perf_df["runtime_s"].sum())
    kWh_total = power_watts * pue * total_runtime_s / 3_600_000.0
    df = intensity_df.copy()
    df["kWh"] = kWh_total
    df["kgCO2e"] = df["Intensity"] * df["kWh"]
    summary = pd.DataFrame({
        "total_runtime_s": [total_runtime_s],
        "power_watts": [power_watts],
        "PUE": [pue],
        "kWh_total": [kWh_total],
        "median_intensity_kg_per_kWh": [float(df["Intensity"].median())],
        "mean_kgCO2e_across_countries": [float(df["kgCO2e"].mean())],
    })
    return df.sort_values("kgCO2e", ascending=False), summary

# ======================= Plotting helpers =======================

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("pca_components_eff")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["pca_components_eff"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs PCA components"); plt.xlabel("PCA components"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_components.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("pca_components_eff")["accuracy"].mean().reset_index()
    plt.figure(); plt.plot(g2["pca_components_eff"], g2["accuracy"], marker="s")
    plt.title("Performance: Mean accuracy vs PCA components"); plt.xlabel("PCA components"); plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_accuracy_vs_components.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    g = df.groupby("pca_components_eff")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["pca_components_eff"], g["runtime_s"], "o")
    plt.title("Scalability: Runtime (log–log) vs PCA components"); plt.xlabel("PCA components"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_components.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("pca_components_eff")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["pca_components_eff"], g2["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs PCA components"); plt.xlabel("PCA components"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_components.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("pca_components_eff")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["pca_components_eff"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs PCA components (acc ≥ tol)"); plt.xlabel("PCA components"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_components.png"), **plt_kwargs); plt.close()

    data = [df[df["pca_components_eff"]==k]["accuracy"].values for k in sorted(df["pca_components_eff"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["pca_components_eff"].unique()))
    plt.title("Reliability: Accuracy distribution"); plt.xlabel("PCA components"); plt.ylabel("Accuracy")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_accuracy_boxplot.png"), **plt_kwargs); plt.close()

def plot_carbon(df: pd.DataFrame, outdir: str):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# ======================= Main =======================

def main():
    parser = argparse.ArgumentParser(description="Classical SVM on Iris — Benchmarks (version-proof)")
 
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="PCA components to use (1..4).")
    parser.add_argument("--steps", type=int, default=1000, help="SVM max_iter (−1 = no limit).")
    parser.add_argument("--trials", type=int, default=10, help="Trials per size (different seeds).")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear","rbf","poly","sigmoid"],
                        help="SVM kernel.")
    parser.add_argument("--C", type=float, default=1.0, help="SVM C (regularization).")
    parser.add_argument("--gamma", type=str, default="scale", help="SVM gamma ('scale','auto' or numeric).")
    parser.add_argument("--degree", type=int, default=3, help="Degree for poly kernel.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test fraction for train/test split.")
    parser.add_argument("--acc-tol", type=float, default=0.95, help="Success if accuracy ≥ acc_tol.")
    # carbon/infra
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    args = parser.parse_args()

    # Output folders
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting SVM (Iris) benchmarks...")
    rows = []
    jobs = []
    for comp in args.sizes:
        for i in range(args.trials):
            seed = 1000 + 17*int(comp) + i
            jobs.append((comp, args.steps, args.kernel, args.C, args.gamma, args.degree,
                         args.test_size, args.acc_tol, seed))

#consume collects a single experiment result, appends it to the results list, and prints a concise progress/status line.
#This block executes all jobs either in parallel using ProcessPoolExecutor or sequentially, depending on worker availability.
#After all runs complete, it aggregates the collected results into a pandas DataFrame for downstream analysis and reporting.

    def consume(res: dict):
        rows.append(res)
        print(f"  - PCA={res['pca_components_eff']} seed={res['seed']} "
              f"acc={res['accuracy']:.3f} runtime={res['runtime_s']:.4f}s success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, *job) for job in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for job in jobs:
            consume(run_single(*job))

    df = pd.DataFrame(rows)

    # ----- Performance -----
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("pca_components_eff").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                mean_accuracy=("accuracy","mean"),
                mean_hinge_loss=("hinge_loss","mean"),
                success_rate=("success","mean"),
                mean_train_samples=("n_train","mean"),
                mean_test_samples=("n_test","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")

    plot_performance(df, perf_dir)

    # ----- Scalability -----
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["pca_components_eff","runtime_s","peak_mem_mb"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("pca_components_eff").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")

    plot_scalability(df, scal_dir)

    # ----- Reliability -----
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["pca_components_eff","seed","accuracy","hinge_loss","success"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("pca_components_eff").agg(
                success_rate=("success","mean"),
                mean_acc=("accuracy","mean"),
                std_acc=("accuracy","std"),
                mean_hinge=("hinge_loss","mean"),
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")

    plot_reliability(df, rel_dir)

    # ----- Carbon (PERFORMANCE runtime only) -----
    excel_path = resolve_excel_path(args.excel)
    try:
        intensity_df = load_carbon_excel(excel_path, year_select=args.year_select)
        carbon_df, summary_df = compute_carbon(df, intensity_df,
                                               power_watts=args.device_power_watts,
                                               pue=args.pue)
        carb_xlsx = os.path.join(carb_dir, "carbon_results.xlsx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.ExcelWriter(carb_xlsx, engine="openpyxl") as w:
                carbon_df.to_excel(w, index=False, sheet_name="per_country")
                summary_df.to_excel(w, index=False, sheet_name="summary")
                intensity_df.to_excel(w, index=False, sheet_name="intensity_input")

        # Plots
        top = carbon_df.nlargest(15, "kgCO2e")
        plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
        plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
        plt.tight_layout(); plt.savefig(os.path.join(carb_dir, "carbon_top15.png"), **plt_kwargs); plt.close()

        plt.figure(); plt.hist(carbon_df["kgCO2e"], bins=30)
        plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(carb_dir, "carbon_distribution.png"), **plt_kwargs); plt.close()

        print(f"[carbon] Wrote Excel and plots to: {carb_dir}")
    except Exception as e:
        print(f"[carbon] ERROR reading Excel '{excel_path}': {e}")
        print("[carbon] Skipping carbon benchmark. Re-run once the file is available.")

    print("\n=== Summary ===")
    print("Performance Excel:", perf_xlsx)
    print("Scalability Excel:", scal_xlsx)
    print("Reliability Excel:", rel_xlsx)
    print("Carbon Excel:", os.path.join(carb_dir, "carbon_results.xlsx"))
    print("Folders created:\n -", perf_dir, "\n -", scal_dir, "\n -", rel_dir, "\n -", carb_dir)

if __name__ == "__main__":
    main()
