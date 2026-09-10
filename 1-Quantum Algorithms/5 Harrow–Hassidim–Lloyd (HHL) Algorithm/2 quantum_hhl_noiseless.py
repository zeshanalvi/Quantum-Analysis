
# Ideal (Noiseless) HHL Algorithm:: Quantum Baseline

# ----------------------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# How to run from Windows CMD (example):
# > python filename (Dear Researcher! You can change as per your requirements/logic).py ^
"""
To Run, I used the following baselines:

  python quantum_hhl_noiseless.py ^
    --sizes 200 400 800 1200 ^
    --trials 3 ^
    --excel "Filtered CO2 intensity 236 Countries.xlsx" ^
    --device-power-watts 65 --pue 1.2 ^
    --year-select latest ^
    --outdir carbon_by_country ^
    --combine
"""


# ------------********------------ Imports

#Imports standard libraries for filesystem operations, timing, argument parsing, memory tracking, and warnings management.
#Loads NumPy, Pandas, and Matplotlib (in non-interactive Agg mode) for numerical computation, data handling, and plot generation.

import os, sys, time, argparse, tracemalloc, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------
# ============================== Ideal HHL Core ==============================

def hhl_ideal_solve_diagonal(A_diag: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Ideal noiseless HHL effect for diagonal Hermitian positive-definite A:
      |x> ∝ A^{-1} |b>
    Returns the (unnormalized) classical vector x_pred = A^{-1} b,
    which matches the exact solution in this idealized setting.
    """
    # Guard against tiny eigenvalues to avoid division issues
    eps = 1e-12
    return b / np.maximum(A_diag, eps)

def run_experiment(N: int, seed: int) -> dict:
    """
    Generate a Hermitian positive-definite diagonal A of size N,
    a ground-truth x_true, set b := A x_true, and recover x via ideal HHL.
    Measure runtime, peak memory, and relative error vs x_true.
    """
    rng = np.random.default_rng(seed)

    # Construct diagonal SPD matrix: diag entries approx in [5,6)
    A_diag = 5.0 + rng.random(N)

    # Ground-truth solution and right-hand side
    x_true = rng.random(N)
    b = A_diag * x_true

    # Measure time + peak memory for the HHL pipeline
    tracemalloc.start()
    t0 = time.perf_counter()

    # ------------********------------ Ideal HHL solve (noiseless) ---
    x_pred = hhl_ideal_solve_diagonal(A_diag, b)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Accuracy: same metric as classical baseline
    rel_error = np.linalg.norm(x_pred - x_true) / np.linalg.norm(x_true)

    return {
        "N": N,
        "seed": seed,
        "method": "hhl_ideal_diagonal",
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024 ** 2)),
        "rel_error": float(rel_error),
    }

# ------------********------------ Carbon I/O

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists():
        return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

def load_carbon_excel(path: str, year_select: str = "latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    cand_country = next((v for k, v in cols.items() if "country" in k or "nation" in k or k == "location"), None)
    if cand_country is None:
        raise ValueError("No 'Country' column found.")
    cand_year = next((v for k, v in cols.items() if "year" in k or "date" in k), None)
    cand_intensity = next((v for k, v in cols.items()
                           if "intensity" in k or ("co2" in k and "kwh" in k) or "kgco2" in k or "gco2" in k), None)
    if cand_intensity is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if cand_year in numeric_cols:
            numeric_cols.remove(cand_year)
        if not numeric_cols:
            raise ValueError("No numeric intensity column detected.")
        cand_intensity = numeric_cols[0]
    keep = [cand_country] + ([cand_year] if cand_year else []) + [cand_intensity]
    df = df[keep].copy()
    df.columns = ["Country", "Year", "Intensity"] if len(keep) == 3 else ["Country", "Intensity"]
    if "Year" in df.columns and year_select.lower() == "latest":
        df = df.sort_values(["Country", "Year"]).groupby("Country", as_index=False).tail(1)

    # Heuristic unit fix: if median looks like g/kWh, convert to kg/kWh
    med = float(df["Intensity"].dropna().median())
    if med > 50:
        df["Intensity"] = df["Intensity"] / 1000.0

    return df.dropna(subset=["Country", "Intensity"]).reset_index(drop=True)

def compute_carbon(perf_df: pd.DataFrame, intensity_df: pd.DataFrame,
                   power_watts: float, pue: float) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return df, summary

# ------------********------------ Plot Helpers =

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    # Matplotlib 3.9 renamed 'labels' -> 'tick_labels'
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs Matrix Size"); plt.xlabel("Matrix Size N"); plt.ylabel("Runtime (s)")
    plt.grid(True); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["rel_error"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["rel_error"], marker="s")
    plt.title("Performance: Mean Relative Error vs N"); plt.xlabel("Matrix Size N"); plt.ylabel("Relative Error")
    plt.grid(True); plt.savefig(os.path.join(outdir, "perf_error_vs_N.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["peak_mem_mb"], marker="^")
    plt.title("Scalability: Memory vs Matrix Size"); plt.xlabel("Matrix Size N"); plt.ylabel("Memory (MB)")
    plt.grid(True); plt.savefig(os.path.join(outdir, "scal_memory_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g2["N"], g2["runtime_s"], marker="o")
    plt.title("Scalability: Runtime (log-log)"); plt.xlabel("N"); plt.ylabel("Runtime (s)")
    plt.grid(True, which="both"); plt.savefig(os.path.join(outdir, "scal_loglog_runtime.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    data = [df[df["N"] == n]["rel_error"].values for n in sorted(df["N"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["N"].unique()))
    plt.title("Reliability: Error Distribution by N"); plt.xlabel("Matrix Size N"); plt.ylabel("Relative Error")
    plt.grid(True); plt.savefig(os.path.join(outdir, "rel_error_boxplot.png"), **plt_kwargs); plt.close()

    g = df.groupby("N")["rel_error"].std().reset_index()
    plt.figure(); plt.plot(g["N"], g["rel_error"], marker="o")
    plt.title("Reliability: Std. Dev. of Relative Error vs N"); plt.xlabel("N"); plt.ylabel("Std(Relative Error)")
    plt.grid(True); plt.savefig(os.path.join(outdir, "rel_error_std_vs_N.png"), **plt_kwargs); plt.close()

def plot_carbon(df: pd.DataFrame, outdir: str):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5))
    plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 Countries (kgCO2e)")
    plt.xlabel("kg CO2e"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=25)
    plt.title("Carbon: Emission Distribution"); plt.xlabel("kg CO2e"); plt.grid(True)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# ------------********------------ Main  ------------********------------

def main():
    parser = argparse.ArgumentParser(description="Ideal (Noiseless) HHL Benchmarks – Quantum Baseline")
    # Keep the SAME parameter names as the classical script:
    parser.add_argument("--sizes", nargs="+", type=int, default=[200, 400, 800, 1200])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--method", type=str, default="hhl", choices=["hhl"])  # preserved flag; fixed to 'hhl'
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    args = parser.parse_args()

    # Output folders (same structure)
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting ideal HHL (noiseless) benchmarks...")
    jobs = [(N, 1000 + i) for N in args.sizes for i in range(args.trials)]
    rows = []

    def _consume(r: dict):
        rows.append(r)
        print(f"  - N={r['N']} seed={r['seed']} runtime={r['runtime_s']:.6f}s rel_error={r['rel_error']:.2e}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_experiment, N, seed) for (N, seed) in jobs]
            for f in as_completed(futs):
                _consume(f.result())
    else:
        for (N, seed) in jobs:
            _consume(run_experiment(N, seed))

    df = pd.DataFrame(rows)

# ------------********------------
    # ----- Performance -----

    perf_path = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw")
            df.groupby("N").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

# ------------********------------
    # ----- Scalability -----

    scal_path = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw")
    plot_scalability(df, scal_dir)

# ------------********------------
    # ----- Reliability -----

    rel_path = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_path, engine="openpyxl") as w:
            df[["N", "seed", "rel_error"]].to_excel(w, index=False, sheet_name="errors")
    plot_reliability(df, rel_dir)

# ------------********------------
    # ----- Carbon Footprint (from PERFORMANCE runtimes only) -----

    excel_path = resolve_excel_path(args.excel)
    try:
        intensity_df = load_carbon_excel(excel_path, year_select=args.year_select)
        carbon_df, summary_df = compute_carbon(df, intensity_df, args.device_power_watts, args.pue)
        carb_path = os.path.join(carb_dir, "carbon_results.xlsx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.ExcelWriter(carb_path, engine="openpyxl") as w:
                carbon_df.to_excel(w, index=False, sheet_name="per_country")
                summary_df.to_excel(w, index=False, sheet_name="summary")
        plot_carbon(carbon_df, carb_dir)
        print(f"[carbon] Results saved to {carb_dir}")
    except Exception as e:
        print(f"[carbon] Failed: {e}")

    print("\n=== Summary ===")
    print(f"Performance → {perf_path}")
    print(f"Scalability → {scal_path}")
    print(f"Reliability → {rel_path}")
    print(f"Carbon → {os.path.join(carb_dir, 'carbon_results.xlsx')}")
    print("All results + plots saved in respective folders.")

if __name__ == "__main__":
    main()
