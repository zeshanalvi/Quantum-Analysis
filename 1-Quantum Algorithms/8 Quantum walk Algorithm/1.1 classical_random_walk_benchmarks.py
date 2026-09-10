#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Classical Random Walk on Classical Computer
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

"""
Classical Random Walk (Reflecting 1-D Lattice) — Empirical Benchmarks

Folders created (in current working directory):
  Performance/
  Scalability/
  Reliability/
  Carbon footprints/<outdir>/

Default problem:
  - num_positions (N) = 21   (odd; center start at N//2)
  - num_steps      (T) = 1000
  - boundaries: reflecting
"""

# ------------********------------ Imports

import os, sys, time, argparse, warnings, pathlib, tracemalloc, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------
# ========================= Core: exact distribution (reflecting) 
# ------------********------------

#def exact_distribution_reflecting () computes the exact probability distribution of a 1-D random walk with reflecting boundaries.
#It iteratively updates position probabilities according to 
#    deterministic edge transitions and symmetric interior steps.
#It returns the final distribution after a specified number of steps starting from a chosen position.

def exact_distribution_reflecting(N: int, T: int, start: int | None = None) -> np.ndarray:
    """
    Exact distribution after T steps on a 1-D line of N positions (0..N-1) with reflecting boundaries.
    Transition:
      from 0     -> 1 with prob 1
      from N-1   -> N-2 with prob 1
      from i in 1..N-2 -> i-1 with 1/2, i+1 with 1/2
    """
    if start is None:
        start = N // 2
    p = np.zeros(N, dtype=np.float64)
    p[start] = 1.0
    for _ in range(T):
        pn = np.zeros_like(p)
        # from 0 -> 1
        pn[1] += p[0]
        # from N-1 -> N-2
        pn[N-2] += p[N-1]
        # interior
        if N > 2:
            pn[0:N-2] += 0.5 * p[1:N-1]  # from i in 1..N-2 to i-1
            pn[2:N]   += 0.5 * p[1:N-1]  # from i in 1..N-2 to i+1
        p = pn
    return p

# ------------********------------
# ========================= Monte Carlo sampler (empirical distribution) 
# ------------********------------

#  def simulate_random_walk_reflecting ()  function simulates multiple independent 1-D random walks with reflecting boundaries.

def simulate_random_walk_reflecting(N: int, T: int, paths: int, seed: int, start: int | None = None) -> np.ndarray:
    """
    Simulate 'paths' independent T-step walks; return empirical distribution of final positions.
    Reflecting boundaries; equal prob ±1 at interior sites; start at center by default.
    """
    if start is None:
        start = N // 2
    rng = np.random.default_rng(seed)
    counts = np.zeros(N, dtype=np.int64)
    for _ in range(paths):
        pos = start
        # simulate T steps
        for _ in range(T):
            if pos == 0:
                pos = 1
            elif pos == N-1:
                pos = N-2
            else:
                # left/right with equal prob
                if rng.random() < 0.5:
                    pos -= 1
                else:
                    pos += 1
        counts[pos] += 1
    return counts / counts.sum()

# ------------********------------
# ========================= Metric: Total Variation (TV) distance 
# ------------********------------

#This function computes the total variation distance between two probability distributions.
#It calculates half of the L1 norm of their difference. It returns a scalar measure of distributional divergence.

def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance = 0.5 * L1 norm."""
    return 0.5 * float(np.abs(p - q).sum())

# ------------********------------
# ========================= One experiment 
# ------------********------------

#This following function runs a single experiment comparing exact and 
#    simulated random-walk distributions.
#It measures total variation distance, runtime, and peak memory usage for the simulation.
#It returns performance metrics along with a success flag 
#     based on a tolerance threshold.

def run_single(N: int, T: int, paths: int, tv_tol: float, seed: int) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()

    p_exact = exact_distribution_reflecting(N, T)
    p_emp   = simulate_random_walk_reflecting(N, T, paths, seed)

    tv = tv_distance(p_emp, p_exact)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "n": N,                        # number of positions
        "steps": T,                    # time horizon
        "paths": paths,                # Monte Carlo paths
        "seed": seed,
        "tv_distance": tv,
        "success": bool(tv <= tv_tol),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
    }

# ------------********------------
# ========================= Carbon I/O (Performance only)
# ------------********------------

# Resolving carbon file paths

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists():
        return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

def load_carbon_excel(path: str, year_select: str = "latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
    if country is None:
        raise ValueError("No 'Country' column found.")
    year = next((v for k,v in cols.items() if "year" in k or "date" in k), None)
    intensity = next((v for k,v in cols.items()
                      if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)
    if intensity is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if year in numeric: numeric.remove(year)
        if not numeric:
            raise ValueError("No numeric intensity column detected.")
        intensity = numeric[0]
    keep = [country] + ([year] if year else []) + [intensity]
    df = df[keep].copy()
    df.columns = ["Country","Year","Intensity"] if len(keep)==3 else ["Country","Intensity"]
    if "Year" in df.columns and year_select.lower()=="latest":
        df = df.sort_values(["Country","Year"]).groupby("Country", as_index=False).tail(1)
    # heuristic unit fix: g/kWh -> kg/kWh
    med = float(df["Intensity"].dropna().median())
    if med > 50:
        df["Intensity"] = df["Intensity"] / 1000.0
    return df.dropna(subset=["Country","Intensity"]).reset_index(drop=True)

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
    return df.sort_values("kgCO2e", ascending=False), summary

# ------------********------------
# ========================= Plotting helpers 
# ------------********------------

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    # Runtime vs N
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs positions N"); plt.xlabel("N (positions)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    # TV distance vs N
    g2 = df.groupby("n")["tv_distance"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["tv_distance"], marker="s")
    plt.title("Performance: Mean TV distance vs N"); plt.xlabel("N (positions)"); plt.ylabel("TV distance")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_tv_vs_N.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    # Runtime vs N (log-log)
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["n"], g["runtime_s"], marker="o")
    plt.title("Scalability: Runtime (log–log)"); plt.xlabel("N (positions)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_N.png"), **plt_kwargs); plt.close()

    # Peak memory vs N
    g2 = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs N"); plt.xlabel("N (positions)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    # Success rate vs N
    g = df.groupby("n")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs N"); plt.xlabel("N (positions)"); plt.ylabel("Success rate (TV<=tol)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_N.png"), **plt_kwargs); plt.close()

    # TV distance distribution by N
    data = [df[df["n"]==k]["tv_distance"].values for k in sorted(df["n"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["n"].unique()))
    plt.title("Reliability: TV distance distribution"); plt.xlabel("N (positions)"); plt.ylabel("TV distance")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_tv_boxplot.png"), **plt_kwargs); plt.close()

def plot_carbon(df: pd.DataFrame, outdir: str):
    # Top 15
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    # Distribution
    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# ------------********------------
# ========================= Main =========================================
# ------------********------------

#This function parses command-line arguments to configure classical 1D random-walk benchmark experiments.
#It sets up output directories for performance, scalability, reliability, and carbon analysis results.

def main():
    p = argparse.ArgumentParser(description="Classical 1D Random Walk (Reflecting) Benchmarks")
    # Keep the same CLI style
    p.add_argument("--sizes", nargs="+", type=int, default=[21], help="Number of positions N (odd recommended).")
    p.add_argument("--steps", type=int, default=1000, help="Number of time steps T.")
    p.add_argument("--paths-per-trial", type=int, default=500, help="Monte Carlo paths per trial (empirical dist).")
    p.add_argument("--trials", type=int, default=10, help="Trials per N (distinct seeds).")
    p.add_argument("--tv-tol", type=float, default=0.05, help="Reliability threshold on TV distance.")
    # carbon/infra (unchanged)
    p.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    p.add_argument("--device-power-watts", type=float, default=65.0)
    p.add_argument("--pue", type=float, default=1.2)
    p.add_argument("--year-select", type=str, default="latest")
    p.add_argument("--outdir", type=str, default="carbon_by_country")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    args = p.parse_args()

    # Output layout
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting classical random-walk experiments...")
    jobs = []
    for N in args.sizes:
        for i in range(args.trials):
            seed = 1000 + 17*N + i
            jobs.append((N, args.steps, args.paths-per_trial if hasattr(args,'paths-per-trial') else args.paths_per_trial, args.tv_tol, seed))

    # Patch for argparse hyphen/underscore attr:
    paths_per_trial = getattr(args, "paths_per_trial", getattr(args, "paths-per-trial", 500))

    rows = []

# This def consume(res: dict) code block collects results from each random-walk experiment 
#    and prints a concise progress summary.
# It runs experiments in parallel when multiple workers are available, 
#    otherwise executes them sequentially.
# It aggregates all trial results into a pandas DataFrame for further analysis and reporting.

    def consume(res: dict):
        rows.append(res)
        print(f"  - N={res['n']} seed={res['seed']} TV={res['tv_distance']:.3e} "
              f"runtime={res['runtime_s']:.4f}s success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, N, args.steps, paths_per_trial, args.tv_tol, seed) for (N,_,__,___,seed) in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for (N,_,__,___,seed) in jobs:
            consume(run_single(N, args.steps, paths_per_trial, args.tv_tol, seed))

    df = pd.DataFrame(rows)

# ------------********------------
    # ------------------ Performance ------------------
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("n").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                mean_tv_distance=("tv_distance","mean"),
                success_rate=("success","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

# ------------********------------
    # ------------------ Scalability ------------------
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["n","runtime_s","peak_mem_mb"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

# ------------********------------
    # ------------------ Reliability ------------------
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["n","seed","tv_distance","success"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("n").agg(
                success_rate=("success","mean"),
                mean_tv=("tv_distance","mean"),
                std_tv=("tv_distance","std"),
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

# ------------********------------
    # ------------------ Carbon (from PERFORMANCE runtime only) ------------------
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
        plot_carbon(carbon_df, carb_dir)
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
