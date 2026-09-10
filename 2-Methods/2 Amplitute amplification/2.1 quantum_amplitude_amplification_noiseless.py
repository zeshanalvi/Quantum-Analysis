#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Amplitude Amplification
# Quantum Amplitude Amplification (Grover)

"""
Quantum Amplitude Amplification (Grover) — Ideal/Noiseless Benchmarks

Problem:
  - Unstructured search over N items with a hidden marked set S of size M.
  - Oracle marks S with a phase flip (|i> -> -|i> for i in S).
  - Start from uniform superposition; apply k Grover iterations G = D * O, where
        k = round((pi/4) * sqrt(N / M))    (for known M)
  - Measure one index; success if it's in S.

Per (N, seed) batch of --steps independent instances:
  - success_rate (fraction of instances that returned a marked index)
  - mean_queries (≈ k, one oracle phase per Grover iteration)
  - runtime_s, peak_mem_mb
  - success (batch passes) = success_rate >= --acc-tol

Outputs (folders):
  Performance/  Scalability/  Reliability/
  Carbon footprints/<outdir>/
"""
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
#These follwing imports provide core utilities for filesystem access, timing, argument parsing, and memory tracking.
#NumPy and pandas support numerical computation and tabular data handling for experiments.
#Matplotlib is configured for headless plotting suitable for batch or server environments.
#Concurrent futures enable parallel execution and coordinated collection of worker results.

import os, time, argparse, warnings, pathlib, tracemalloc, math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------
# --------------------------- Helpers for Grover ---------------------------
# ------------********------------

#This function constructs a boolean mask that marks a fixed number of items in a search space.
#It validates inputs, randomly selects unique indices, and encodes them as marked positions.

def make_marked_mask(N: int, M: int, rng: np.random.Generator) -> np.ndarray:
    if M < 0 or M > N:
        raise ValueError(f"Invalid M={M} for N={N}")
    mask = np.zeros(N, dtype=bool)
    if M > 0:
        idx = rng.choice(N, size=M, replace=False)
        mask[idx] = True
    return mask

#This def grover_iterations_k helper computes the recommended number of Grover iterations 
#     based on problem size and marked items.
#It follows the standard theoretical scaling and enforces a minimum iteration count for valid cases.
def grover_iterations_k(N: int, M: int) -> int:
    if M <= 0:
        return 0
    k = int(round((math.pi/4.0) * math.sqrt(N / M)))
    return max(1, k)

def phase_oracle_inplace(state: np.ndarray, mask: np.ndarray):
    """Phase flip on marked indices (ideal, noiseless)."""
    state[mask] *= -1.0

def diffusion_inplace(state: np.ndarray):
    # Diffusion (reflection about |s>): D|ψ> = 2|s><s|ψ> - |ψ|.
    # For uniform |s>, this is equivalent to: state <- 2*avg - state.
    
    avg = state.mean()
    # vector of ones implicit; in-place reflection:
    state[:] = 2.0 * avg - state

#This function performs a single Grover iteration on the quantum state.
#It applies the phase oracle followed by the diffusion operator to amplify marked amplitudes.
def grover_cycle(state: np.ndarray, mask: np.ndarray):
    """One Grover iteration: phase oracle then diffusion."""
    phase_oracle_inplace(state, mask)
    diffusion_inplace(state)

def run_instance(N: int, M: int, rng: np.random.Generator):
    # One quantum search instance:
    #  - Create a fresh marked set S
    #  - Prepare uniform |s>
    #  - Apply k Grover iterations
    #  - Measure once; return (success, queries_used, k)
   
    mask = make_marked_mask(N, M, rng)
    k = grover_iterations_k(N, M)
    if M == 0 or k == 0:
        return False, 0, 0

    # Uniform superposition
    state = np.ones(N, dtype=np.complex128) / math.sqrt(N)

    # k Grover iterations
    for _ in range(k):
        grover_cycle(state, mask)

    # Measurement
    probs = (state.real**2 + state.imag**2)
    s = probs.sum()
    if s <= 0.0:
        # numerical corner (shouldn't happen), fall back to uniform
        probs = np.ones(N)/N
    else:
        probs = probs / s

    i = rng.choice(N, p=probs)
    success = bool(mask[i])
    queries_used = k  # one phase oracle per iteration
    return success, queries_used, k

# ------------********------------
# --------------------------- Batch runner (one row per (N, seed)) ---------------------------
# ------------********------------

#This routine runs multiple Grover-style search trials for fixed problem parameters and a random seed.
#    It tracks per-instance success, query counts, and iteration budgets while measuring time and memory usage.
#    Aggregate success rates and averages are computed to summarize algorithm performance.
#    The function returns a structured dictionary of metrics for later analysis and reporting.

def run_single(N: int, steps: int, acc_tol: float, seed: int, M: int) -> dict:
    rng = np.random.default_rng(seed)

    tracemalloc.start()
    t0 = time.perf_counter()

    successes = 0
    queries_total = 0
    budgets = []  # the k used in each instance (constant for given N,M)

    for _ in range(steps):
        ok, q, k = run_instance(N, M, rng)
        successes += int(ok)
        queries_total += q
        budgets.append(k)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    success_rate = successes / max(1, steps)
    mean_q = queries_total / max(1, steps)
    mean_k = float(np.mean(budgets)) if budgets else 0.0

    return {
        "N": int(N),
        "steps": int(steps),
        "seed": int(seed),
        "M": int(M),
        "success_rate": float(success_rate),
        "mean_queries": float(mean_q),
        "mean_grover_k": mean_k,
        "success": bool(success_rate >= acc_tol),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "instances_processed": int(steps),
    }

# ------------********------------
# --------------------------- Carbon accounting (Performance only) 
# ------------********------------

#These following functions locate and load an Excel file containing country-level carbon intensity data.
#   The path resolver flexibly checks absolute paths and common Desktop locations for user convenience.
#   The loader infers relevant columns by name and data type, standardizes them, and optionally selects the latest year.
#   It also normalizes units when needed and cleans the dataset by removing invalid or missing entries.

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

# ------------********------------
# --------------------------- Plotting helpers ---------------------------
# ------------********------------

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)  # Matplotlib ≥3.9
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["runtime_s"], marker="o")
    plt.title("Quantum AA Performance: Runtime vs N"); plt.xlabel("N"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["mean_queries"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["mean_queries"], marker="s")
    plt.title("Quantum AA Performance: Mean oracle queries vs N"); plt.xlabel("N"); plt.ylabel("Mean queries (≈k)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_queries_vs_N.png"), **plt_kwargs); plt.close()

# ------------********------------
def plot_scalability(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["N"], g["runtime_s"], "o")
    plt.title("Quantum AA Scalability: Runtime (log–log) vs N"); plt.xlabel("N"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["peak_mem_mb"], marker="^")
    plt.title("Quantum AA Scalability: Peak memory vs N"); plt.xlabel("N"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

# ------------********------------
def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["success_rate"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["success_rate"], marker="o"); plt.ylim(0,1.05)
    plt.title("Quantum AA Reliability: Success rate vs N"); plt.xlabel("N"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_N.png"), **plt_kwargs); plt.close()

    data = [df[df["N"]==k]["success_rate"].values for k in sorted(df["N"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["N"].unique()))
    plt.title("Quantum AA Reliability: Success-rate distribution by N"); plt.xlabel("N"); plt.ylabel("Success rate")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_boxplot.png"), **plt_kwargs); plt.close()

# ------------********------------
# --------------------------- Main ---------------------------
# ------------********------------

def main():
    parser = argparse.ArgumentParser(description="Quantum Amplitude Amplification (Grover) — Ideal/Noiseless Benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128, 256],
                        help="Search space sizes N (powers of two convenient but not required).")
    parser.add_argument("--steps", type=int, default=200, help="Instances per (N, seed) batch.")
    parser.add_argument("--trials", type=int, default=10, help="Seeds per N.")
    parser.add_argument("--marked-count", type=int, default=1, help="Number of marked items M per instance.")
    parser.add_argument("--acc-tol", type=float, default=0.9, help="Batch success if success_rate >= acc-tol.")
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
    cwd = os.getcwd()
    perf_dir = os.path.join(cwd, "Performance")
    scal_dir = os.path.join(cwd, "Scalability")
    rel_dir  = os.path.join(cwd, "Reliability")
    carb_root = os.path.join(cwd, "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting Quantum Amplitude Amplification (ideal, noiseless) benchmarks...")
    rows, jobs = [], []
    for N in args.sizes:
        for i in range(args.trials):
            seed = 1000 + 23*int(N) + i
            jobs.append((N, args.steps, args.acc_tol, seed, args.marked_count))

    def consume(res: dict):
        rows.append(res)
        print(f"  - N={res['N']} seed={res['seed']} "
              f"succ_rate={res['success_rate']:.3f} mean_q~{res['mean_queries']:.2f} "
              f"k~{res['mean_grover_k']:.2f} runtime={res['runtime_s']:.4f}s "
              f"success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, *job) for job in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for job in jobs:
            consume(run_single(*job))

    df = pd.DataFrame(rows)

    # ----- Performance (Excel + plots) -----
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("N").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                success_rate=("success_rate","mean"),
                mean_queries=("mean_queries","mean"),
                mean_grover_k=("mean_grover_k","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

    # ----- Scalability (Excel + plots) -----
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["N","runtime_s","peak_mem_mb","mean_queries"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("N").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

    # ----- Reliability (Excel + plots) -----
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["N","seed","success_rate","mean_queries","mean_grover_k","success"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("N").agg(
                success_rate=("success_rate","mean"),
                std_success=("success_rate","std"),
                mean_queries=("mean_queries","mean"),
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

# ------------********------------ Carbon plots
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
