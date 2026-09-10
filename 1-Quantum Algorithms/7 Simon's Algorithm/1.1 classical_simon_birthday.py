#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Classical Probabilistic Algorithm for Simon's Problem (Birthday Paradox Method)
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

import os, sys, time, math, argparse, warnings, pathlib, tracemalloc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------ Simon oracle generator

#This function constructs a Simon problem oracle with a hidden XOR mask s.
#It ensures the function outputs are identical for inputs differing by the hidden string.
#It uses a deterministic random mapping to generate consistent oracle behavior.

def simon_oracle_factory(n: int, s_bits: np.ndarray):
    """Build a function f(x) with hidden string s satisfying f(x)=f(y) iff y=x⊕s."""
    rng = np.random.default_rng(12345)
    mapping = {}

    def f(x_bits: np.ndarray):
        key = tuple(x_bits)
        if key in mapping:
            return mapping[key]
        x_xor_s = tuple(np.bitwise_xor(x_bits, s_bits))
        if x_xor_s in mapping:
            mapping[key] = mapping[x_xor_s]
        else:
            mapping[key] = tuple(rng.integers(0, 2, size=n, dtype=np.uint8))
        return mapping[key]

    return f


# ------------********------------ Classical probabilistic Simon solver

#This function implements a classical probabilistic solution to 
#  Simon’s problem using the birthday paradox.
#It randomly queries the oracle and looks for output collisions to infer the hidden XOR string.
#It returns the guessed string, number of queries used, and a 
#   success flag indicating whether a collision was found.

def classical_simon_birthday(f, n: int, max_trials: int = None, seed: int = None):
    """
    Classical probabilistic approach using the birthday paradox:
      1. Randomly sample x ∈ {0,1}ⁿ, record f(x)
      2. If f(x_i) = f(x_j) for any i≠j, return s = x_i ⊕ x_j
    """
    rng = np.random.default_rng(seed)
    if max_trials is None:
        max_trials = 2 ** (n // 2 + 2)

    table = {}
    queries = 0
    for _ in range(max_trials):
        x = rng.integers(0, 2, size=n, dtype=np.uint8)
        fx = tuple(f(x))
        queries += 1
        if fx in table:
            s_guess = np.bitwise_xor(x, table[fx])
            return s_guess, queries, True  # success
        table[fx] = x
    return np.zeros(n, dtype=np.uint8), queries, False  # failed (no collision)


# ------------********------------ One experiment =

#This function runs a single classical Simon experiment with a randomly generated nonzero secret.
#It measures runtime and peak memory usage while attempting to recover the hidden string.
#It returns performance metrics along with a success indicator based on correctness.

def run_single(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # Generate nonzero secret
    s = np.zeros(n, dtype=np.uint8)
    while not s.any():
        s = rng.integers(0, 2, size=n, dtype=np.uint8)

    f = simon_oracle_factory(n, s)

    tracemalloc.start()
    t0 = time.perf_counter()
    s_guess, queries, success = classical_simon_birthday(f, n, seed=seed)
    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    correct = success and np.array_equal(s_guess, s)

    return {
        "n": n,
        "seed": seed,
        "queries_used": queries,
        "runtime_s": runtime,
        "peak_mem_mb": peak / (1024**2),
        "success": bool(correct)
    }

# ------------********------------ Carbon footprint utilities 

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
    year = next((v for k,v in cols.items() if "year" in k or "date" in k), None)
    intensity = next((v for k,v in cols.items()
                      if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)
    if intensity is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if year in numeric: numeric.remove(year)
        if not numeric: raise ValueError("No numeric column found.")
        intensity = numeric[0]
    keep = [country] + ([year] if year else []) + [intensity]
    df = df[keep].copy()
    df.columns = ["Country","Year","Intensity"] if len(keep)==3 else ["Country","Intensity"]
    if "Year" in df.columns and year_select.lower()=="latest":
        df = df.sort_values(["Country","Year"]).groupby("Country", as_index=False).tail(1)
    med = float(df["Intensity"].dropna().median())
    if med > 50:
        df["Intensity"] /= 1000.0
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

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs n"); plt.xlabel("n"); plt.ylabel("Runtime (s)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "perf_runtime_vs_n.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["queries_used"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["queries_used"], marker="s")
    plt.title("Performance: Mean Queries vs n"); plt.xlabel("n"); plt.ylabel("Queries")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "perf_queries_vs_n.png"), **plt_kwargs); plt.close()

# ------------********------------

def plot_scalability(df: pd.DataFrame, outdir: str):
    plt.figure(); plt.loglog(df["queries_used"], df["runtime_s"], "o")
    plt.title("Scalability: Runtime vs Queries (log-log)")
    plt.xlabel("Queries"); plt.ylabel("Runtime (s)")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(outdir, "scal_runtime_vs_queries.png"), **plt_kwargs); plt.close()

    g = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak Memory vs n"); plt.xlabel("n"); plt.ylabel("Peak Memory (MB)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "scal_peakmem_vs_n.png"), **plt_kwargs); plt.close()

# ------------********------------

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs n"); plt.xlabel("n"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "rel_success_vs_n.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["queries_used"].agg(["mean","std"]).reset_index()
    plt.figure(); plt.errorbar(g2["n"], g2["mean"], yerr=g2["std"], fmt="-s")
    plt.title("Reliability: Mean±Std Queries vs n"); plt.xlabel("n"); plt.ylabel("Queries used")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "rel_queries_mean_std.png"), **plt_kwargs); plt.close()

# ------------********------------

def plot_carbon(df: pd.DataFrame, outdir: str):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 Countries"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# ------------********------------Main Function

#This function parses command-line arguments to configure classical Simon experiments using the birthday paradox method.
#It creates output directories for performance, scalability, reliability, and carbon analysis results.
#It prepares a list of experiment jobs across input sizes and trials for execution.

def main():
    parser = argparse.ArgumentParser(description="Classical Simon's Problem via Birthday Paradox (Probabilistic)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[6, 8, 10],
                        help="Input sizes n (bits)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per size")
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    args = parser.parse_args()

    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting classical Simon (birthday paradox) experiments...")
    jobs = [(n, 1000 + 17*n + i) for n in args.sizes for i in range(args.trials)]
    results = []


#This block collects and logs results from each Simon experiment run.
#It executes jobs in parallel when multiple workers are available,
#    otherwise running them sequentially.
#It aggregates all experiment outcomes into a pandas DataFrame for analysis and export.

    def consume(res):
        results.append(res)
        print(f"  - n={res['n']} seed={res['seed']} queries={res['queries_used']} "
              f"runtime={res['runtime_s']:.6f}s success={int(res['success'])}")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, n, seed) for (n, seed) in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for (n, seed) in jobs:
            consume(run_single(n, seed))

    df = pd.DataFrame(results)

    # ------------********------------ Performance
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="raw")
        df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

    # ------------********------------ Scalability
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="raw")
        df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

    # ------------********------------ Reliability
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="raw")
        df.groupby("n").agg(success_rate=("success","mean"),
                            mean_queries=("queries_used","mean"),
                            std_queries=("queries_used","std")).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

    # ------------********------------ Carbon (Performance only)
    excel_path = resolve_excel_path(args.excel)
    try:
        intensity_df = load_carbon_excel(excel_path, args.year_select)
        carbon_df, summary_df = compute_carbon(df, intensity_df,
                                               args.device_power_watts, args.pue)
        carb_xlsx = os.path.join(carb_dir, "carbon_results.xlsx")
        with pd.ExcelWriter(carb_xlsx, engine="openpyxl") as w:
            carbon_df.to_excel(w, index=False, sheet_name="per_country")
            summary_df.to_excel(w, index=False, sheet_name="summary")
        plot_carbon(carbon_df, carb_dir)
    except Exception as e:
        print(f"[carbon] ERROR: {e}")

    print("\n=== Summary ===")
    print("Performance Excel:", perf_xlsx)
    print("Scalability Excel:", scal_xlsx)
    print("Reliability Excel:", rel_xlsx)
    print("Carbon Excel:", os.path.join(carb_dir, 'carbon_results.xlsx'))

if __name__ == "__main__":
    main()
