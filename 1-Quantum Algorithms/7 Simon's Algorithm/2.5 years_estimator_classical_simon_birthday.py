#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Probabilistic Algorithm for Simon's Problem (Birthday Paradox)
with n>=128 early-exit estimator (prints years and exits).

Outputs (for n < 128):
  Performance/, Scalability/, Reliability/, Carbon footprints/<outdir>/
"""

# ------------********------------ Imports
#This code section imports standard libraries for system access, timing, memory tracking, and argument parsing.
#It sets up numerical computing, data analysis, and plotting libraries for batch execution.
#It also imports concurrency utilities to enable parallel experiment execution.

import os, sys, time, math, argparse, warnings, pathlib, tracemalloc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========================= Strict Simon oracle (no spurious collisions) =========================
#def simon_oracle_factory () defines a strict Simon oracle that enforces f(x) = f(x ⊕ s) for a hidden string s.
#It maintains explicit mappings to ensure only intended input pairs share outputs.
#It prevents accidental collisions by tracking and reusing outputs only within valid pairs.

def simon_oracle_factory(n: int, s_bits: np.ndarray):
    """
    Strict Simon oracle: f(x) = f(x ⊕ s) and distinct pairs map to distinct outputs.
    Prevents accidental 'collisions' between unrelated pairs.
    """
    rng = np.random.default_rng(12345)
    mapping = {}          # x -> output
    pair_output = {}      # canonical pair rep -> output
    used_outputs = set()  # avoid reusing outputs across pairs

#This def canonical_rep () is a helper function 
#It computes a canonical representation for an input under XOR with the hidden string.
#It selects a consistent ordering between x and x ⊕ s to identify equivalent pairs.
#It enables stable mapping of Simon oracle input pairs to shared outputs.

    def canonical_rep(x):
        x_t = tuple(x)
        y_t = tuple(np.bitwise_xor(x, s_bits))
        return x_t if x_t < y_t else y_t

#The def fresh_output() function generates a new, unused output value for the Simon oracle.
#It repeatedly samples random bit strings until a unique output is found.
#It tracks previously used outputs to prevent unintended collisions.

    def fresh_output():
        while True:
            y = tuple(rng.integers(0, 2, size=n, dtype=np.uint8))
            if y not in used_outputs:
                used_outputs.add(y)
                return y

# def f(x_bits: np.ndarray) function evaluates the Simon oracle 
#     by returning consistent outputs for paired inputs.
#It assigns a shared output to each x and x ⊕ s pair while 
#     caching results for efficiency.
#It enforces the two-to-one promise required by Simon’s problem.

    def f(x_bits: np.ndarray):
        key = tuple(x_bits)
        if key in mapping:
            return mapping[key]
        rep = canonical_rep(x_bits)
        if rep in pair_output:
            out = pair_output[rep]
        else:
            out = fresh_output()
            pair_output[rep] = out
        mapping[key] = out
        # also set partner explicitly for the 2-to-1 promise
        partner = tuple(np.bitwise_xor(x_bits, s_bits))
        mapping[partner] = out
        return out

    return f

# ========================= Classical birthday-paradox solver =========================

def classical_simon_birthday(f, n: int, max_trials: int | None = None, seed: int | None = None):
    rng = np.random.default_rng(seed)
    if max_trials is None:
        max_trials = 2 ** (n // 2 + 2)  # generous cap

    table = {}
    queries = 0
    for _ in range(max_trials):
        x = rng.integers(0, 2, size=n, dtype=np.uint8)
        fx = tuple(f(x))
        queries += 1
        if fx in table:
            s_guess = np.bitwise_xor(x, table[fx])
            return s_guess, queries, True
        table[fx] = x
    return np.zeros(n, dtype=np.uint8), queries, False

# ========================= One experiment run =========================

#This def run_single(n: int, seed: int) -> dict:  function executes a single classical Simon experiment 
# with a randomly generated nonzero secret string.
#It measures runtime and peak memory usage while attempting to recover 
#  the hidden string via the birthday paradox method.
#It returns a dictionary containing performance metrics and a success flag based on correctness.

def run_single(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # nonzero secret
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

# ========================= Carbon utilities (Performance only) =========================

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
    if med > 50:  # likely g/kWh
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

#These above three utility functions resolve the Excel file path, 
#  load and normalize carbon intensity data, 
#  and compute carbon emissions.
#They handle flexible column naming, year selection, unit normalization, and missing data robustly.
#They calculate total energy use and emissions, returning per-country results and a summary table.

# ========================= Plotting helpers =========================

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs n"); plt.xlabel("n"); plt.ylabel("Runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_n.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["queries_used"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["queries_used"], marker="s")
    plt.title("Performance: Mean Queries vs n"); plt.xlabel("n"); plt.ylabel("Queries")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_queries_vs_n.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    plt.figure(); plt.loglog(df["queries_used"], df["runtime_s"], "o")
    plt.title("Scalability: Runtime vs Queries (log-log)")
    plt.xlabel("Queries"); plt.ylabel("Runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_runtime_vs_queries.png"), **plt_kwargs); plt.close()

    g = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak Memory vs n"); plt.xlabel("n"); plt.ylabel("Peak Memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_n.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs n"); plt.xlabel("n"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_n.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["queries_used"].agg(["mean","std"]).reset_index()
    plt.figure(); plt.errorbar(g2["n"], g2["mean"], yerr=g2["std"], fmt="-s")
    plt.title("Reliability: Mean±Std Queries vs n"); plt.xlabel("n"); plt.ylabel("Queries used")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_queries_mean_std.png"), **plt_kwargs); plt.close()

# ========================= n>=128 Early-exit estimator =========================

#This def estimate_years_for_n128 () function estimates the number of years required to solve Simon’s problem at n=128 classically.
#It calibrates query throughput on a small instance and extrapolates using the birthday paradox expectation.
#It reports calibration details and returns the projected runtime in years.

def estimate_years_for_n128(calib_trials: int = 6, calib_n: int = 12, seed0: int = 4242) -> float:
    """
    Calibrate queries/second on a tiny instance, then estimate runtime
    for n=128 using E[queries] ≈ sqrt(pi/2) * 2^{(n-1)/2}. Return years.
    """
    # Calibrate throughput on small n (fast)
    total_q = 0
    total_t = 0.0
    for t in range(calib_trials):
        res = run_single(calib_n, seed0 + t)
        total_q += int(res["queries_used"])
        total_t += float(res["runtime_s"])
    qps = total_q / max(total_t, 1e-12)          # queries per second

    # Expected queries for Simon collision (birthday paradox on range size 2^{n-1})
    c = math.sqrt(math.pi / 2.0)                 # ≈ 1.253314137
    expected_queries = c * (2 ** ((128 - 1) / 2))  # 2^{63.5} * c

    seconds = expected_queries / max(qps, 1e-12)
    years = seconds / (365.25 * 24 * 3600)

    print(f"[estimator] Calibration on n={calib_n}: total_queries={total_q}, total_time={total_t:.6f}s, "
          f"throughput≈{qps:,.2f} q/s")
    print(f"[estimator] Expected queries for n=128 ≈ {expected_queries:,.3e}")
    return years

# ========================= Main =====================================

def main():
    parser = argparse.ArgumentParser(description="Classical Simon (Birthday Paradox) with n>=128 early-exit")
    parser.add_argument("--sizes", nargs="+", type=int, default=[6, 8, 10], help="Input sizes n (bits)")
    parser.add_argument("--trials", type=int, default=10, help="Trials per size")
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    args = parser.parse_args()

    # If any requested size is >= 128, estimate and exit
    if any(n >= 128 for n in args.sizes):
        years = estimate_years_for_n128()
        print(f"\nIt will take ≈ {years:,.2f} years to solve this problem (n=128) on this machine. Exiting.")
        sys.exit(0)

    # Normal benchmark path for n < 128
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

    # Performance
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="raw")
        df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    # Plots
    def plot_performance(df, outdir):
        g = df.groupby("n")["runtime_s"].mean().reset_index()
        plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o"); plt.title("Performance: Runtime vs n")
        plt.xlabel("n"); plt.ylabel("Runtime (s)"); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, "perf_runtime_vs_n.png"), dpi=140, bbox_inches="tight"); plt.close()
        g2 = df.groupby("n")["queries_used"].mean().reset_index()
        plt.figure(); plt.plot(g2["n"], g2["queries_used"], marker="s"); plt.title("Performance: Mean Queries vs n")
        plt.xlabel("n"); plt.ylabel("Queries"); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, "perf_queries_vs_n.png"), dpi=140, bbox_inches="tight"); plt.close()
    plot_performance(df, perf_dir)

    # Scalability
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="raw")
        df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    def plot_scalability(df, outdir):
        plt.figure(); plt.loglog(df["queries_used"], df["runtime_s"], "o"); plt.title("Scalability: Runtime vs Queries (log-log)")
        plt.xlabel("Queries"); plt.ylabel("Runtime (s)"); plt.grid(True, which="both", alpha=0.3)
        plt.savefig(os.path.join(outdir, "scal_runtime_vs_queries.png"), dpi=140, bbox_inches="tight"); plt.close()
        g = df.groupby("n")["peak_mem_mb"].mean().reset_index()
        plt.figure(); plt.plot(g["n"], g["peak_mem_mb"], marker="^"); plt.title("Scalability: Peak Memory vs n")
        plt.xlabel("n"); plt.ylabel("Peak Memory (MB)"); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, "scal_peakmem_vs_n.png"), dpi=140, bbox_inches="tight"); plt.close()
    plot_scalability(df, scal_dir)

    # Reliability
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="raw")
        df.groupby("n").agg(success_rate=("success","mean"),
                            mean_queries=("queries_used","mean"),
                            std_queries=("queries_used","std")).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    def plot_reliability(df, outdir):
        g = df.groupby("n")["success"].mean().reset_index()
        plt.figure(); plt.plot(g["n"], g["success"], marker="o"); plt.ylim(0,1.05)
        plt.title("Reliability: Success rate vs n"); plt.xlabel("n"); plt.ylabel("Success rate")
        plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_n.png"), dpi=140, bbox_inches="tight"); plt.close()
        g2 = df.groupby("n")["queries_used"].agg(["mean","std"]).reset_index()
        plt.figure(); plt.errorbar(g2["n"], g2["mean"], yerr=g2["std"], fmt="-s")
        plt.title("Reliability: Mean±Std Queries vs n"); plt.xlabel("n"); plt.ylabel("Queries used")
        plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_queries_mean_std.png"), dpi=140, bbox_inches="tight"); plt.close()
    plot_reliability(df, rel_dir)

    # Carbon (Performance only)
    try:
        intensity_df = load_carbon_excel(resolve_excel_path(args.excel), args.year_select)
        carbon_df, summary_df = compute_carbon(df, intensity_df, args.device_power_watts, args.pue)
        carb_dir = os.path.join(os.getcwd(), "Carbon footprints", args.outdir)
        os.makedirs(carb_dir, exist_ok=True)
        carb_xlsx = os.path.join(carb_dir, "carbon_results.xlsx")
        with pd.ExcelWriter(carb_xlsx, engine="openpyxl") as w:
            carbon_df.to_excel(w, index=False, sheet_name="per_country")
            summary_df.to_excel(w, index=False, sheet_name="summary")
        # simple plots
        top = carbon_df.nlargest(15, "kgCO2e"); plt.figure(figsize=(8,5))
        plt.barh(top["Country"][::-1], top["kgCO2e"][::-1]); plt.title("Carbon: Top 15 Countries (kgCO2e)")
        plt.xlabel("kg CO2e"); plt.tight_layout(); plt.savefig(os.path.join(carb_dir, "carbon_top15.png"), dpi=140, bbox_inches="tight"); plt.close()
        plt.figure(); plt.hist(carbon_df["kgCO2e"], bins=30); plt.title("Carbon: Distribution")
        plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3); plt.savefig(os.path.join(carb_dir, "carbon_distribution.png"), dpi=140, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"[carbon] ERROR: {e}")

    print("\n=== Summary ===")
    print("Performance Excel:", perf_xlsx)
    print("Scalability Excel:", scal_xlsx)
    print("Reliability Excel:", rel_xlsx)
    print("Carbon Excel:", os.path.join(os.getcwd(), "Carbon footprints", args.outdir, 'carbon_results.xlsx'))

if __name__ == "__main__":
    main()
