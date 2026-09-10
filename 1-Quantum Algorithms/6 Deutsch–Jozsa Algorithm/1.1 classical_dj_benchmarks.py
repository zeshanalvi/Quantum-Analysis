#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# ----------------------------------------------------------------------------------------------------------------
# Deterministic Classical Baseline for the Deutsch–Jozsa Problem
# 6 DJ Algo
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
My Goal is to evaluate a deterministic classical decision procedure that, given oracle access
to a promised Boolean function f : {0,1}^n -> {0,1} (either CONSTANT or BALANCED), decides which case holds.

I willl simulate:
 - CONSTANT oracles: f(x) ≡ c ∈ {0,1}
 - BALANCED oracles: parity family f_s(x) = (s · x) mod 2 with random non-zero s

Deterministic algorithm (worst-case optimal):
 1) Query f(0...0) → v0.
 2) Enumerate distinct inputs; stop on the first y s.t. f(y) ≠ v0 ⇒ BALANCED.
 3) If no discrepancy after 2^{n-1}+1 queries ⇒ CONSTANT (by the promise).

Benchmarks produced: 1) Performance, 2) Scalability, 3) Reliability, and CO₂ per country using ONLY total PERFORMANCE runtim

Run example that I will use is as the following:
  python classical_dj_benchmarks.py ^
    --sizes 6 8 10 12 ^
    --trials 10 ^
    --excel "Filtered CO2 intensity 236 Countries.xlsx" ^
    --device-power-watts 65 --pue 1.2 ^
    --year-select latest ^
    --outdir carbon_by_country ^
    --combine
"""

 # ------------********------------ Imports

import os, sys, time, argparse, tracemalloc, pathlib, warnings, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

 # ------------********------------
# ========================= Oracle generators (promised DJ)

#Creates an oracle function representing a constant Boolean function where every input x maps to the same output c.
#Returns a closure f(x_bits) so queries always yield c, matching the structure expected by Deutsch–Jozsa–type algorithms.
#Useful for testing quantum oracles without depending on input-dependent logic.

def oracle_constant(c: int):
    """f(x) = c for all x."""
    def f(_x_bits: np.ndarray) -> int:
        return c
    return f

#Defines a balanced oracle of the parity form f(x) = (s · x) mod 2, 
# where the secret bitstring s determines the output.
#Uses bitwise operations to compute the dot product over GF(2), ensuring fast evaluation for arbitrary input bittstrings.
#Returns a callable oracle suitable for Deutsch–Jozsa or Bernstein–Vazirani–style algorithm testing.

def oracle_balanced_parity(s_bits: np.ndarray):
    """Balanced oracle f(x) = (s · x) mod 2 with s != 0 (parity family)."""
    s = s_bits.astype(np.uint8)
    def f(x_bits: np.ndarray) -> int:
        # dot over GF(2) using bitcount
        return int(np.bitwise_and(s, x_bits).sum() & 1)
    return f

 # ------------********------------ Deterministic classical DJ algorithm 

#Implements a deterministic classical strategy for the Deutsch–Jozsa problem 
#  by querying f(0ⁿ) and then scanning inputs in Gray-code order.
#Stops early if it encounters any input whose output differs from the first query, 
#  certifying the oracle as BALANCED.
#If no disagreement is found within the theoretical query bound, 
#  concludes the oracle is CONSTANT and reports total queries used.

def dj_classical_deterministic(f, n: int, query_budget: int | None = None) -> tuple[str, int]:
    """
    Deterministic decision under the DJ promise.
    Returns: (prediction, num_queries_used)
      prediction ∈ {"CONSTANT","BALANCED"}
    """
    # Always query 0^n first
    q = 0
    v0 = f(np.zeros(n, dtype=np.uint8))
    q += 1

    # Upper bound 2^{n-1}+1 queries suffices (worst case: constant)
    max_required = (1 << (n - 1)) + 1
    hard_cap = max_required if query_budget is None else min(query_budget, max_required)

    # Enumerate inputs in Gray code order (nice property: flips one bit at a time)
    # to avoid cache/pathological behaviors; start from 1 (we already used 0)
    for k in range(1, hard_cap):
        y = k ^ (k >> 1)              # Gray code integer
        if y == 0:                    # skip 0 if it appears
            continue
        # convert integer to bit vector (little-endian)
        y_bits = np.fromiter(((y >> i) & 1 for i in range(n)), count=n, dtype=np.uint8)
        if f(y_bits) != v0:
            return "BALANCED", q + 1  # include this query
        q += 1

    return "CONSTANT", q  # certificate by promise

 # ------------********------------ Single experiment instance 

#Constructs either a constant or balanced-parity Deutsch–Jozsa oracle from the given seed and dimension n.
# Runs the deterministic classical DJ procedure while measuring runtime, memory usage, and number of oracle queries.
# Reports correctness of the classification along with performance metrics for benchmarking.

def run_single(n: int, seed: int, oracle_type: str) -> dict:
    """
    Generates an oracle instance (constant or balanced-parity) and runs the
    deterministic DJ algorithm, measuring runtime, memory, and queries used.
    """
    rng = np.random.default_rng(seed)

    if oracle_type == "CONSTANT":
        c = int(rng.integers(0, 2))
        f = oracle_constant(c)
        true_label = "CONSTANT"
    elif oracle_type == "BALANCED":
        # random non-zero s in {0,1}^n
        s = np.zeros(n, dtype=np.uint8)
        while not s.any():
            s = rng.integers(0, 2, size=n, dtype=np.uint8)
        f = oracle_balanced_parity(s)
        true_label = "BALANCED"
    else:
        raise ValueError("oracle_type must be CONSTANT or BALANCED.")

    # Measure performance
    tracemalloc.start()
    t0 = time.perf_counter()
    pred, q_used = dj_classical_deterministic(f, n)
    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    correct = (pred == true_label)

    return {
        "n": n,
        "seed": seed,
        "oracle": true_label,
        "prediction": pred,
        "correct": bool(correct),
        "queries_used": int(q_used),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024 ** 2)),
    }

 # ------------********------------ Carbon accounting (Performance only)

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try Desktop
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

def load_carbon_excel(path: str, year_select: str = "latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    country = next((v for k, v in cols.items() if "country" in k or "nation" in k or k == "location"), None)
    if country is None:
        raise ValueError("Could not find a 'Country' column in the Excel file.")
    year = next((v for k, v in cols.items() if "year" in k or "date" in k), None)
    intensity = next((v for k, v in cols.items()
                      if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)
    if intensity is None:
        # fallback: first numeric column (other than Year if present)
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if year in numeric: numeric.remove(year)
        if not numeric:
            raise ValueError("No numeric carbon intensity column found.")
        intensity = numeric[0]
    keep = [country] + ([year] if year else []) + [intensity]
    df = df[keep].copy()
    df.columns = ["Country", "Year", "Intensity"] if len(keep) == 3 else ["Country", "Intensity"]
    if "Year" in df.columns and year_select.lower() == "latest":
        df = df.sort_values(["Country", "Year"]).groupby("Country", as_index=False).tail(1)

    # Heuristic: if median suggests g/kWh, convert to kg/kWh
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
    return df.sort_values("kgCO2e", ascending=False), summary

 # ------------********------------ Plot helpers =

# function Provides a compatibility wrapper for Matplotlibs boxplot API across versions that renamed labels tick_labels.
#  Attempts the newer keyword first; if unsupported, falls back to the legacy argument.
#    Ensures stable labeling behavior in all environments without modifying calling code.

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)


# ------------********------------ Plot helpers = Performance
def plot_performance(df: pd.DataFrame, outdir: str):
    # Runtime vs n
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs n"); plt.xlabel("n (input bits)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_n.png"), **plt_kwargs); plt.close()

    # Queries vs n
    g2 = df.groupby("n")["queries_used"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["queries_used"], marker="s")
    plt.title("Performance: Mean queries vs n"); plt.xlabel("n (input bits)"); plt.ylabel("Mean queries used")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_queries_vs_n.png"), **plt_kwargs); plt.close()

# ------------********------------ Plot helpers = Scalability
def plot_scalability(df: pd.DataFrame, outdir: str):
    # Runtime (log-log) vs queries
    plt.figure()
    plt.loglog(df["queries_used"], df["runtime_s"], "o", alpha=0.5)
    plt.title("Scalability: Runtime vs Queries (log-log)"); plt.xlabel("Queries used"); plt.ylabel("Runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_queries.png"), **plt_kwargs); plt.close()

    # Peak memory vs n
    g = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs n"); plt.xlabel("n (input bits)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_n.png"), **plt_kwargs); plt.close()

    # (Optional third) Queries distribution by oracle type
    g2 = df.groupby(["n","oracle"])["queries_used"].mean().reset_index()
    for n in sorted(df["n"].unique()):
        sub = g2[g2["n"]==n]
        if len(sub) >= 1:
            plt.figure(); plt.bar(sub["oracle"], sub["queries_used"])
            plt.title(f"Scalability: Mean queries by oracle (n={n})"); plt.xlabel("Oracle type"); plt.ylabel("Mean queries")
            plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, f"scal_queries_by_oracle_n{n}.png"), **plt_kwargs); plt.close()

# ------------********------------ Plot helpers = Reliability
def plot_reliability(df: pd.DataFrame, outdir: str):
    # Success rate vs n
    g = df.groupby("n")["correct"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["correct"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs n"); plt.xlabel("n (input bits)"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_n.png"), **plt_kwargs); plt.close()

    # Boxplot of queries per n
    data = [df[df["n"]==k]["queries_used"].values for k in sorted(df["n"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["n"].unique()))
    plt.title("Reliability: Queries distribution by n"); plt.xlabel("n (input bits)"); plt.ylabel("Queries used")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_queries_boxplot.png"), **plt_kwargs); plt.close()

    # Error bars: mean±std queries vs n
    g2 = df.groupby("n")["queries_used"].agg(["mean","std"]).reset_index()
    plt.figure(); plt.errorbar(g2["n"], g2["mean"], yerr=g2["std"], fmt="-s")
    plt.title("Reliability: Mean±Std of queries vs n"); plt.xlabel("n (input bits)"); plt.ylabel("Queries used")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_queries_mean_std.png"), **plt_kwargs); plt.close()


# ------------********------------ Plot helpers = CO2
def plot_carbon(df: pd.DataFrame, outdir: str):
    # Top 15 kgCO2e
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    # Bottom 15 kgCO2e
    bot = df.nsmallest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(bot["Country"][::-1], bot["kgCO2e"][::-1])
    plt.title("Carbon: Bottom 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_bottom15.png"), **plt_kwargs); plt.close()

    # Distribution plotter
    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

    # CDF
    xs = np.sort(df["kgCO2e"].values); ys = np.arange(1, len(xs)+1)/len(xs)
    plt.figure(); plt.plot(xs, ys)
    plt.title("Carbon: CDF of emissions"); plt.xlabel("kg CO2e"); plt.ylabel("CDF")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "carbon_cdf.png"), **plt_kwargs); plt.close()

 # ------------********------------ Main Function =========================================
#This function sets up command-line arguments to control experiment sizes, trial counts, system parameters, and output paths.
#It creates necessary directories for storing results related to performance, scalability, reliability, and carbon impact.
#It prepares deterministic experiment jobs by iterating over input sizes and trials for both constant and balanced oracle cases.

def main():
    parser = argparse.ArgumentParser(description="Deterministic Classical Deutsch–Jozsa Benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=[6, 8, 10, 12],
                        help="Input sizes n (bits). Keep small so laptop runs fast.")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per (n, oracle_type). Half constant, half balanced.")
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx",
                        help="Excel with country carbon intensities; placed on Desktop or absolute path.")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2),
                        help="Parallel processes for different seeds.")
    args = parser.parse_args()

    # Output folders
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting deterministic classical DJ experiments...")
    rows = []
    jobs = []
    # Split trials across oracle types (balanced/constant)
    for n in args.sizes:
        for i in range(args.trials):
            jobs.append((n, 1000 + 17*n + i, "CONSTANT"))
            jobs.append((n, 2000 + 17*n + i, "BALANCED"))


#This block of code defines a helper function to collect results, append them to a list, 
#  and print a formatted summary for each experiment run.
#It executes jobs either in parallel using a process pool or sequentially, 
#   depending on the number of available workers.
#It aggregates all collected results into a pandas DataFrame for further analysis or storage.

    def _consume(res: dict):
        rows.append(res)
        print(f"  - n={res['n']}, oracle={res['oracle']}, seed={res['seed']}, "
              f"queries={res['queries_used']}, runtime={res['runtime_s']:.6f}s, "
              f"correct={int(res['correct'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, n, seed, typ) for (n, seed, typ) in jobs]
            for f in as_completed(futs):
                _consume(f.result())
    else:
        for (n, seed, typ) in jobs:
            _consume(run_single(n, seed, typ))

    df = pd.DataFrame(rows)

 # ------------********------------
    # ------------------ Performance ------------------

    perf_path = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("n").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_queries=("queries_used","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean")
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

 # ------------********------------
    # ------------------ Scalability ------------------

    scal_path = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_path, engine="openpyxl") as w:
            df[["n","oracle","queries_used","runtime_s","peak_mem_mb"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby(["n","oracle"]).mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

 # ------------********------------
    # ------------------ Reliability ------------------

    rel_path = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_path, engine="openpyxl") as w:
            df[["n","seed","oracle","prediction","correct","queries_used"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("n").agg(
                success_rate=("correct","mean"),
                mean_queries=("queries_used","mean"),
                std_queries=("queries_used","std")
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

 # ------------********------------
    # ------------------ Carbon (Performance only) ------------------

    excel_path = resolve_excel_path(args.excel)
    try:
        intensity_df = load_carbon_excel(excel_path, year_select=args.year_select)
        carbon_df, summary_df = compute_carbon(df, intensity_df,
                                               power_watts=args.device_power_watts,
                                               pue=args.pue)
        carb_path = os.path.join(carb_dir, "carbon_results.xlsx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.ExcelWriter(carb_path, engine="openpyxl") as w:
                carbon_df.to_excel(w, index=False, sheet_name="per_country")
                summary_df.to_excel(w, index=False, sheet_name="summary")
                intensity_df.to_excel(w, index=False, sheet_name="intensity_input")
        plot_carbon(carbon_df, carb_dir)
        print(f"[carbon] Wrote Excel and plots to: {carb_dir}")
    except Exception as e:
        print(f"[carbon] ERROR reading Excel '{excel_path}': {e}")
        print("[carbon] Skipping carbon benchmark. Re-run once the file is available.")

# ------------********------------
                                 # ------------------ Summary ------------------
# ------------********------------

#This section prints a console summary showing the locations of generated Excel files and created output directories.
#It provides a clear overview of where performance, scalability, reliability, and carbon results are stored.


    print("\n=== Summary (console) ===")
    print("Performance Excel:", perf_path)
    print("Scalability Excel:", scal_path)
    print("Reliability Excel:", rel_path)
    print("Carbon Excel:", os.path.join(carb_dir, "carbon_results.xlsx"))
    print("Folders created:\n -", perf_dir, "\n -", scal_dir, "\n -", rel_dir, "\n -", carb_dir)

#It ensures the main function runs only when the script is executed directly.

if __name__ == "__main__":
    main()
