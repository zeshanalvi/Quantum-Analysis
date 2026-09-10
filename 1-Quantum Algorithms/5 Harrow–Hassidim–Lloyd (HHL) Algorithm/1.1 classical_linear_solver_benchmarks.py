

# Classical Linear Equation Solvers (Baseline for HHL Algorithm)

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
To run, I used the following baselines:
  python classical_linear_solver_benchmarks.py ^
    --sizes 200 400 800 1200 ^
    --trials 3 ^
    --excel "Filtered CO2 intensity 236 Countries.xlsx" ^
    --device-power-watts 65 --pue 1.2 ^
    --year-select latest ^
    --outdir carbon_by_country ^
    --combine
"""
# ------------********------------ Imports

import os, sys, time, argparse, tracemalloc, pathlib, warnings   # Import standard library modules for system operations, CLI parsing, memory tracing, paths, and warnings
import numpy as np                                               # Import NumPy for numerical computations
import pandas as pd                                              # Import Pandas for data manipulation and analysis
import matplotlib                                                 # Import Matplotlib (base)
matplotlib.use("Agg")                                            # Use a non-GUI backend suitable for scripts and servers
import matplotlib.pyplot as plt                                   # Import plotting functions from Matplotlib
from concurrent.futures import ProcessPoolExecutor, as_completed  # Import tools for parallel processing using multiple processes


# ------------********------------ Solve Linear System ---------------------

def solve_linear_system(A, b, method="numpy"):
    if method == "numpy":
        x = np.linalg.solve(A, b)       # Solve Ax=b using NumPy's optimized linear algebra solver
    elif method == "gauss":
        x = gaussian_elimination(A, b)  # Use custom Gaussian elimination implementation
    else:
        raise ValueError("Unknown solver")
    return x


def gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    for i in range(n):# Pivot: swap with the row having the largest absolute value in column i
        # Pivot
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        # Eliminate: zero out entries below the pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    # Back substitution: solve from bottom to top
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

# ------------********------------ Experiment Runner ---------------------

def run_experiment(N, seed, method="numpy"):
    rng = np.random.default_rng(seed)
    A = rng.random((N, N)) + np.eye(N) * 5.0   # Make A diagonally dominant to ensure numerical stability
    x_true = rng.random(N)
    b = A @ x_true                              # Construct system so the true solution is known

    tracemalloc.start()                         # Begin tracking memory usage
    start = time.perf_counter()                 # High-precision timer start
    x_pred = solve_linear_system(A, b, method)
    runtime = time.perf_counter() - start       # Measure elapsed runtime
    _, peak = tracemalloc.get_traced_memory()   # Capture peak memory usage
    tracemalloc.stop()

    err = np.linalg.norm(x_pred - x_true) / np.linalg.norm(x_true)  # Relative solution error
    return {"N": N, "seed": seed, "method": method, "runtime_s": runtime,
            "peak_mem_mb": peak / (1024 ** 2), "rel_error": err}


# ------------********------------ Carbon Functions ---------------------
#Resolves an Excel filepath by checking absolute paths first and falling back to the user’s Desktop if needed.

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists(): return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

#Loads a carbon-intensity Excel file, automatically detecting country, year, 
# and intensity columns with flexible name matching.
#Cleans the dataset by selecting the latest year per country (if requested) and normalizing units from g/kWh to kg/kWh when appropriate.

def load_carbon_excel(path: str, year_select: str = "latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    cand_country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
    if cand_country is None: raise ValueError("No 'Country' column found.")
    cand_year = next((v for k,v in cols.items() if "year" in k or "date" in k), None)
    cand_intensity = next((v for k,v in cols.items() if "intensity" in k or ("co2" in k and "kwh" in k)), None)
    if cand_intensity is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cand_intensity = numeric_cols[0]
    keep = [cand_country] + ([cand_year] if cand_year else []) + [cand_intensity]
    df = df[keep].copy()
    df.columns = ["Country","Year","Intensity"] if len(keep)==3 else ["Country","Intensity"]
    if "Year" in df.columns and year_select.lower()=="latest":
        df = df.sort_values(["Country","Year"]).groupby("Country", as_index=False).tail(1)
    med = float(df["Intensity"].dropna().median())
    if med > 50: df["Intensity"] = df["Intensity"]/1000.0
    return df.dropna(subset=["Country","Intensity"]).reset_index(drop=True)

def compute_carbon(perf_df: pd.DataFrame, intensity_df: pd.DataFrame,
                   power_watts: float, pue: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_runtime_s = perf_df["runtime_s"].sum()
    kWh_total = power_watts * pue * total_runtime_s / 3_600_000.0
    df = intensity_df.copy()
    df["kWh"] = kWh_total
    df["kgCO2e"] = df["Intensity"] * df["kWh"]
    summary = pd.DataFrame({
        "total_runtime_s": [total_runtime_s],
        "power_watts": [power_watts],
        "PUE": [pue],
        "kWh_total": [kWh_total],
        "median_intensity_kg_per_kWh": [df["Intensity"].median()],
        "mean_kgCO2e_across_countries": [df["kgCO2e"].mean()]
    })
    return df, summary

# ------------********------------ Plot Helpers ---------------------

plt_kwargs = dict(dpi=140, bbox_inches="tight")  # Common settings for saving plots

def plot_performance(df, outdir):
    g = df.groupby("N")["runtime_s"].mean().reset_index()  # Compute mean runtime for each matrix size
    plt.figure(); plt.plot(g["N"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs Matrix Size"); plt.xlabel("Matrix Size N"); plt.ylabel("Runtime (s)")
    plt.grid(True); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["rel_error"].mean().reset_index()  # Compute mean relative error for each matrix size
    plt.figure(); plt.plot(g2["N"], g2["rel_error"], marker="s")
    plt.title("Performance: Mean Relative Error vs N"); plt.xlabel("Matrix Size N"); plt.ylabel("Relative Error")
    plt.grid(True); plt.savefig(os.path.join(outdir, "perf_error_vs_N.png"), **plt_kwargs); plt.close()


def plot_scalability(df, outdir):
    g = df.groupby("N")["peak_mem_mb"].mean().reset_index()     # Average memory usage for each matrix size
    plt.figure(); plt.plot(g["N"], g["peak_mem_mb"], marker="^")
    plt.title("Scalability: Memory vs Matrix Size"); plt.xlabel("Matrix Size N"); plt.ylabel("Memory (MB)")
    plt.grid(True); plt.savefig(os.path.join(outdir, "scal_memory_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["runtime_s"].mean().reset_index()      # Average runtime for each matrix size
    plt.figure(); plt.loglog(g2["N"], g2["runtime_s"], marker="o")  # Log-log plot to highlight scaling behavior
    plt.title("Scalability: Runtime (log-log)"); plt.xlabel("N"); plt.ylabel("Runtime (s)")
    plt.grid(True, which="both"); plt.savefig(os.path.join(outdir, "scal_loglog_runtime.png"), **plt_kwargs); plt.close()

def plot_reliability(df, outdir):
    plt.figure(); plt.boxplot([df[df["N"]==n]["rel_error"] for n in sorted(df["N"].unique())],
                              tick_labels=sorted(df["N"].unique()))              # Boxplot of error distribution per matrix size
    plt.title("Reliability: Error Distribution by N"); plt.xlabel("Matrix Size N"); plt.ylabel("Relative Error")
    plt.grid(True); plt.savefig(os.path.join(outdir, "rel_error_boxplot.png"), **plt_kwargs); plt.close()

    g = df.groupby("N")["rel_error"].std().reset_index()        # Compute error variability (stddev) per matrix size
    plt.figure(); plt.plot(g["N"], g["rel_error"], marker="o")
    plt.title("Reliability: Std. Dev. of Relative Error vs N"); plt.xlabel("N"); plt.ylabel("Std(Relative Error)")
    plt.grid(True); plt.savefig(os.path.join(outdir, "rel_error_std_vs_N.png"), **plt_kwargs); plt.close()

def plot_carbon(df, outdir):
    top = df.nlargest(15, "kgCO2e")                              # Select top 15 highest carbon-emitting entries
    plt.figure(figsize=(8,5))
    plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])          # Horizontal bar chart, reversed for descending order
    plt.title("Carbon: Top 15 Countries (kgCO2e)")
    plt.xlabel("kg CO2e"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=25)                # Histogram of emission distribution
    plt.title("Carbon: Emission Distribution"); plt.xlabel("kg CO2e"); plt.grid(True)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()


# ------------********------------ Main ---------------------
#Main Parses command-line arguments for running classical linear-solver benchmarks, 
#    including matrix sizes, trial counts, carbon inputs, and CPU parallelization.
#Creates all required output directories, builds a job list, and 
# e xecutes experiments either in parallel via ProcessPoolExecutor or sequentially.
#Collects all per-run results into a DataFrame, printing concise progress updates 
#   for each completed benchmark instance.

def main():
    parser = argparse.ArgumentParser(description="Classical Linear Solver Benchmarks (Baseline for HHL)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[200,400,800,1200])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--method", type=str, default="numpy", choices=["numpy","gauss"])
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))  # Use half of available CPU cores
    args = parser.parse_args()

    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]: os.makedirs(d, exist_ok=True)  # Ensure all output directories exist

    print("[run] Starting classical solver benchmarks...")
    jobs = [(N, 1000 + i, args.method) for N in args.sizes for i in range(args.trials)]  # Create job list: (matrix size, seed, method)
    rows = []


       def _consume(r): rows.append(r); print(f"  - N={r['N']} seed={r['seed']} runtime={r['runtime_s']:.3f}s error={r['rel_error']:.2e}")
        # Collect a single experiment result and print a summary line

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:   # Run experiments in parallel across multiple CPU processes
            futs = [ex.submit(run_experiment, N, seed, m) for (N, seed, m) in jobs]
            for f in as_completed(futs): _consume(f.result())       # Handle results as soon as each worker finishes
    else:
        for N, seed, m in jobs: _consume(run_experiment(N, seed, m))  # Run sequentially if only one worker

    df = pd.DataFrame(rows)    # Convert collected results into a DataFrame for analysis/plotting

    # ------------********------------ Performance -----
    perf_path = os.path.join(perf_dir, "performance_results.xlsx")
    with pd.ExcelWriter(perf_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="raw")
        df.groupby("N").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

    # ------------********------------ Scalability -----
    scal_path = os.path.join(scal_dir, "scalability_results.xlsx")
    with pd.ExcelWriter(scal_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="raw")
    plot_scalability(df, scal_dir)

    # ------------********------------ Reliability -----
    rel_path = os.path.join(rel_dir, "reliability_results.xlsx")
    with pd.ExcelWriter(rel_path, engine="openpyxl") as w:
        df[["N","seed","rel_error"]].to_excel(w, index=False, sheet_name="errors")
    plot_reliability(df, rel_dir)

    # ------------********------------ Carbon Footprint -----

    #Resolves and loads the carbon-intensity Excel file, computes country-wise CO₂ estimates 
    #  from benchmark runtime, and writes detailed + summary sheets to an output workbook.
    #Generates carbon-impact plots and saves them to the designated Carbon output folder, 
    #  handling failures gracefully if inputs are missing or invalid.
    #Prints a final summary of all benchmark outputs—performance, scalability, reliability, 
    ##and carbon—along with file paths for easy navigation.

    excel_path = resolve_excel_path(args.excel)
    try:
        intensity_df = load_carbon_excel(excel_path, year_select=args.year_select)
        carbon_df, summary_df = compute_carbon(df, intensity_df, args.device_power_watts, args.pue)
        carb_path = os.path.join(carb_dir, "carbon_results.xlsx")
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
