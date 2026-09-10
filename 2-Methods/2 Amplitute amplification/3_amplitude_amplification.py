#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Amplitude Amplification
# This script implements GPU-accelerated Grover's algorithm for unstructured search.
"""
GPU-accelerated Quantum Amplitude Amplification (Grover) — Ideal/Noiseless Benchmarks

This script produces four benchmarks:
  1) Performance     – runtime, memory, success rate
  2) Scalability     – runtime & memory vs problem size
  3) Reliability     – stability of success rate across seeds
  4) Carbon footprint– CO₂ based solely on Performance runtimes

Folders created in the current directory:
  Performance/, Scalability/, Reliability/, Carbon footprints/<outdir>/
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

import os, sys, time, tracemalloc, pathlib, warnings, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------********------------
# GPU Setup and Dependencies
# ------------********------------

#This block detects whether a CUDA-capable GPU is available and functional using CuPy.
#   It performs a minimal computation test to verify GPU execution and reports hardware details.
#   If CuPy is missing or initialization fails, the code safely falls back to NumPy on the CPU.
#   Clear status messages indicate which computational backend is ultimately selected.

print("🔍 Checking GPU availability...")

HAS_GPU = False
cp = None

try:
    import cupy as cp
    # Test if GPU is available and working
    with cp.cuda.Device(0):
        # Simple test to verify GPU functionality
        test_array = cp.zeros(10, dtype=cp.float32)
        result = cp.sum(test_array)
        HAS_GPU = True
        print(f"✅ GPU detected: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"   Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB")
        print(f"   Compute Capability: {cp.cuda.runtime.getDeviceProperties(0)['major']}.{cp.cuda.runtime.getDeviceProperties(0)['minor']}")
except ImportError:
    print("❌ CuPy not available, falling back to NumPy")
    import numpy as cp
except Exception as e:
    print(f"⚠️  GPU setup failed: {e}, falling back to NumPy")
    import numpy as cp

# If GPU failed, use NumPy
if not HAS_GPU:
    import numpy as cp
    print("🔄 Using NumPy (CPU) backend")


#This setup code checks for required Python packages and installs any that are missing at runtime.
#   It uses dynamic imports to detect dependencies and invokes pip automatically when needed.
#   Core scientific and plotting libraries are then imported and configured for headless execution.

# Install required packages
def _ensure(pkgs: List[str]):
    import importlib, subprocess
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)
        except ImportError:
            missing.append(p)
    if missing:
        print(f"[setup] Installing missing packages: {missing}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        except Exception as e:
            print(f"[setup] Auto-install failed: {e}")

_ensure(["pandas", "matplotlib", "openpyxl", "psutil"])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPU utility functions
#These helper routines abstract hardware details and resource management for GPU-enabled execution.
#   The device name function reports whether computation is running on a GPU or CPU with safe fallback handling.


def get_device_name() -> str:
    """Get current device name"""
    if HAS_GPU:
        try:
            return f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
        except:
            return "GPU: Unknown"
    return "CPU: NumPy"

#   A cleanup utility releases cached GPU memory to avoid leaks between runs.
#   A global lock is defined to ensure thread-safe access when multiple threads share GPU resources.
def safe_gpu_cleanup():
    """Safely cleanup GPU memory"""
    if HAS_GPU:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass

# Thread-safe GPU operations
_gpu_lock = threading.Lock()

def run_experiment_thread_safe(args):
    
    # Thread-safe GPU experiment runner.
    # Uses a lock to prevent multiple threads from accessing GPU simultaneously.
    
    N, steps, acc_tol, seed, M = args
    with _gpu_lock:  # Ensure only one thread uses GPU at a time
        return _run_single_experiment(N, steps, acc_tol, seed, M)

# ------------********------------
# ========================= Quantum Amplitude Amplification Core 
# ------------********------------

def make_marked_mask(N: int, M: int, rng) -> cp.ndarray:
    #   Boolean mask of length N with exactly M marked items."""
    if M < 0 or M > N:
        raise ValueError(f"Invalid M={M} for N={N}")
    mask = cp.zeros(N, dtype=cp.bool_)
    if M > 0:
        # Generate indices on CPU then transfer to GPU
        idx = rng.choice(N, size=M, replace=False)
        mask_gpu = cp.asarray(idx)
        mask[mask_gpu] = True
    return mask

def grover_iterations_k(N: int, M: int) -> int:
    #  k ≈ round((π/4)*sqrt(N/M)); force >=1 when M>0."""
    if M <= 0:
        return 0
    k = int(round((math.pi/4.0) * math.sqrt(N / M)))
    return max(1, k)

def phase_oracle_inplace(state: cp.ndarray, mask: cp.ndarray):
    #   Phase flip on marked indices (ideal, noiseless)."""
    state[mask] *= -1.0

def diffusion_inplace(state: cp.ndarray):
    
    #  Diffusion (reflection about |s>): D|ψ> = 2|s><s|ψ> - |ψ|.
    #  For uniform |s>, this is equivalent to: state <- 2*avg - state.
    
    avg = cp.mean(state)
    state[:] = 2.0 * avg - state

def grover_cycle(state: cp.ndarray, mask: cp.ndarray):
    #   One Grover iteration: phase oracle then diffusion."""
    phase_oracle_inplace(state, mask)
    diffusion_inplace(state)

def run_instance(N: int, M: int, rng):
   
    #One quantum search instance:
    # - Create a fresh marked set S
    #  - Prepare uniform |s>
    # - Apply k Grover iterations
    #  - Measure once; return (success, queries_used, k)
    
    mask = make_marked_mask(N, M, rng)
    k = grover_iterations_k(N, M)
    if M == 0 or k == 0:
        return False, 0, 0

    # Uniform superposition
    state = cp.ones(N, dtype=cp.complex128) / math.sqrt(N)

    # k Grover iterations
    for _ in range(k):
        grover_cycle(state, mask)

    # Measurement probabilities
    probs = cp.real(state * cp.conj(state))

    # Normalize probabilities
    s = cp.sum(probs)
    if s <= 0.0:
        probs = cp.ones(N, dtype=cp.float64) / N
    else:
        probs = probs / s

    # Convert to CPU for random choice (CuPy doesn't have choice with probabilities)
    probs_cpu = cp.asnumpy(probs)
    i = rng.choice(N, p=probs_cpu)

    # Check if marked
    mask_cpu = cp.asnumpy(mask)
    success = bool(mask_cpu[i])
    queries_used = k  # one phase oracle per iteration

    return success, queries_used, k

def _run_single_experiment(N: int, steps: int, acc_tol: float, seed: int, M: int) -> dict:
   
   # Core experiment logic - run single Grover's algorithm benchmark.
  
    # Use numpy RNG for consistency
    rng_np = np.random.default_rng(seed)

    tracemalloc.start()
    t0 = time.perf_counter()

    successes = 0
    queries_total = 0
    budgets = []  # the k used in each instance

    for _ in range(steps):
        success, queries, k = run_instance(N, M, rng_np)
        successes += int(success)
        queries_total += queries
        budgets.append(k)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    success_rate = successes / max(1, steps)
    mean_queries = queries_total / max(1, steps)
    mean_k = float(np.mean(budgets)) if budgets else 0.0

    # Cleanup GPU memory
    safe_gpu_cleanup()

    return {
        "N": int(N),
        "steps": int(steps),
        "seed": int(seed),
        "M": int(M),
        "success_rate": float(success_rate),
        "mean_queries": float(mean_queries),
        "mean_grover_k": mean_k,
        "success": bool(success_rate >= acc_tol),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "instances_processed": int(steps),
        "device": get_device_name()
    }

# ------------********------------
# ============================== Carbon I/O ==============================
# ------------********------------
#This helper resolves an Excel file path by checking absolute paths first.
#If needed, it also looks for the file on the Desktop before falling back to the original argument.

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
                   power_watts: float, pue: float, combine: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_runtime_s = float(perf_df["runtime_s"].sum())
    kWh_total = power_watts * pue * total_runtime_s / 3_600_000.0
    df = intensity_df.copy()
    df["kWh"] = kWh_total
    df["kgCO2e"] = df["Intensity"] * df["kWh"]

    # ALWAYS return country-wise results (ignore combine parameter)
    carbon_output = df.sort_values("kgCO2e", ascending=False)

    summary = pd.DataFrame({
        "total_runtime_s": [total_runtime_s],
        "power_watts": [power_watts],
        "PUE": [pue],
        "kWh_total": [kWh_total],
        "median_intensity_kg_per_kWh": [float(df["Intensity"].median())],
        "mean_kgCO2e_across_countries": [float(df["kgCO2e"].mean())],
        "device": [perf_df["device"].iloc[0] if "device" in perf_df.columns else "Unknown"],
        "combine_mode": [combine],
        "countries_analyzed": [len(df)]
    })
    return carbon_output, summary

# ------------********------------
# ============================== Plot Helpers ==============================
# ------------********------------

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    # Matplotlib 3.9 renamed 'labels' -> 'tick_labels'
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    #  Plot performance metrics
    # Runtime vs N
    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(g["N"], g["runtime_s"], marker="o", linewidth=2, markersize=8)
    plt.title(f"Performance: Runtime vs Search Space Size\n{get_device_name()}")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Runtime (s)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs)
    plt.close()

    # Success rate vs N
    g2 = df.groupby("N")["success_rate"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(g2["N"], g2["success_rate"], marker="s", linewidth=2, markersize=8)
    plt.title(f"Performance: Success Rate vs Search Space Size\n{get_device_name()}")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Success Rate")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(outdir, "perf_success_vs_N.png"), **plt_kwargs)
    plt.close()

    # Queries vs N
    g3 = df.groupby("N")["mean_queries"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(g3["N"], g3["mean_queries"], marker="^", linewidth=2, markersize=8, color='green')
    plt.title(f"Performance: Mean Oracle Queries vs Search Space Size\n{get_device_name()}")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Mean Oracle Queries")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "perf_queries_vs_N.png"), **plt_kwargs)
    plt.close()

# ------------********------------
def plot_scalability(df: pd.DataFrame, outdir: str):
    #  Plot scalability metrics
    # Runtime scaling with N (log-log)

    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.loglog(g["N"], g["runtime_s"], marker="o", markersize=8)
    plt.title(f"Scalability: Runtime vs Search Space Size (log-log)\n{get_device_name()}")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Runtime (s)")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_N.png"), **plt_kwargs)
    plt.close()

    # Memory scaling with N
    g2 = df.groupby("N")["peak_mem_mb"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(g2["N"], g2["peak_mem_mb"], marker="s", linewidth=2, markersize=8, color='red')
    plt.title(f"Scalability: Memory vs Search Space Size\n{get_device_name()}")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Peak Memory (MB)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "scal_memory_vs_N.png"), **plt_kwargs)
    plt.close()

# ------------********------------
def plot_reliability(df: pd.DataFrame, outdir: str):
    #  Plot reliability metrics
    # Success rate distribution by N
    data = [df[df["N"] == n]["success_rate"].values for n in sorted(df["N"].unique())]
    plt.figure(figsize=(10, 6))
    _boxplot_with_labels(data, labels=sorted(df["N"].unique()))
    plt.title(f"Reliability: Success Rate Distribution by Search Space Size\n{get_device_name()}")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Success Rate")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "rel_success_distribution.png"), **plt_kwargs)
    plt.close()

    # Success rate trend
    success_rates = df.groupby('N')['success'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(success_rates['N'], success_rates['success'], marker='o', linewidth=2, markersize=8)
    plt.title(f"Reliability: Batch Success Rate vs Search Space Size\n{get_device_name()}")
    plt.xlabel("Search Space Size (N)")
    plt.ylabel("Batch Success Rate")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(outdir, "rel_batch_success.png"), **plt_kwargs)
    plt.close()

# ------------********------------
def plot_carbon(df: pd.DataFrame, combine: bool, outdir: str):
    #   Generate carbon footprint plots - ALWAYS country-wise
    # Top 15 highest emissions
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(12, 8))
    plt.barh(top["Country"][::-1], top["kgCO2e"][::-1], color='red', alpha=0.7)
    plt.xlabel("kg CO2e (higher = dirtier grid)", fontsize=12)
    plt.title(f"Carbon: Top 15 Highest Emissions\n{get_device_name()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs)
    plt.close()

    # Bottom 15 lowest emissions
    bot = df.nsmallest(15, "kgCO2e")
    plt.figure(figsize=(12, 8))
    plt.barh(bot["Country"][::-1], bot["kgCO2e"][::-1], color='green', alpha=0.7)
    plt.xlabel("kg CO2e (lower = cleaner grid)", fontsize=12)
    plt.title(f"Carbon: Bottom 15 Lowest Emissions\n{get_device_name()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "carbon_bottom15.png"), **plt_kwargs)
    plt.close()

    # Distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["kgCO2e"], bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel("kg CO2e for this experiment", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Carbon: Emission Distribution\n{get_device_name()}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs)
    plt.close()

    # CDF plot
    xs = np.sort(df["kgCO2e"].values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, linewidth=2, color='purple')
    plt.xlabel("kg CO2e", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title(f"Carbon: Cumulative Distribution\n{get_device_name()}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_cdf.png"), **plt_kwargs)
    plt.close()

# ------------********------------
# ============================== Utility Functions ==============================
# ------------********------------

#These utility functions support output management for experiment results.
#   def ensure_dir function  ensures required directories exist without raising errors.
#   The def to_excel writes multiple pandas DataFrames into a single Excel file using separate, safely named sheets.

def ensure_dir(d: str):
    """Ensure directory exists"""
    os.makedirs(d, exist_ok=True)

def to_excel(dfs: Dict[str, pd.DataFrame], path: str):
    """Save multiple DataFrames to Excel with different sheets"""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in dfs.items():
            df.to_excel(w, index=False, sheet_name=name[:31])

# ------------********------------
# ============================== Main Experiment Runner ==============================
# ------------********------------

def run_grover_experiment():

    #  Main Grover's Algorithm experiment runner with hardcoded parameters for Kaggle.
    #  Equivalent to: --sizes 32 64 128 256 --steps 200 --trials 10 --marked-count 1 --acc-tol 0.9
    #               --device-power-watts 65 --pue 1.2 --combine
   
    # Experiment parameters
    sizes = [32, 64, 128, 256]  # Search space sizes
    trials = 10                  # seeds per N
    steps = 200                  # instances per (N, seed) batch
    marked_count = 1             # number of marked items M
    acc_tol = 0.9               # batch success threshold
    method = "grover_search"
    workers = 4                  # Number of parallel threads

    # Carbon parameters
    device_power_watts = 65.0
    pue = 1.2
    combine = True

    print(f"🚀 Starting Quantum Amplitude Amplification (Grover) experiments on {get_device_name()}")
    print(f"🔧 Configuration:")
    print(f"   Search space sizes: {sizes}")
    print(f"   Trials per size: {trials}")
    print(f"   Instances per trial: {steps}")
    print(f"   Marked items: {marked_count}")
    print(f"   Accuracy tolerance: {acc_tol}")
    print(f"   Method: {method}")
    print(f"   Workers: {workers}")
    print(f"   Device power: {device_power_watts}W")
    print(f"   PUE: {pue}")
    print(f"   Combine carbon: {combine}")

    # Create output folders
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, "carbon_by_country")

    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        ensure_dir(d)

    # Prepare jobs
    jobs = []
    for N in sizes:
        for i in range(trials):
            seed = 1000 + 23 * N + i
            jobs.append((N, steps, acc_tol, seed, marked_count))

    all_rows = []

    def _consume(row):
        #  Process completed job results
        all_rows.append(row)
        print(f"  ✅ N={row['N']}, seed={row['seed']} "
              f"(runtime={row['runtime_s']:.4f}s, success_rate={row['success_rate']:.3f}, "
              f"queries={row['mean_queries']:.1f}, success={row['success']})")

    # Execute jobs with thread-based parallelization
    print(f"🔀 Running {len(jobs)} jobs in parallel with {workers} threads...")

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(run_experiment_thread_safe, job): job for job in jobs}

            # Collect results as they complete
            for future in as_completed(future_to_job):
                try:
                    result = future.result()
                    _consume(result)
                except Exception as e:
                    job = future_to_job[future]
                    print(f"❌ Job failed N={job[0]}, seed={job[3]}: {e}")
    else:
        # Fallback to sequential execution
        for job in jobs:
            result = run_experiment_thread_safe(job)
            _consume(result)

    # Check if we have any successful results
    if not all_rows:
        print("💥 CRITICAL: No experiments completed successfully!")
        return None

    perf_df = pd.DataFrame(all_rows)

# ------------********------------
    # ---------------- Performance Results ----------------
# ------------********------------

#This block aggregates Grover performance metrics by problem size and prepares summary statistics.
#       Both raw run data and averaged results are written to an Excel workbook with multiple sheets.
#       Warnings are suppressed during file output to keep logs clean.
#       Performance plots are generated to visualize trends across different values of N.

    print("📊 Generating Performance results...")
    perf_excel = os.path.join(perf_dir, "performance_results.xlsx")
    perf_agg = perf_df.groupby("N").agg({
        "runtime_s": "mean",
        "success_rate": "mean",
        "mean_queries": "mean",
        "mean_grover_k": "mean",
        "peak_mem_mb": "mean",
        "success": "mean"
    }).reset_index()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_excel({
            "raw_runs": perf_df,
            "aggregated": perf_agg
        }, perf_excel)

    plot_performance(perf_df, perf_dir)

# ------------********------------
    # ---------------- Scalability Results ----------------
# ------------********------------

    print("📈 Generating Scalability results...")
    scal_excel = os.path.join(scal_dir, "scalability_results.xlsx")
    scal_agg = perf_df.groupby("N").agg({
        "runtime_s": "mean",
        "peak_mem_mb": "mean",
        "mean_queries": "mean"
    }).reset_index()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_excel({
            "from_perf_runs": perf_df[["N", "seed", "runtime_s", "peak_mem_mb", "mean_queries"]],
            "aggregated": scal_agg
        }, scal_excel)

    plot_scalability(perf_df, scal_dir)

# ------------********------------
    # ---------------- Reliability Results ----------------
# ------------********------------

    print("🎯 Generating Reliability results...")
    rel_excel = os.path.join(rel_dir, "reliability_results.xlsx")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_excel({
            "success_analysis": perf_df[["N", "seed", "success_rate", "mean_queries", "success"]]
        }, rel_excel)

    plot_reliability(perf_df, rel_dir)

# ------------********------------
    # ---------------- Carbon Footprint Results ----------------
# ------------********------------

    print("🌍 Generating Carbon footprint results...")

    # Load carbon data from Kaggle input directory
    excel_path = "/kaggle/input/masters/Filtered CO2 intensity 236 Countries.xlsx"
    intensity_df = None

    try:
        # Try to load from the exact path
        intensity_df = pd.read_excel(excel_path)
        print(f"✅ Loaded carbon data from: {excel_path}")
        print(f"   Columns found: {list(intensity_df.columns)}")
        print(f"   Countries loaded: {len(intensity_df)}")

        # Auto-detect and process columns
        cols = {c.lower(): c for c in intensity_df.columns}
        cand_country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
        if cand_country is None:
            raise ValueError("No 'Country' column found.")

        cand_intensity = next((v for k,v in cols.items() if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)

        if cand_intensity is None:
            numeric_cols = [c for c in intensity_df.columns if pd.api.types.is_numeric_dtype(intensity_df[c])]
            if not numeric_cols:
                raise ValueError("No numeric intensity column detected.")
            cand_intensity = numeric_cols[0]

        # Keep only necessary columns
        intensity_df = intensity_df[[cand_country, cand_intensity]].copy()
        intensity_df.columns = ["Country", "Intensity"]

        # Convert gCO2/kWh to kgCO2/kWh if needed
        med = float(intensity_df["Intensity"].dropna().median())
        if med > 50:
            intensity_df["Intensity"] = intensity_df["Intensity"] / 1000.0

        intensity_df = intensity_df.dropna(subset=["Country", "Intensity"]).reset_index(drop=True)
        print(f"✅ Processed carbon data for {len(intensity_df)} countries")

    except Exception as e:
        print(f"❌ ERROR loading carbon data: {e}")
        print("💡 Using sample carbon data as fallback...")
        # Create sample data similar to other codes
        countries = ["France", "Sweden", "Norway", "Switzerland", "Austria", "Germany", "United States", "China", "India"]
        intensities = [52, 41, 32, 45, 62, 385, 389, 537, 708]
        intensity_df = pd.DataFrame({
            "Country": countries,
            "Intensity": [i/1000.0 for i in intensities]  # Convert to kg/kWh
        })

    ensure_dir(carb_dir)
    carbon_df, summary_df = compute_carbon(perf_df, intensity_df,
                                           power_watts=device_power_watts,
                                           pue=pue, combine=combine)
    carb_excel = os.path.join(carb_dir, "carbon_results.xlsx")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_excel({
            "carbon_results": carbon_df,
            "summary": summary_df,
            "intensity_input": intensity_df,
            "performance_reference": perf_df[["N", "seed", "runtime_s"]]
        }, carb_excel)

    # Generate carbon plots
    plot_carbon(carbon_df, combine, carb_dir)
    print(f"✅ Carbon analysis complete: {len(carbon_df)} entries")

    # Final summary
    print("\n" + "="*60)
    print("🎉 QUANTUM AMPLITUDE AMPLIFICATION EXPERIMENT COMPLETE - RESULTS SUMMARY")
    print("="*60)
    print(f"📁 Performance:     {perf_excel}")
    print(f"📁 Scalability:     {scal_excel}")
    print(f"📁 Reliability:     {rel_excel}")
    print(f"📁 Carbon:          {carb_excel}")
    print(f"🖥️  Device:          {get_device_name()}")
    print(f"📊 Total runs:      {len(perf_df)}")
    print(f"⚡ Total runtime:   {perf_df['runtime_s'].sum():.2f}s")
    print(f"🔬 Search spaces:   {sizes}")
    print(f"🎯 Success rate:    {perf_df['success'].mean() * 100:.1f}%")
    print(f"🔍 Total instances: {perf_df['instances_processed'].sum():,}")

    print(f"\n📂 All results saved to:")
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        print(f"   • {d}")

    return perf_df

# ------------********------------
# ------------********------------ Main execution for Kaggle
# ------------********------------

#This main execution block of code initializes the environment and launches the full Grover benchmark workflow.
#       It performs GPU cleanup, runs all experiments, and reports aggregate performance statistics.
#       A detailed summary of results and generated artifacts is printed, including output files and directories.
#       The block concludes with a success or failure message depending on whether results were produced.

if __name__ == "__main__":
    # Clean up GPU state
    safe_gpu_cleanup()

    print("🚀 Starting Quantum Amplitude Amplification (Grover) Benchmark on Kaggle")
    print("=" * 50)

    # Run the complete experiment
    results_df = run_grover_experiment()

    if results_df is not None:
        # Display final results summary
        print("\n📊 FINAL RESULTS SUMMARY")
        print("=" * 30)
        print(f"Total experiments: {len(results_df)}")
        print(f"Search space sizes: {sorted(results_df['N'].unique())}")
        print(f"Average runtime: {results_df['runtime_s'].mean():.6f}s")
        print(f"Success rate: {results_df['success'].mean() * 100:.1f}%")
        print(f"Average queries per instance: {results_df['mean_queries'].mean():.1f}")
        print(f"Total quantum instances: {results_df['instances_processed'].sum():,}")
        print(f"Device used: {get_device_name()}")

        # Show folder structure
        print("\n📁 GENERATED OUTPUT STRUCTURE:")
        base_dir = '/kaggle/working'
        for root, dirs, files in os.walk(base_dir):
            # Only show our output directories
            if any(x in root for x in ['Performance', 'Scalability', 'Reliability', 'Carbon footprints']):
                level = root.replace(base_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}📁 {os.path.basename(root)}/')
                sub_indent = ' ' * 2 * (level + 1)
                for file in files:
                    if file.endswith(('.xlsx', '.png')):
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path) / 1024  # KB
                        icon = "📊" if file.endswith('.xlsx') else "🖼️"
                        print(f'{sub_indent}{icon} {file} ({size:.1f} KB)')

        print("\n✅ All Quantum Amplitude Amplification deliverables generated successfully!")
        print("   • Performance/performance_results.xlsx + perf_*.png")
        print("   • Scalability/scalability_results.xlsx + scal_*.png")
        print("   • Reliability/reliability_results.xlsx + rel_*.png")
        print("   • Carbon footprints/carbon_by_country/carbon_results.xlsx + carbon_*.png")
    else:
        print("❌ Quantum Amplitude Amplification benchmark failed - no results generated")

