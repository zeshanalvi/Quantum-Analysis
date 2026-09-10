#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Quantum Fourier Transform (QFT) — Ideal/Noiseless Benchmarks (Dense Unitary)
# ----------------------------------------------------------------------------------------------------------------

"""
Quantum Fourier Transform (QFT) — Ideal/Noiseless Benchmarks (Dense Unitary)

Per (N, seed):
  - It wl Generate --steps synthetic 2-tone signals of length N with AWGN.
  - Normalize to unit L2 norm; treat as amplitude vector |ψ>.
  - Apply IDEAL QFT via dense matrix F_{k,n} = exp(+2π i k n / N) / sqrt(N).
  - Metrics per batch: mean tone-recovery accuracy (top-2, one-sided),
    mean RMSE vs sqrt(N) * IFFT(x), runtime_s, peak_mem_mb.
  - Success = mean accuracy >= --acc-tol.

Outputs:
  Performance/  Scalability/  Reliability/  Carbon footprints/<outdir>/
"""
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
import os, time, argparse, warnings, pathlib, tracemalloc, math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------
# --------------------------- Signal generation ---------------------------
# ------------********------------

#This function synthesizes a real-valued signal composed of one or two cosine tones placed at integer frequency bins.
#It randomly selects frequency bins and phases using the provided NumPy random generator for reproducibility.
#The signal power is measured to compute additive white Gaussian noise corresponding to a target SNR in dB.
#Finally, noise is added and the function returns the noisy signal along with the true underlying frequency bins.

def make_signal_2tone(N: int, rng: np.random.Generator, snr_db: float = 30.0):
    """Length-N real 2-tone signal with integer bins in [1, N//2-1] + AWGN."""
    if N < 8:
        k1 = rng.integers(1, max(2, N//2))
        true_bins = [int(k1)]
    else:
        k1 = rng.integers(1, N//2)
        k2 = rng.integers(1, N//2 - 1)
        if k2 >= k1: k2 += 1
        true_bins = sorted([int(k1), int(k2)])

    n = np.arange(N, dtype=np.float64)
    ph1 = rng.uniform(0, 2*np.pi)
    ph2 = rng.uniform(0, 2*np.pi)
    x = np.cos(2*np.pi*k1*n/N + ph1)
    if len(true_bins) == 2:
        x = x + np.cos(2*np.pi*k2*n/N + ph2)

    sig_pow = np.mean(x**2)
    snr_lin = 10.0**(snr_db/10.0)
    noise_pow = sig_pow / snr_lin
    x = x + rng.normal(0.0, math.sqrt(noise_pow), size=N)
    return x.astype(np.float64), true_bins


#This utility extracts the indices of the k largest-magnitude frequency components from a one-sided spectrum.
#It operates on the non-redundant FFT bins and optionally suppresses the DC component to emphasize tonal peaks.
#The resulting indices are sorted and returned as integer bin locations corresponding to dominant frequencies.
def topk_one_sided_bins(X: np.ndarray, k: int):
    N = X.shape[0]
    one = X[:N//2+1]
    mag = np.abs(one)
    if k >= 2 and len(mag) > 0:
        mag = mag.copy(); mag[0] = 0.0  # suppress DC when looking for tones
    idx = np.argsort(-mag)[:k]
    return sorted(int(i) for i in idx)

#These following two helper functions evaluate spectral estimation quality and signal reconstruction error.
#The accuracy routine measures how many true frequency bins are recovered within a tolerance from the strongest spectral peaks.
#It normalizes the count to produce a recovery fraction that is robust to edge cases.


def accuracy_recovery(X: np.ndarray, true_bins, tol_bins: int = 0):
    k = len(true_bins)
    est = topk_one_sided_bins(X, k)
    matched = sum(1 for t in true_bins if any(abs(t - e) <= tol_bins for e in est))
    return matched / max(1, k)
#The RMSE function computes the root-mean-square error between two complex-valued arrays by combining real and imaginary parts.
def rmse_complex(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean((diff.real**2 + diff.imag**2))))

# ------------********------------
# --------------------------- Dense QFT (ideal unitary) ---------------------------
# ------------********------------

#This function constructs the full dense matrix representation of an ideal Quantum Fourier Transform.
#It uses outer products of index grids to generate complex exponential phase terms and applies proper normalization.
def qft_dense_matrix(N: int) -> np.ndarray:
    """Ideal QFT matrix F with entries e^{+2π i k n / N} / sqrt(N)."""
    n = np.arange(N).reshape(1, N)
    k = np.arange(N).reshape(N, 1)
    return np.exp(2j * np.pi * (k @ n) / N) / math.sqrt(N)

# ------------********------------
# --------------------------- Single trial (N, seed) ---------------------------
# ------------********------------

#This routine runs a complete experiment loop for a fixed signal length and random seed.
#It precomputes the QFT matrix, then repeatedly generates noisy test signals and normalizes them.
#Each iteration applies the QFT, compares the result against a reference FFT, and records accuracy and error metrics.
#Execution time and peak memory usage are tracked using high-resolution timers and tracemalloc.
#The function aggregates statistics and returns a structured summary of performance and success criteria.

def run_single(N: int, steps: int, acc_tol: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    F = qft_dense_matrix(N)  # precompute QFT once for this N

    tracemalloc.start()
    t0 = time.perf_counter()

    accs, rmses = [], []
    for _ in range(steps):
        x, true_bins = make_signal_2tone(N, rng, snr_db=30.0)
        norm = np.linalg.norm(x)
        x_amp = (np.ones(N)/math.sqrt(N)).astype(np.complex128) if norm == 0.0 else (x / norm).astype(np.complex128)

        # Apply QFT
        state = F @ x_amp

        # Reference spectrum (matches QFT definition)
        X_ref = np.fft.ifft(x_amp) * math.sqrt(N)

        accs.append(accuracy_recovery(state, true_bins, tol_bins=0))
        rmses.append(rmse_complex(state, X_ref))

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    acc_mean = float(np.mean(accs))
    rmse_mean = float(np.mean(rmses))

    return {
        "N": int(N),
        "steps": int(steps),
        "seed": int(seed),
        "accuracy_mean": acc_mean,
        "rmse_vs_fft": rmse_mean,
        "success": bool(acc_mean >= acc_tol),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "signals_processed": int(steps),
    }
# ------------********------------
# --------------------------- Carbon accounting (Performance only) ---------------------------
# ------------********------------

#These utilities handle locating and loading an Excel file containing country-level carbon intensity data.
#The path resolver flexibly checks absolute paths and common desktop locations for user convenience.


def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists(): return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

#The loader function infers relevant columns by name and data type, standardizes them, and optionally selects the latest year.
#It also normalizes units when needed and cleans the dataset by removing missing or invalid entries.

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

#This function estimates total energy consumption and associated carbon emissions from performance measurements.
#It aggregates runtime to compute total energy use, accounting for hardware power draw and data center efficiency.
#The carbon intensity data is combined to estimate per-country emissions and generate summary statistics.

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

#This section defines plotting helpers and visualization routines for summarizing experiment results.
#   It ensures compatibility across Matplotlib versions while configuring consistent figure output settings.
#   The plotting function aggregates results by signal length and generates runtime and accuracy trend figures.
#   Each plot is labeled, styled, and saved to disk for later analysis and reporting.

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)  # Matplotlib ≥3.9
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["runtime_s"], marker="o")
    plt.title("QFT Performance: Runtime vs N"); plt.xlabel("Signal length N"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["accuracy_mean"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["accuracy_mean"], marker="s")
    plt.title("QFT Performance: Mean accuracy vs N"); plt.xlabel("Signal length N"); plt.ylabel("Accuracy (tone recovery)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_accuracy_vs_N.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["N"], g["runtime_s"], "o")
    plt.title("QFT Scalability: Runtime (log–log) vs N"); plt.xlabel("Signal length N"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["peak_mem_mb"], marker="^")
    plt.title("QFT Scalability: Peak memory vs N"); plt.xlabel("Signal length N"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("QFT Reliability: Success rate vs N (acc ≥ tol)"); plt.xlabel("Signal length N"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_N.png"), **plt_kwargs); plt.close()

    data = [df[df["N"]==k]["accuracy_mean"].values for k in sorted(df["N"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["N"].unique()))
    plt.title("QFT Reliability: Accuracy distribution by N"); plt.xlabel("Signal length N"); plt.ylabel("Accuracy")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_accuracy_boxplot.png"), **plt_kwargs); plt.close()

# ------------********------------
# --------------------------- Main ---------------------------
# ------------********------------

def main():
    parser = argparse.ArgumentParser(description="Quantum Fourier Transform (QFT) — Ideal/Noiseless Benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512],
                        help="Signal lengths N (powers of two recommended).")
    parser.add_argument("--steps", type=int, default=50, help="Signals per trial (batch).")
    parser.add_argument("--trials", type=int, default=10, help="Trials (seeds) per N.")
    parser.add_argument("--acc-tol", type=float, default=0.9, help="Success if mean accuracy ≥ this threshold.")
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

    print("[run] Starting QFT (ideal/noiseless, dense) benchmarks...")
    rows, jobs = [], []
    for N in args.sizes:
        for i in range(args.trials):
            seed = 1000 + 37*int(N) + i
            jobs.append((N, args.steps, args.acc_tol, seed))

#This logic collects and reports results from individual experiment runs in a consistent format.
#   It supports both parallel execution using a process pool and sequential execution as a fallback.
#        Completed jobs are consumed as they finish, enabling real-time progress reporting.
#             All results are accumulated into a list and finally converted into a pandas DataFrame for analysis.

    def consume(res: dict):
        rows.append(res)
        print(f"  - N={res['N']} seed={res['seed']} "
              f"acc={res['accuracy_mean']:.3f} rmse={res['rmse_vs_fft']:.3e} "
              f"runtime={res['runtime_s']:.4f}s success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, *job) for job in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for job in jobs:
            consume(run_single(*job))

    df = pd.DataFrame(rows)



#The following blocks export experiment results into structured Excel reports and generates corresponding plots.
#They organize outputs into performance, scalability, reliability, and CO2 views using grouped statistical summaries.
#    Warnings are suppressed to keep file generation clean while writing multiple sheets per workbook.
#    Dedicated plotting functions are invoked to visualize trends and variability across signal sizes.

# ------------********------------
    # ----- Performance (Excel + plots) -----
# ------------********------------

    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("N").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                mean_accuracy=("accuracy_mean","mean"),
                mean_rmse_vs_fft=("rmse_vs_fft","mean"),
                success_rate=("success","mean"),
                mean_signals=("signals_processed","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

# ------------********------------
    # ----- Scalability (Excel + plots) -----
# ------------********------------

    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["N","runtime_s","peak_mem_mb"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("N").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

# ------------********------------
    # ----- Reliability (Excel + plots) -----
# ------------********------------

    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["N","seed","accuracy_mean","rmse_vs_fft","success"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("N").agg(
                success_rate=("success","mean"),
                mean_acc=("accuracy_mean","mean"),
                std_acc=("accuracy_mean","std"),
                mean_rmse=("rmse_vs_fft","mean"),
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

# ------------********------------
    # ----- Carbon (PERFORMANCE runtime only) -----
# ------------********------------

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
        # Carbon plots
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
