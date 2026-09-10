#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# swap_test_benchmarks.py
# Ideal (noiseless) quantum SWAP test using amplitude-encoded vectors.
#
# Notes:
# - For normalized vectors x,y (amplitude encoding), SWAP test gives:
#     p0 = (1 + |<x|y>|^2) / 2  ,  S_swap = 2*p0 - 1 = |<x|y>|^2 = (cos(theta))^2
# - We compute p0 analytically (ideal), and optionally simulate measurement with --shots.
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
#These imports set up system utilities for timing, memory tracking, argument parsing, and process management.
#Scientific libraries support numerical computation, data handling, and plotting of benchmark results.
#Typing and decimal helpers improve precision control and code clarity, while warnings are suppressed for clean output.

import os
import sys
import time
import argparse
import warnings
import tracemalloc
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ------------********-------------
# ------------********------------- Filesystem & plotting utils
# ------------********-------------
#This helper ensures that an output directory exists before files are written.
#It safely creates the directory and ignores the case where it already exists.

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

#This function writes multiple pandas DataFrames into a single Excel workbook.
#If Excel writing fails, it falls back to exporting each table as a separate CSV file.

def write_excel(df_dict: Dict[str, pd.DataFrame], out_xlsx: str) -> None:
    """Write multiple DataFrames to one Excel file; fall back to CSVs if engine missing."""
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            for sheet, df in df_dict.items():
                df.to_excel(writer, index=False, sheet_name=(sheet[:31] or "Sheet1"))
        print(f"[OK] Wrote Excel: {out_xlsx}")
    except Exception as e:
        print(f"[WARN] Could not write Excel ({e}). Falling back to CSVs.")
        base = os.path.splitext(out_xlsx)[0]
        for sheet, df in df_dict.items():
            path = f"{base}__{sheet}.csv"
            df.to_csv(path, index=False)
            print(f"[OK] Wrote CSV fallback: {path}")

#This utility saves a Matplotlib figure with consistent layout and resolution.
#It closes the figure after saving to free resources and reports the output location.
def save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved plot: {path}")

# ------------********-------------
# ------------********------------- Desktop file resolver
# ------------********-------------

def resolve_desktop_file(user_arg: Optional[str], default_basenames: List[str]) -> Optional[str]:
   
    #Resolve a CO2 file on Desktop. If user_arg is:
    #  - a full path -> use it if exists,
     # - a bare name (with/without extension) -> search Desktop with .xlsx/.xls/.csv.
    #If user_arg is None -> try defaults on Desktop.
   
    desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
    def exists(p): return os.path.isfile(p)

    if user_arg:
        base = user_arg.strip().strip('"').strip("'")
        if exists(base):
            return os.path.abspath(base)
        candidate = os.path.join(desktop, base)
        if exists(candidate):
            return os.path.abspath(candidate)
        if not os.path.splitext(base)[1]:
            for ext in (".xlsx", ".xls", ".csv"):
                candidate = os.path.join(desktop, base + ext)
                if exists(candidate):
                    return os.path.abspath(candidate)
        # Case-insensitive scan on Desktop
        tail = os.path.basename(base).lower()
        for fname in os.listdir(desktop):
            fname_low = fname.lower()
            if fname_low == tail or any((not os.path.splitext(tail)[1]) and fname_low == (tail + ext)
                                        for ext in (".xlsx", ".xls", ".csv")):
                cand = os.path.join(desktop, fname)
                if exists(cand):
                    return os.path.abspath(cand)
        return None

    for b in default_basenames:
        for ext in ("", ".xlsx", ".xls", ".csv"):
            cand = os.path.join(desktop, b + ext)
            if exists(cand):
                return os.path.abspath(cand)
    return None

# ------------********-------------
# ------------********------------- Vector generation & swap test math
# ------------********-------------

#This function generates a list of dimensions that are powers of two within a specified range.
#It ensures a minimum dimension of 2 and iteratively builds valid sizes up to the maximum.

def dims_from_range(dmin: int, dmax: int) -> List[int]:
    """Powers of two between [dmin, dmax]."""
    if dmin < 2:
        dmin = 2
    dims = []
    d = 1
    while d < dmin:
        d *= 2
    while d <= dmax:
        dims.append(d)
        d *= 2
    return dims

#This function creates paired vectors for similarity experiments.
#It supports uniform or normal distributions and returns matched batches A and B.

def generate_pairs(dim: int, pairs: int, distribution: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if distribution == "uniform":
        A = rng.uniform(-1.0, 1.0, size=(pairs, dim)).astype(np.float64)
        B = rng.uniform(-1.0, 1.0, size=(pairs, dim)).astype(np.float64)
    else:  # normal
        A = rng.standard_normal(size=(pairs, dim), dtype=np.float64)
        B = rng.standard_normal(size=(pairs, dim), dtype=np.float64)
    return A, B

#This routine computes dot products and cosine similarities for batches of vector pairs.
#It optionally enforces normalization to mimic amplitude encoding and returns cos² values.

def cosine_batch(A: np.ndarray, B: np.ndarray, enforce_normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (dot, cosine, cosine_squared). If enforce_normalize=True, we normalize
    vectors to unit length (amplitude encoding requirement).
    """
    nA = np.linalg.norm(A, axis=1)
    nB = np.linalg.norm(B, axis=1)
    dot = np.einsum("ij,ij->i", A, B)

    # True cosine (direction-only similarity)
    cos = dot / (nA * nB + 1e-16)
    cos = np.clip(cos, -1.0, 1.0)

    if enforce_normalize:
        # Normalize A,B to unit vectors (amplitude encoding); not needed to compute cos,
        # but we do it to faithfully reflect the state-prep step.
        A /= (nA[:, None] + 1e-16)
        B /= (nB[:, None] + 1e-16)

    cos_sq = cos ** 2
    return dot, cos, cos_sq

#This helper computes the ideal SWAP-test probability p0 from cosine-squared values.
#It represents the noiseless theoretical relationship used in quantum similarity tests.

def swap_p0_from_cos_sq(cos_sq: np.ndarray) -> np.ndarray:
    """Ideal (noiseless) SWAP test: p0 = (1 + cos^2)/2."""
    return 0.5 * (1.0 + cos_sq)

#This function simulates measurement outcomes by sampling ancilla results.
#It returns both the estimated p0 and a transformed estimator for cosine squared.


def sample_p0(p0: np.ndarray, shots: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample ancilla outcomes with given p0 per pair. Returns (p0_hat, s_hat),
    where s_hat = 2*p0_hat - 1 is an estimator of cos^2.
    """
    if shots <= 0:
        raise ValueError("shots must be >= 1 for sampling.")
    k = rng.binomial(n=shots, p=np.clip(p0, 0.0, 1.0))
    p0_hat = k.astype(np.float64) / float(shots)
    s_hat = 2.0 * p0_hat - 1.0
    return p0_hat, s_hat

#This helper computes cosine similarity using high-precision arithmetic.
#It serves as a reference for validating numerical accuracy when sampling is disabled.
# High-precision reference for reliability (when shots=0)
def cosine_high_prec(a: np.ndarray, b: np.ndarray, prec: int = 60) -> float:
    with localcontext() as ctx:
        ctx.prec = prec
        d_dot = Decimal(0)
        d_n1 = Decimal(0)
        d_n2 = Decimal(0)
        for x, y in zip(a, b):
            dx = Decimal(str(float(x)))
            dy = Decimal(str(float(y)))
            d_dot += dx * dy
            d_n1 += dx * dx
            d_n2 += dy * dy
        if d_n1 == 0 or d_n2 == 0:
            return 0.0
        denom = d_n1.sqrt() * d_n2.sqrt()
        try:
            val = d_dot / denom
        except Exception:
            return 0.0
        return float(max(-1.0, min(1.0, val)))

# ------------********-------------
# ------------********------------- Work units for multiprocessing
# ------------********-------------

#This worker function runs a single performance benchmark for cosine similarity computation.
#It measures runtime and peak memory usage while collecting aggregate statistics over vector pairs.

def _perf_task(args):
    dim, pairs, distribution, enforce_norm, repeat, seed, shots = args
    rng = np.random.default_rng(seed + dim * 101 + repeat * 17)
    A, B = generate_pairs(dim, pairs, distribution, rng)

    tracemalloc.start()
    t0 = time.perf_counter()

    # Ideal p0
    _, cos, cos_sq = cosine_batch(A, B, enforce_normalize=enforce_norm)
    p0 = swap_p0_from_cos_sq(cos_sq)

    # If sampling, include it in performance timing
    if shots and shots > 0:
        p0_hat, s_hat = sample_p0(p0, shots, rng)
        mean_s_measure = float(np.mean(s_hat))
    else:
        mean_s_measure = float(np.mean(2.0 * p0 - 1.0))  # equals mean cos^2

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "dimension": dim,
        "pairs": pairs,
        "repeat": repeat,
        "runtime_sec": runtime,
        "peak_mem_mb": peak / (1024 ** 2),
        "mean_p0": float(np.mean(p0)),
        "mean_cosine_sq": float(np.mean(cos_sq)),
        "mean_swap_estimate": mean_s_measure,  # equals mean_cosine_sq if shots=0
    }

#This helper evaluates numerical reliability on a small subset of vector pairs.
#It compares standard cosine results against a high-precision reference and records absolute errors.

def _reli_subset(args):
    dim, pairs_for_rel, distribution, enforce_norm, repeat, seed, shots = args
    rng = np.random.default_rng(seed + 7919 + dim * 23 + repeat)
    k = pairs_for_rel
    A, B = generate_pairs(dim, k, distribution, rng)
    _, cos, cos_sq = cosine_batch(A, B, enforce_normalize=enforce_norm)
    p0 = swap_p0_from_cos_sq(cos_sq)

    if shots and shots > 0:
        p0_hat, s_hat = sample_p0(p0, shots, rng)
        # Reliability error: sampled S_hat vs ground-truth cos^2
        errs = np.abs(s_hat - cos_sq)
    else:
        # Noiseless: compare float64 cos^2 vs high-precision (cos)^2
        errs = []
        for i in range(k):
            ref_cos = cosine_high_prec(A[i], B[i], prec=60)
            errs.append(abs((ref_cos ** 2) - float(cos_sq[i])))
        errs = np.array(errs, dtype=np.float64)

    return pd.DataFrame({
        "dimension": [dim] * k,
        "repeat": [repeat] * k,
        "pair_index": list(range(k)),
        "abs_error": errs
    })

# ------------********--------------- Reliability benchmark (subset) ---
def run_reliability(dims: List[int],
                    pairs_for_rel: int,
                    distribution: str,
                    enforce_norm: bool,
                    seed: int,
                    repeats: int,
                    workers: int,
                    shots: int) -> pd.DataFrame:
    tasks = [(d, pairs_for_rel, distribution, enforce_norm, r, seed, shots)
             for d in dims for r in range(repeats)]
    dfs = []
    if workers and workers > 1:
        workers = min(workers, cpu_count())
        with Pool(processes=workers) as pool:
            for sub in pool.imap_unordered(_reli_subset, tasks):
                dfs.append(sub)
    else:
        for t in tasks:
            dfs.append(_reli_subset(t))
    if not dfs:
        return pd.DataFrame(columns=["dimension", "repeat", "pair_index", "abs_error"])
    return pd.concat(dfs, ignore_index=True)

# ------------********-------------
# ------------********------------- Carbon logic + plots
# ------------********-------------

def load_country_co2(path: str, year_select: str = "latest") -> pd.DataFrame:
    
    # Reads CSV/XLSX with at least 'country' and one intensity-like column:
    #  - 'intensity' (preferred) OR any header containing 'co2' & 'kwh', or ('kg'|'g') & 'kwh'.
    # Optional 'year' column is preserved. Normalizes to kg CO2e/kWh.
   
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # country
    cname = next((c for c in df.columns if "country" in c or c in ("name", "nation")), None)
    if cname is None:
        raise ValueError("Could not find a 'country' column.")
    # year (optional)
    yname = next((c for c in df.columns if "year" in c), None)
    # intensity
    cand = [c for c in df.columns if "intensity" in c]
    if not cand:
        cand = [c for c in df.columns if ("co2" in c and "kwh" in c)]
    if not cand:
        cand = [c for c in df.columns if (("kg" in c or "g" in c) and "kwh" in c)]
    if not cand:
        raise ValueError("No intensity column found (looked for 'intensity' or '*co2* & *kwh*').")
    icol = cand[0]

    use_cols = [cname, icol] + ([yname] if yname else [])
    out = df[use_cols].dropna().copy()
    out.columns = ["country", "intensity_raw"] + (["year"] if yname else [])
    out["intensity_raw"] = pd.to_numeric(out["intensity_raw"], errors="coerce")
    out = out.dropna(subset=["intensity_raw"])

    # auto-detect g/kWh vs kg/kWh
    median = float(np.nanmedian(out["intensity_raw"].values))
    if median > 10.0:
        out["kg_per_kwh"] = out["intensity_raw"] / 1000.0
    else:
        out["kg_per_kwh"] = out["intensity_raw"]

    out["country"] = out["country"].astype(str).str.strip().str.title()

    if yname and year_select == "latest":
        out = out.sort_values(["country", "year"]).groupby("country", as_index=False).tail(1)

    cols = ["country", "kg_per_kwh"] + (["year"] if yname else [])
    return out[cols].drop_duplicates()

def compute_emissions(perf_df: pd.DataFrame,
                      co2_df: pd.DataFrame,
                      power_watts: float,
                      pue: float) -> pd.DataFrame:
    perf = perf_df.copy()
    perf["energy_kwh"] = (perf["runtime_sec"] * power_watts) / 3.6e6
    perf["energy_kwh"] *= pue

    co2 = co2_df.rename(columns={"kg_per_kwh": "kgco2_per_kwh"}).copy()
    perf["_k"] = 1
    co2["_k"] = 1
    joined = perf.merge(co2, on="_k").drop(columns="_k")
    joined["emissions_kgCO2"] = joined["energy_kwh"] * joined["kgco2_per_kwh"]
    return joined

def build_worstcase_country_table(perf_df: pd.DataFrame,
                                  co2_df: pd.DataFrame,
                                  power_watts: float,
                                  pue: float) -> Tuple[pd.DataFrame, dict]:
    dim_heavy = int(perf_df["dimension"].max())
    mask = perf_df["dimension"] == dim_heavy
    mean_runtime_s = float(perf_df.loc[mask, "runtime_sec"].mean())
    energy_kwh = mean_runtime_s * power_watts / 3.6e6 * pue

    tbl = co2_df.copy()
    if "year" not in tbl.columns:
        tbl["year"] = pd.NA

    out = pd.DataFrame({
        "Country": tbl["country"].astype(str),
        "Year": tbl["year"],
        "Intensity": tbl["kg_per_kwh"],  # kg CO2e/kWh
        "kWh": energy_kwh,               # same kWh across countries for the worst case
    })
    out["kgCO2e"] = out["Intensity"] * out["kWh"]
    out = out.sort_values("kgCO2e", ascending=False).reset_index(drop=True)

    meta = {
        "dimension_heavy": dim_heavy,
        "mean_runtime_s": mean_runtime_s,
        "power_watts": power_watts,
        "pue": pue,
        "kWh_used": energy_kwh,
    }
    return out, meta

# ------------********-------------
# ------------********------------- Plots: performance / reliability / scalability / carbon
# ------------********-------------

# ------------********------------- Performance
def plot_perf_runtime_vs_dimension(perf_df: pd.DataFrame, out_path: str) -> None:
    sub = perf_df.groupby("dimension", as_index=False)["runtime_sec"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["dimension"], sub["runtime_sec"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Performance: runtime vs dimension (SWAP test)")
    save_fig(fig, out_path)

def plot_perf_memory_vs_dimension(perf_df: pd.DataFrame, out_path: str) -> None:
    sub = perf_df.groupby("dimension", as_index=False)["peak_mem_mb"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["dimension"], sub["peak_mem_mb"], marker="s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Peak allocated (MB, tracemalloc)")
    ax.set_title("Performance: memory vs dimension (SWAP test)")
    save_fig(fig, out_path)

# ------------********------------- Reliability 
def plot_rel_error_vs_dimension(rel_df: pd.DataFrame, out_path: str) -> None:
    if rel_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No reliability data", ha="center", va="center")
        ax.axis("off")
        save_fig(fig, out_path)
        return
    g = rel_df.groupby("dimension", as_index=False)["abs_error"].agg(["mean", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g["dimension"], g["mean"], marker="o", label="mean abs error")
    ax.plot(g["dimension"], g["max"], marker="s", label="max abs error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Absolute error")
    ax.set_title("Reliability: SWAP estimate vs ground truth")
    ax.legend()
    save_fig(fig, out_path)

def plot_rel_error_histogram(rel_df: pd.DataFrame, out_path: str) -> None:
    if rel_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No reliability data", ha="center", va="center")
        ax.axis("off")
        save_fig(fig, out_path)
        return
    d_max = int(rel_df["dimension"].max())
    sub = rel_df[rel_df["dimension"] == d_max]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(sub["abs_error"].values, bins=30)
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Count")
    ax.set_title(f"Reliability: error distribution at dimension {d_max}")
    save_fig(fig, out_path)

def plot_rel_error_boxplot_by_dimension(rel_df: pd.DataFrame, out_path: str) -> None:
    if rel_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No reliability data", ha="center", va="center")
        ax.axis("off")
        save_fig(fig, out_path)
        return
    dims_sorted = sorted(rel_df["dimension"].unique())
    data = [rel_df[rel_df["dimension"] == d]["abs_error"].values for d in dims_sorted]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=[str(d) for d in dims_sorted], showfliers=False)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Absolute error")
    ax.set_title("Reliability: error boxplot by dimension")
    save_fig(fig, out_path)

# ------------********------------- Scalability 
def plot_pairs_scaling(perf_pairs_df: pd.DataFrame, out_path: str) -> None:
    sub = perf_pairs_df.groupby("pairs", as_index=False)["runtime_sec"].mean().sort_values("pairs")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["pairs"], sub["runtime_sec"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of vector pairs")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Scalability: runtime vs number of pairs (largest dimension)")
    save_fig(fig, out_path)

def plot_perf_runtime_vs_dimension_loglog(perf_df: pd.DataFrame, out_path: str) -> None:
    sub = perf_df.groupby("dimension", as_index=False)["runtime_sec"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["dimension"], sub["runtime_sec"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Scalability: runtime vs dimension (log-log)")
    save_fig(fig, out_path)

def plot_runtime_per_pair_vs_pairs(perf_pairs_df: pd.DataFrame, out_path: str) -> None:
    sub = perf_pairs_df.groupby("pairs", as_index=False)["runtime_sec"].mean().sort_values("pairs")
    sub["runtime_per_pair"] = sub["runtime_sec"] / sub["pairs"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["pairs"], sub["runtime_per_pair"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Number of vector pairs")
    ax.set_ylabel("Runtime per pair (s/pair)")
    ax.set_title("Scalability: runtime per pair vs pairs (largest dimension)")
    save_fig(fig, out_path)

# ------------********------------- Carbon 
def plot_carbon_distribution(worst_table: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(worst_table["kgCO2e"].values, bins=30)
    ax.set_xlabel("kg CO2e")
    ax.set_title("Carbon: Emission distribution (worst-case scenario)")
    save_fig(fig, out_path)

def plot_carbon_topN(worst_table: pd.DataFrame, N: int, out_path: str) -> None:
    s = worst_table.sort_values("kgCO2e", ascending=False).head(N)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(s["Country"], s["kgCO2e"])
    ax.invert_yaxis()
    ax.set_xlabel("kg CO2e")
    ax.set_title(f"Carbon: Top {N} countries (worst-case scenario)")
    save_fig(fig, out_path)

def plot_carbon_median_vs_dimension(carbon_df: pd.DataFrame, out_path: str) -> None:
    med = carbon_df.groupby("dimension")["emissions_kgCO2"].median().reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(med["dimension"], med["emissions_kgCO2"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Median emissions across countries (kg CO2e)")
    ax.set_title("Median emissions vs dimension")
    save_fig(fig, out_path)

# ------------********-------------
# ------------********------------- Benchmark drivers
# ------------********-------------

def run_performance(dims: List[int], pairs: int, repeats: int,
                    distribution: str, enforce_norm: bool,
                    seed: int, workers: int, shots: int) -> pd.DataFrame:
    tasks = [(d, pairs, distribution, enforce_norm, r, seed, shots) for d in dims for r in range(repeats)]
    rows = []
    if workers > 1:
        workers = min(workers, cpu_count())
        with Pool(processes=workers) as pool:
            for row in pool.imap_unordered(_perf_task, tasks):
                rows.append(row)
    else:
        for t in tasks:
            rows.append(_perf_task(t))
    df = pd.DataFrame(rows)
    return df

def run_scalability_pairs(dim_heavy: int, base_pairs: int, distribution: str,
                          enforce_norm: bool, seed: int, multipliers: List[float],
                          repeats: int, workers: int, shots: int) -> pd.DataFrame:

    #Micro-bench at the largest dimension: runtime vs number of pairs."""
    pgrid = sorted(set(max(1, int(round(base_pairs * m))) for m in multipliers))
    tasks = [(dim_heavy, p, distribution, enforce_norm, r, seed, shots) for p in pgrid for r in range(repeats)]
    rows = []
    if workers > 1:
        workers = min(workers, cpu_count())
        with Pool(processes=workers) as pool:
            for row in pool.imap_unordered(_perf_task, tasks):
                rows.append(row)
    else:
        for t in tasks:
            rows.append(_perf_task(t))
    df = pd.DataFrame(rows)
    df["is_pairs_scaling"] = True
    return df

# ------------********-------------
# ------------********------------- Main
# ------------********-------------

def main():
    parser = argparse.ArgumentParser(description="Ideal (noiseless) SWAP test benchmarks (quantum).")

    # Problem size & data (same shape as classical CLI)
    parser.add_argument("--dim-min", type=int, default=128)
    parser.add_argument("--dim-max", type=int, default=4096)
    parser.add_argument("--pairs", type=int, default=1000, help="Number of vector pairs per dimension.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--distribution", choices=["normal", "uniform"], default="normal")
    parser.add_argument("--normalize", action="store_true",
                        help="(kept for parity) If set, vectors are explicitly L2-normalized before use.")

    # Quantum sampling
    parser.add_argument("--shots", type=int, default=4096,
                        help="If >0, simulate measurement sampling; if 0, use ideal probabilities only.")

    # Carbon & file inputs 
    parser.add_argument("--excel", type=str, default=None,
                        help='CO2 file name or path; e.g. "Filtered CO2 intensity 236 Countries.xlsx"')
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", choices=["latest", "all"], default="latest")

    # Execution/output
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--combine", action="store_true", help="Accepted for compatibility; no additional effect.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    # Dims (powers of two)
    dims = dims_from_range(args.dim_min, args.dim_max)
    if not dims:
        print("No dimensions found in the requested range.")
        sys.exit(1)

    # Resolve CO2 file on Desktop
    co2_path = resolve_desktop_file(
        args.excel,
        default_basenames=["Filtered CO2 intensity 236 Countries", "countries_co2", "co2_countries"]
    )
    if co2_path:
        print(f"[INFO] Using CO2 file: {co2_path}")
    else:
        print("[WARN] CO2 file not found on Desktop. Carbon outputs will include a placeholder unless provided.")

    # Output structure
    root = os.path.abspath(args.outdir)
    perf_dir = os.path.join(root, "Performance")
    scal_dir = os.path.join(root, "Scalability")
    reli_dir = os.path.join(root, "Reliability")
    carb_dir = os.path.join(root, "Carbon")
    for d in (perf_dir, scal_dir, reli_dir, carb_dir):
        ensure_dir(d)

    enforce_norm = True  # amplitude encoding requires normalized states (keep normalize flag only for parity)
    if args.normalize:
        pass  # no-op; we always enforce normalization in cosine_batch

   # ------------********-------------
    # ------------********------------- PERFORMANCE (Excel + 2 plots)
  # ------------********-------------
    print("[INFO] Running performance bench (SWAP) ...")
    perf_df = run_performance(
        dims=dims, pairs=args.pairs, repeats=args.repeats,
        distribution=args.distribution, enforce_norm=enforce_norm,
        seed=args.seed, workers=max(1, int(args.workers)), shots=args.shots
    )
    perf_sum = perf_df.groupby(["dimension"], as_index=False).agg(
        pairs=("pairs", "max"),
        mean_runtime_sec=("runtime_sec", "mean"),
        std_runtime_sec=("runtime_sec", "std"),
        mean_peak_mem_mb=("peak_mem_mb", "mean"),
        std_peak_mem_mb=("peak_mem_mb", "std"),
        mean_p0=("mean_p0", "mean"),
        mean_cosine_sq=("mean_cosine_sq", "mean"),
        mean_swap_estimate=("mean_swap_estimate", "mean"),
    )
    write_excel(
        {"raw": perf_df.sort_values(["dimension", "repeat"]),
         "summary": perf_sum.sort_values("dimension")},
        os.path.join(perf_dir, "performance.xlsx")
    )
    plot_perf_runtime_vs_dimension(perf_df, os.path.join(perf_dir, "runtime_vs_dimension.png"))
    plot_perf_memory_vs_dimension(perf_df, os.path.join(perf_dir, "memory_vs_dimension.png"))

   # ------------********-------------
   # ------------********-------------RELIABILITY (Excel + 3 plots)
   # ------------********-------------
    print("[INFO] Running reliability check ...")
    pairs_for_rel = max(1, min(10, args.pairs))  # small subset to keep runtime modest
    rel_raw = run_reliability(
        dims=dims, pairs_for_rel=pairs_for_rel, distribution=args.distribution,
        enforce_norm=enforce_norm, seed=args.seed, repeats=args.repeats,
        workers=max(1, int(args.workers)), shots=args.shots
    )

    if rel_raw.empty:
        rel_sum = pd.DataFrame({"dimension": [], "mean_abs_error": [], "max_abs_error": [], "n": []})
    else:
        g = rel_raw.groupby("dimension", as_index=False)
        rel_sum = g.agg(
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
            n=("abs_error", "count"),
        )

    write_excel(
        {"errors_raw": rel_raw.sort_values(["dimension", "repeat", "pair_index"]),
         "errors_summary": rel_sum.sort_values("dimension")},
        os.path.join(reli_dir, "reliability.xlsx")
    )
    plot_rel_error_vs_dimension(rel_raw, os.path.join(reli_dir, "error_vs_dimension.png"))
    plot_rel_error_histogram(rel_raw, os.path.join(reli_dir, "error_histogram_at_largest_dimension.png"))
    plot_rel_error_boxplot_by_dimension(rel_raw, os.path.join(reli_dir, "error_boxplot_by_dimension.png"))

    # ------------********-------------
    # ------------********------------- SCALABILITY (Excel + 3 plots)
    # ------------********-------------
    print("[INFO] Running scalability micro-bench (pairs at largest dimension) ...")
    dim_heavy = max(dims)
    try:
        mults = [float(x.strip()) for x in "0.25,0.5,1,2".split(",")]
    except Exception:
        mults = [0.5, 1.0, 2.0]
    pairs_scal_df = run_scalability_pairs(
        dim_heavy=dim_heavy, base_pairs=args.pairs, distribution=args.distribution,
        enforce_norm=enforce_norm, seed=args.seed, multipliers=mults,
        repeats=args.repeats, workers=max(1, int(args.workers)), shots=args.shots
    )
    scaling_fit = pd.DataFrame()
    if len(dims) >= 2:
        # log-log fit: runtime ~ dimension^alpha (using perf_df means)
        sub = perf_df.groupby("dimension", as_index=False)["runtime_sec"].mean()
        x = np.log(sub["dimension"].values + 1e-16)
        y = np.log(sub["runtime_sec"].values + 1e-16)
        alpha, _ = np.polyfit(x, y, 1)
        scaling_fit = pd.DataFrame({"fit_parameter": ["alpha_dim_scaling"], "estimate": [alpha]})
    write_excel(
        {"perf_raw": perf_df.sort_values(["dimension", "repeat"]),
         "pairs_scaling_raw": pairs_scal_df.sort_values(["pairs", "repeat"]),
         "scaling_fit": scaling_fit},
        os.path.join(scal_dir, "scalability.xlsx")
    )
    plot_pairs_scaling(pairs_scal_df, os.path.join(scal_dir, "runtime_vs_pairs_at_largest_dimension.png"))
    plot_perf_runtime_vs_dimension_loglog(perf_df, os.path.join(scal_dir, "runtime_vs_dimension_loglog.png"))
    plot_runtime_per_pair_vs_pairs(pairs_scal_df, os.path.join(scal_dir, "runtime_per_pair_vs_pairs.png"))

    # ------------********-------------
    # ------------********------------- CARBON (Excel + 3 plots) — uses ONLY Performance runtimes
    # ------------********-------------
    if co2_path and os.path.isfile(co2_path):
        print("[INFO] Computing carbon footprints ...")
        co2_df = load_country_co2(co2_path, year_select=args.year_select)
        carbon_df = compute_emissions(perf_df, co2_df, power_watts=args.device_power_watts, pue=args.pue)
        worst_table, meta = build_worstcase_country_table(perf_df, co2_df, power_watts=args.device_power_watts, pue=args.pue)

        write_excel(
            {
                "co2_by_country": (co2_df.rename(columns={"kg_per_kwh": "kgco2_per_kwh"})
                                   if "kg_per_kwh" in co2_df.columns else co2_df),
                "emissions_all": carbon_df.sort_values(["dimension", "country"]),
                "worst_case_table": worst_table,  # Country | Year | Intensity | kWh | kgCO2e
                "worst_case_metadata": pd.DataFrame([meta]),
            },
            os.path.join(carb_dir, "carbon_footprint.xlsx")
        )
        plot_carbon_distribution(worst_table, os.path.join(carb_dir, "carbon_distribution.png"))
        plot_carbon_topN(worst_table, 15, os.path.join(carb_dir, "carbon_top15.png"))
        plot_carbon_median_vs_dimension(carbon_df, os.path.join(carb_dir, "median_emissions_vs_dimension.png"))
    else:
        placeholder = pd.DataFrame({
            "note": ["CO2 file not provided. Carbon workbook/plots were skipped."],
            "hint": ["Run again with --excel \"Filtered CO2 intensity 236 Countries.xlsx\""]
        })
        write_excel({"info": placeholder}, os.path.join(carb_dir, "carbon_placeholder.xlsx"))

    print("[DONE] All benchmarks complete.")
    print(f"[OUT] Root: {root}")
    print("      - Performance\\performance.xlsx + 2 PNGs")
    print("      - Scalability\\scalability.xlsx + 3 PNGs")
    print("      - Reliability\\reliability.xlsx + 3 PNGs")
    print("      - Carbon\\carbon_footprint.xlsx + 3 PNGs (if CO2 file found)")

if __name__ == "__main__":
    main()
