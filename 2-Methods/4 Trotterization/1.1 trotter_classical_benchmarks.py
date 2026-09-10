#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# trotter_classical_benchmarks.py
# Classical (CPU) evaluation of first-order Trotterization with four metrics:
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

import os
import sys
import time
import math
import warnings
import argparse
import tracemalloc
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ------------********------------
# Utility
# ------------********------------
#These following utility functions handle filesystem setup, robust result export, and figure saving.
#   Excel output is with a CSV fallback for error resilience.
#       Directory creation and plotting helpers ensure clean output generation and resource cleanup.

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_excel(df_dict: Dict[str, pd.DataFrame], out_xlsx: str) -> None:
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            for sheet, df in df_dict.items():
                sheet_name = (sheet[:31] or "Sheet1")
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        print(f"[OK] Wrote Excel: {out_xlsx}")
    except Exception as e:
        print(f"[WARN] Could not write Excel ({e}). Falling back to CSVs.")
        base = os.path.splitext(out_xlsx)[0]
        for sheet, df in df_dict.items():
            csv_path = f"{base}__{sheet}.csv"
            df.to_csv(csv_path, index=False)
            print(f"[OK] Wrote CSV fallback: {csv_path}")

def save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved plot: {path}")

def resolve_desktop_file(user_arg: str | None,
                         default_candidates: List[str]) -> str | None:
   
    #Resolve a file path. If user_arg is given:
    #  - If it's an existing path: return it.
    #  - If it has no directory component, search Desktop for:
    #           exact name, then add common extensions (.xlsx, .xls, .csv).
    #If user_arg is None, try default_candidates on Desktop.
 
    desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")

    def exists(p): return os.path.isfile(p)

    if user_arg:
        # Strip quotes/spaces
        base = user_arg.strip().strip('"').strip("'")
        # If absolute or relative path exists, use it
        if exists(base):
            return os.path.abspath(base)
        # If given without directory, look on Desktop
        candidate = os.path.join(desktop, base)
        if exists(candidate):
            return os.path.abspath(candidate)
        # Try with extensions
        exts = [".xlsx", ".xls", ".csv"]
        if not os.path.splitext(base)[1]:  # missing extension
            for ext in exts:
                candidate = os.path.join(desktop, base + ext)
                if exists(candidate):
                    return os.path.abspath(candidate)
        # Last try: case-insensitive search on Desktop
        head, tail = os.path.split(base)
        tail_lower = tail.lower()
        for fname in os.listdir(desktop):
            if fname.lower() == tail_lower or any(
                (not os.path.splitext(tail)[1]) and fname.lower() == (tail_lower + ext)
                for ext in [".xlsx", ".xls", ".csv"]
            ):
                cand = os.path.join(desktop, fname)
                if exists(cand):
                    return os.path.abspath(cand)
        return None

    # No user_arg: try defaults
    for cand in default_candidates:
        if exists(cand):
            return os.path.abspath(cand)
    return None

# ------------********------------
# ------------********------------ Linear algebra primitives
# ------------********------------

#This function defines the four standard Pauli matrices used in quantum mechanics.
#Each matrix is constructed explicitly as a 2×2 complex NumPy array.
#The identity and X, Y, Z operators are returned together for convenient reuse.

def paulis() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z

#This helper computes the Kronecker product of a list of matrices in sequence.
#It incrementally builds the full tensor product needed for multi-qubit operators.
def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

#This function constructs a full n-qubit operator that applies a given single-qubit operation at position i.
#It embeds the operator into the larger Hilbert space using identities and Kronecker products.
def single_qubit_op(n: int, i: int, op: np.ndarray) -> np.ndarray:
    I, _, _, _ = paulis()
    mats = [I] * n
    mats[i] = op
    return kron_all(mats)

#This helper builds a two-qubit operator acting on adjacent qubits in an n-qubit system.
#It enforces a nearest-neighbor constraint to simplify operator construction.
#The specified single-qubit operators are embedded using identities and Kronecker products.
def two_qubit_op(n: int, i: int, j: int, op_i: np.ndarray, op_j: np.ndarray) -> np.ndarray:
    assert j == i + 1, "This helper expects nearest-neighbour pairs (j = i+1)."
    I, _, _, _ = paulis()
    mats = [I] * n
    mats[i] = op_i
    mats[j] = op_j
    return kron_all(mats)

#This following function computes the matrix exponential of a Hermitian operator using eigendecomposition.
#Eigenvalues are exponentiated to form phase factors, ensuring numerical stability.
#The result reconstructs the unitary evolution operator via the eigenbasis.
def expm_hermitian(H: np.ndarray, dt: float) -> np.ndarray:
    w, V = np.linalg.eigh(H)
    phase = np.exp(-1j * dt * w)
    return (V * phase) @ V.conj().T

# ------------********------------
# ------------********------------ Heisenberg model
# ------------********------------

#This function constructs the Heisenberg spin-chain Hamiltonian and its Trotter-friendly components.
#   It separates nearest-neighbor interactions into even and odd bond groups and adds a transverse field term.
#   Full-system operators are built via Kronecker products to form exact matrix representations.
#   The total Hamiltonian and grouped sub-Hamiltonians are returned for simulation and decomposition.

def build_heisenberg_groups(n: int, J: float = 1.0, h: float = 0.5) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    I, X, Y, Z = paulis()
    dim = 2 ** n

    H_even = np.zeros((dim, dim), dtype=complex)
    H_odd = np.zeros((dim, dim), dtype=complex)
    H_field = np.zeros((dim, dim), dtype=complex)

    for i in range(0, n - 1, 2):
        H_even += (two_qubit_op(n, i, i + 1, X, X)
                   + two_qubit_op(n, i, i + 1, Y, Y)
                   + two_qubit_op(n, i, i + 1, Z, Z))
    for i in range(1, n - 1, 2):
        H_odd += (two_qubit_op(n, i, i + 1, X, X)
                  + two_qubit_op(n, i, i + 1, Y, Y)
                  + two_qubit_op(n, i, i + 1, Z, Z))
    for i in range(n):
        H_field += single_qubit_op(n, i, Z) * h

    H_total = J * (H_even + H_odd) + H_field
    groups = {"even": J * H_even, "odd": J * H_odd, "field": H_field}
    return H_total, groups

# ------------********------------
# ------------********------------ Simulation
# ------------********------------

#These routines compare exact and approximate quantum time evolution methods.
#   The exact simulator applies the full matrix exponential to evolve the initial state.

def simulate_exact(H: np.ndarray, psi0: np.ndarray, T: float) -> np.ndarray:
    U = expm_hermitian(H, T)
    return U @ psi0

#   The Trotter simulator approximates evolution by repeatedly applying grouped Hamiltonian exponentials.
def simulate_trotter(groups: Dict[str, np.ndarray], m_steps: int, T: float, psi0: np.ndarray) -> np.ndarray:
    dt = T / m_steps
    E_even = expm_hermitian(groups["even"], dt)
    E_odd = expm_hermitian(groups["odd"], dt)
    E_field = expm_hermitian(groups["field"], dt)

    psi = psi0.copy()
    for _ in range(m_steps):
        psi = E_field @ (E_odd @ (E_even @ psi))
    return psi

#   A helper computes the ℓ2 norm of the difference to quantify approximation error.
def l2_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# ------------********------------
# ------------********------------ Plot helpers
# ------------********------------

#These plotting functions visualize performance and accuracy of Trotterized simulations.
#They generate log-scaled runtime, memory, and error trends across increasing Trotter steps and qubit counts.
#A heatmap summarizes error behavior over the full parameter grid for quick comparison.
#All figures are saved using a shared helper to ensure consistent layout and formatting.

def plot_runtime_vs_steps(perf_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, sub in perf_df.groupby("n_qubits"):
        sub = sub.sort_values("trotter_steps")
        ax.plot(sub["trotter_steps"], sub["runtime_sec"], marker="o", label=f"n={n}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Trotter steps (m)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs Trotter steps")
    ax.legend()
    save_fig(fig, out_path)

def plot_memory_vs_steps(perf_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, sub in perf_df.groupby("n_qubits"):
        sub = sub.sort_values("trotter_steps")
        ax.plot(sub["trotter_steps"], sub["peak_mem_mb"], marker="s", label=f"n={n}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Trotter steps (m)")
    ax.set_ylabel("Peak allocated (MB, tracemalloc)")
    ax.set_title("Memory vs Trotter steps")
    ax.legend()
    save_fig(fig, out_path)

def plot_error_vs_steps(rel_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, sub in rel_df.groupby("n_qubits"):
        sub = sub.sort_values("trotter_steps")
        ax.plot(sub["trotter_steps"], sub["mean_error"], marker="o", label=f"n={n}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Trotter steps (m)")
    ax.set_ylabel("Mean L2 error (log scale)")
    ax.set_title("Reliability: Error vs steps")
    ax.legend()
    save_fig(fig, out_path)

def plot_error_heatmap(raw_err: pd.DataFrame, out_path: str) -> None:
    piv = raw_err.pivot_table(index="n_qubits", columns="trotter_steps", values="error", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.log10(piv.values + 1e-16), aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Trotter steps (m)")
    ax.set_ylabel("n_qubits")
    ax.set_title("log10(Error): n vs m")
    fig.colorbar(im, ax=ax, label="log10(L2 error)")
    save_fig(fig, out_path)

def plot_scaling_runtime_vs_dim(perf_df: pd.DataFrame, m_for_plot: int, out_path: str) -> None:
    avail = sorted(perf_df["trotter_steps"].unique())
    m = min(avail, key=lambda x: abs(x - m_for_plot))
    sub = perf_df[perf_df["trotter_steps"] == m].copy()
    sub["dimension"] = 2 ** sub["n_qubits"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["dimension"], sub["runtime_sec"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hilbert space dimension (2^n)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Scalability: runtime vs dimension (m≈{m})")
    save_fig(fig, out_path)

def plot_scaling_runtime_vs_steps(perf_df: pd.DataFrame, n_for_plot: int, out_path: str) -> None:
    sub = perf_df[perf_df["n_qubits"] == n_for_plot].copy().sort_values("trotter_steps")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["trotter_steps"], sub["runtime_sec"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Trotter steps (m)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Scalability: runtime vs steps (n={n_for_plot})")
    save_fig(fig, out_path)

# Carbon visuals 
def plot_carbon_distribution(worst_table: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.hist(worst_table["kgCO2e"].values, bins=30)
    ax.set_xlabel("kg CO2e")
    ax.set_title("Carbon: Emission distribution")
    save_fig(fig, out_path)

def plot_carbon_topN(worst_table: pd.DataFrame, N: int, out_path: str) -> None:
    s = worst_table.sort_values("kgCO2e", ascending=False).head(N)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(s["Country"], s["kgCO2e"])
    ax.invert_yaxis()
    ax.set_xlabel("kg CO2e")
    ax.set_title(f"Carbon: Top {N} countries (kgCO2e)")
    save_fig(fig, out_path)

def plot_carbon_median_vs_steps(country_emissions: pd.DataFrame, n_for_plot: int, out_path: str) -> None:
    sub = country_emissions[country_emissions["n_qubits"] == n_for_plot]
    med = sub.groupby("trotter_steps")["emissions_kgCO2"].median().reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(med["trotter_steps"], med["emissions_kgCO2"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Trotter steps (m)")
    ax.set_ylabel("Median emissions across countries (kg CO2e)")
    ax.set_title(f"Median emissions vs steps (n={n_for_plot})")
    save_fig(fig, out_path)

# ------------********------------
# ------------********------------ Carbon logic
# ------------********------------

def load_country_co2(path: str, year_select: str = "latest") -> pd.DataFrame:
    """
    Accepts CSV/XLSX with columns:
      - 'country' (string; case-insensitive)
      - 'year' (optional)
      - either 'intensity' or any header containing both 'co2' and 'kwh', or ('kg'|'g') and 'kwh'
    Returns: columns ['country','kg_per_kwh'] and (optionally) 'year'.
    If year_select == 'latest', keeps the most recent row per country when 'year' is present.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CO2 file not found: {path}")

    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    # normalize headers to lowercase
    df.columns = [str(c).strip().lower() for c in df.columns]

    # locate country
    cname = next((c for c in df.columns if "country" in c or c in ("name", "nation")), None)
    if cname is None:
        raise ValueError("Could not find a 'country' column.")

    # optional year
    yname = next((c for c in df.columns if "year" in c), None)

    # locate intensity
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

    # coerce to numeric intensity
    out["intensity_raw"] = pd.to_numeric(out["intensity_raw"], errors="coerce")
    out = out.dropna(subset=["intensity_raw"])

    # convert to kg/kWh (auto-detect g/kWh)
    if np.nanmedian(out["intensity_raw"].values) > 10.0:
        out["kg_per_kwh"] = out["intensity_raw"] / 1000.0
    else:
        out["kg_per_kwh"] = out["intensity_raw"]

    out["country"] = out["country"].astype(str).str.strip().str.title()

    # year selection (latest per country)
    if yname and year_select == "latest":
        # keep the max year per country
        out = out.sort_values(["country", "year"]).groupby("country", as_index=False).tail(1)

    cols = ["country", "kg_per_kwh"] + (["year"] if yname else [])
    return out[cols].drop_duplicates()

#This function estimates carbon emissions by combining performance runtimes with carbon intensity data.
#It converts runtime into energy consumption using power draw and PUE adjustments.
#A cross-join applies each country’s intensity to all runs to compute resulting CO₂ emissions.

def compute_emissions(perf_df: pd.DataFrame,
                      co2_df: pd.DataFrame,
                      power_watts: float,
                      pue: float) -> pd.DataFrame:
    perf = perf_df.copy()
    perf["energy_kwh"] = (perf["runtime_sec"] * power_watts) / 3.6e6
    perf["energy_kwh"] *= pue

    co2 = co2_df.rename(columns={"kg_per_kwh": "kgco2_per_kwh"}).copy()
    perf["_key"] = 1
    co2["_key"] = 1
    joined = perf.merge(co2, on="_key").drop(columns="_key")
    joined["emissions_kgCO2"] = joined["energy_kwh"] * joined["kgco2_per_kwh"]
    return joined


#This function estimates worst-case carbon emissions based on the heaviest simulation configuration.
#       It identifies the largest problem size and longest runtime to compute total energy consumption.
#       Country-level carbon intensity data is combined to estimate emissions per country.
#       The function returns a ranked emissions table along with metadata describing the scenario.

def build_worstcase_country_table(perf_df: pd.DataFrame,
                                  co2_df: pd.DataFrame,
                                  power_watts: float,
                                  pue: float) -> Tuple[pd.DataFrame, dict]:
    n_heavy = int(perf_df["n_qubits"].max())
    m_heavy = int(perf_df["trotter_steps"].max())
    mask = (perf_df["n_qubits"] == n_heavy) & (perf_df["trotter_steps"] == m_heavy)
    mean_runtime_s = float(perf_df.loc[mask, "runtime_sec"].mean())
    energy_kwh = mean_runtime_s * power_watts / 3.6e6 * pue

    tbl = co2_df.copy()
    if "year" not in tbl.columns:
        tbl["year"] = pd.NA

    tbl_out = pd.DataFrame({
        "Country": tbl["country"].astype(str),
        "Year": tbl["year"],
        "Intensity": tbl["kg_per_kwh"],   # kg CO2e / kWh
        "kWh": energy_kwh,                # constant across countries for this scenario
    })
    tbl_out["kgCO2e"] = tbl_out["Intensity"] * tbl_out["kWh"]
    tbl_out = tbl_out.sort_values("kgCO2e", ascending=False).reset_index(drop=True)

    meta = {
        "n_heavy": n_heavy,
        "m_heavy": m_heavy,
        "mean_runtime_s": mean_runtime_s,
        "power_watts": power_watts,
        "pue": pue,
        "kWh_used": energy_kwh,
    }
    return tbl_out, meta

# ------------********------------
# ------------********------------ Benchmark driver 
# ------------********------------

#This worker function executes a single Trotter simulation task for a given parameter set.
#   It builds the Heisenberg Hamiltonian, prepares a random initial quantum state, and runs both exact and approximate evolution.
#           Runtime and peak memory usage are measured using high-resolution timers and tracemalloc.
#                   The function returns separate dictionaries for performance metrics and numerical error analysis.

def _one_task(args):
    n, m, r, seed_base, J, h, T = args
    rng = np.random.default_rng(seed_base + (n * 1009) + (m * 9176) + r)
    dim = 2 ** n
    H, groups = build_heisenberg_groups(n, J=J, h=h)

    psi0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi0 = psi0 / np.linalg.norm(psi0)

    tracemalloc.start()
    t0 = time.perf_counter()
    psi_t = simulate_trotter(groups, m_steps=m, T=T, psi0=psi0)
    psi_e = simulate_exact(H, psi0, T=T)
    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    err = l2_error(psi_t, psi_e)
    return {
        "n_qubits": n,
        "dimension": dim,
        "trotter_steps": m,
        "repeat": r,
        "runtime_sec": runtime,
        "peak_mem_mb": peak / (1024 ** 2),
    }, {
        "n_qubits": n,
        "dimension": dim,
        "trotter_steps": m,
        "repeat": r,
        "error": err,
    }

#This function orchestrates the full benchmark sweep across qubit counts, Trotter steps, and repetitions.
#It assembles all experiment tasks and supports parallel execution using multiple worker processes.
#Performance and error metrics from each task are collected into separate result tables.
#The aggregated results are returned as pandas DataFrames for downstream analysis and plotting.

def run_benchmarks(n_list: List[int],
                   m_list: List[int],
                   repeats: int,
                   J: float,
                   h: float,
                   T: float,
                   rng_seed: int,
                   workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [(n, m, r, rng_seed, J, h, T)
             for n in n_list
             for m in m_list
             for r in range(repeats)]
    perf_rows, err_rows = [], []

    if workers and workers > 1:
        workers = min(workers, cpu_count())
        with Pool(processes=workers) as pool:
            for perf_row, err_row in pool.imap_unordered(_one_task, tasks):
                perf_rows.append(perf_row)
                err_rows.append(err_row)
    else:
        for t in tasks:
            perf_row, err_row = _one_task(t)
            perf_rows.append(perf_row)
            err_rows.append(err_row)

    return pd.DataFrame(perf_rows), pd.DataFrame(err_rows)

#This function aggregates raw error measurements to assess simulation reliability.
#Errors are grouped by system size and Trotter step count.
#Summary statistics such as mean, variance, extrema, and sample count are computed for analysis.

def summarize_reliability(err_raw: pd.DataFrame) -> pd.DataFrame:
    g = err_raw.groupby(["n_qubits", "trotter_steps"], as_index=False)
    return g.agg(
        mean_error=("error", "mean"),
        std_error=("error", "std"),
        max_error=("error", "max"),
        min_error=("error", "min"),
        num_runs=("error", "count"),
    )

def summarize_scalability(perf_df: pd.DataFrame) -> pd.DataFrame:
    m_target = perf_df["trotter_steps"].max()
    sub_dim = perf_df.loc[perf_df["trotter_steps"] == m_target].copy()
    sub_dim["dimension"] = 2 ** sub_dim["n_qubits"]
    sub_dim = sub_dim.groupby(["n_qubits", "dimension"], as_index=False)["runtime_sec"].mean()

    x = np.log(sub_dim["dimension"].values + 1e-16)
    y = np.log(sub_dim["runtime_sec"].values + 1e-16)
    alpha, _ = np.polyfit(x, y, 1)

    n_target = perf_df["n_qubits"].max()
    sub_steps = perf_df.loc[perf_df["n_qubits"] == n_target].copy()
    sub_steps = sub_steps.groupby(["trotter_steps"], as_index=False)["runtime_sec"].mean()
    xb = np.log(sub_steps["trotter_steps"].values + 1e-16)
    yb = np.log(sub_steps["runtime_sec"].values + 1e-16)
    beta, _ = np.polyfit(xb, yb, 1)

    return pd.DataFrame({
        "fit_parameter": ["alpha_dim_scaling", "beta_step_scaling"],
        "estimate": [alpha, beta],
        "note": [f"Fixed m≈{m_target}", f"Fixed n={n_target}"]
    })

# ------------********------------
# ------------********------------ Main
# ------------********------------

def main():
    parser = argparse.ArgumentParser(description="Classical evaluation of Trotterization with four metrics and per-metric outputs.")
    # Model/grid
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=6)
    parser.add_argument("--steps", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)

    # Carbon / file inputs (new names + aliases)
    parser.add_argument("--excel", type=str, default=None,
                        help="CO2 file name or path (e.g., 'Filtered CO2 intensity 236 Countries.xlsx').")
    parser.add_argument("--countries-co2-file", type=str, default=None,
                        help="Alias: path to countries' carbon intensity file.")
    parser.add_argument("--device-power-watts", type=float, default=None,
                        help="Average device power draw in Watts.")
    parser.add_argument("--power-watts", type=float, default=None,
                        help="Alias of --device-power-watts.")
    parser.add_argument("--pue", type=float, default=1.2,
                        help="Power Usage Effectiveness (>=1.0).")
    parser.add_argument("--year-select", choices=["latest", "all"], default="latest",
                        help="If 'latest', keep most recent year per country for carbon.")

    # Execution/output
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--combine", action="store_true",
                        help="Accepted for compatibility; no additional effect in this script.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for simulation (Windows-safe).")

    args = parser.parse_args()

    # Parse steps
    try:
        m_list = sorted(set(int(s.strip()) for s in args.steps.split(",") if s.strip()))
        m_list = [m for m in m_list if m > 0]
    except Exception:
        print("Invalid --steps; use e.g. '1,2,4,8,16,32'")
        sys.exit(1)

    # Qubits
    if args.n_max < args.n_min or args.n_min < 1:
        print("Invalid n range.")
        sys.exit(1)
    n_list = list(range(args.n_min, args.n_max + 1))

    # Power draw (resolve alias)
    power_watts = args.device_power_watts if args.device_power_watts is not None else (
        args.power_watts if args.power_watts is not None else 45.0
    )

    # Resolve CO2 file on Desktop
    desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
    defaults = [
        os.path.join(desktop, "Filtered CO2 intensity 236 Countries.xlsx"),
        os.path.join(desktop, "Filtered CO2 intensity 236 Countries.csv"),
        os.path.join(desktop, "countries_co2.xlsx"),
        os.path.join(desktop, "countries_co2.csv"),
    ]
    co2_path = resolve_desktop_file(args.excel or args.countries_co2_file, defaults)
    if co2_path is None:
        print("[WARN] CO2 file not found on Desktop. "
              "Pass --excel \"Filtered CO2 intensity 236 Countries.xlsx\" or a full path.")
    else:
        print(f"[INFO] Using CO2 file: {co2_path}")

    # Output structure
    root = os.path.abspath(args.outdir)
    perf_dir = os.path.join(root, "Performance")
    scal_dir = os.path.join(root, "Scalability")
    reli_dir = os.path.join(root, "Reliability")
    carb_dir = os.path.join(root, "Carbon")
    for d in (perf_dir, scal_dir, reli_dir, carb_dir):
        ensure_dir(d)

    print("[INFO] Running benchmarks ...")
    t_all0 = time.perf_counter()
    perf_df, err_raw = run_benchmarks(
        n_list=n_list,
        m_list=m_list,
        repeats=args.repeats,
        J=args.J,
        h=args.h,
        T=args.T,
        rng_seed=args.seed,
        workers=max(1, int(args.workers))
    )

# ------------********------------ PERFORMANCE

#This following block summarizes performance metrics across qubit counts and Trotter step sizes.
#   Mean and variability of runtime and memory usage are computed for each configuration.
#       Both raw data and aggregated summaries are exported to an Excel report.
#       Runtime and memory scaling plots are generated to visualize performance trends.

    perf_summary = perf_df.groupby(["n_qubits", "trotter_steps"], as_index=False).agg(
        mean_runtime_sec=("runtime_sec", "mean"),
        std_runtime_sec=("runtime_sec", "std"),
        mean_peak_mem_mb=("peak_mem_mb", "mean"),
        std_peak_mem_mb=("peak_mem_mb", "std")
    )
    write_excel(
        {"raw": perf_df.sort_values(["n_qubits", "trotter_steps", "repeat"]),
         "summary": perf_summary.sort_values(["n_qubits", "trotter_steps"])},
        os.path.join(perf_dir, "performance.xlsx")
    )
    plot_runtime_vs_steps(perf_df, os.path.join(perf_dir, "runtime_vs_steps.png"))
    plot_memory_vs_steps(perf_df, os.path.join(perf_dir, "memory_vs_steps.png"))

# ------------********------------ RELIABILITY
    rel_summary = summarize_reliability(err_raw)
    write_excel(
        {"errors_raw": err_raw.sort_values(["n_qubits", "trotter_steps", "repeat"]),
         "errors_summary": rel_summary.sort_values(["n_qubits", "trotter_steps"])},
        os.path.join(reli_dir, "reliability.xlsx")
    )
    plot_error_vs_steps(rel_summary, os.path.join(reli_dir, "error_vs_steps.png"))
    plot_error_heatmap(err_raw, os.path.join(reli_dir, "error_heatmap.png"))

# ------------********------------ SCALABILITY
    scaling_fit = summarize_scalability(perf_df)
    write_excel(
        {"perf_raw": perf_df.sort_values(["n_qubits", "trotter_steps", "repeat"]),
         "scaling_fit": scaling_fit},
        os.path.join(scal_dir, "scalability.xlsx")
    )
    plot_scaling_runtime_vs_dim(perf_df, m_for_plot=max(m_list),
                                out_path=os.path.join(scal_dir, "runtime_vs_dimension_loglog.png"))
    plot_scaling_runtime_vs_steps(perf_df, n_for_plot=max(n_list),
                                  out_path=os.path.join(scal_dir, "runtime_vs_steps_at_largest_n.png"))

# ------------********------------ CARBON (Performance-only)
    if co2_path and os.path.isfile(co2_path):
        co2_df = load_country_co2(co2_path, year_select=args.year_select)
        carbon_df = compute_emissions(perf_df, co2_df, power_watts=power_watts, pue=args.pue)
        worst_table, meta = build_worstcase_country_table(perf_df, co2_df, power_watts, args.pue)

        write_excel(
            {
                "co2_by_country": (co2_df.rename(columns={"kg_per_kwh": "kgco2_per_kwh"})
                                   if "kg_per_kwh" in co2_df.columns else co2_df),
                "emissions_all": carbon_df.sort_values(["n_qubits", "trotter_steps", "country"]),
                "worst_case_table": worst_table,
                "worst_case_metadata": pd.DataFrame([meta]),
            },
            os.path.join(carb_dir, "carbon_footprint.xlsx")
        )
        plot_carbon_distribution(worst_table, os.path.join(carb_dir, "carbon_distribution.png"))
        plot_carbon_topN(worst_table, 15, os.path.join(carb_dir, "carbon_top15.png"))
        plot_carbon_median_vs_steps(carbon_df, n_for_plot=meta["n_heavy"],
                                    out_path=os.path.join(carb_dir, "median_emissions_vs_steps.png"))
    else:
        placeholder = pd.DataFrame({
            "note": ["CO2 file not provided. Carbon workbook/plots were skipped."],
            "hint": ["Run again with --excel \"Filtered CO2 intensity 236 Countries.xlsx\""]
        })
        write_excel({"info": placeholder}, os.path.join(carb_dir, "carbon_placeholder.xlsx"))

    print(f"[DONE] All benchmarks completed in {time.perf_counter() - t_all0:.2f}s")
    print(f"[OUT] Folders created under: {root}")
    print("      - Performance\\performance.xlsx + 2 PNGs")
    print("      - Scalability\\scalability.xlsx + 2 PNGs")
    print("      - Reliability\\reliability.xlsx + 2 PNGs")
    print("      - Carbon\\carbon_footprint.xlsx (if CO2 provided) + PNGs:")
    print("          carbon_distribution.png, carbon_top15.png, median_emissions_vs_steps.png")

if __name__ == "__main__":
    main()
