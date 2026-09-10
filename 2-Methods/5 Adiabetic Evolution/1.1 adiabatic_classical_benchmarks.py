#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# adiabatic_classical_benchmarks.py
# Classical (CPU) evaluation of Adiabatic Evolution with four metrics:
# Performance, Scalability, Reliability, and Carbon footprint.
# CLI mirrors my previous runs (excel / device-power-watts / pue / year-select / combine / workers).
#
# Model: Transverse-Field Ising (TFIM) adiabatic path
#   H(s) = (1 - s) * H0  +  s * H1,  s = t/T (linear schedule)
#   H0 = -Gamma * sum_i X_i
#   H1 = -J * sum_<i,i+1> Z_i Z_{i+1} - h * sum_i Z_i
# Initial state: |+>^{⊗n}, the ground state of H0 at large Gamma
# Time evolution: piecewise-constant integration with m steps (dt = T/m):
#                 psi_{k+1} = exp(-i * dt * H(s_k)) @ psi_k
# Reliability: L2 distance to a high-resolution reference (m_ref = 4 * max(m_list))
# Carbon: derived ONLY from Performance runtimes (no reliability/scalability data used)
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
#These imports provide core system utilities for timing, memory profiling, argument parsing, and parallel execution.
#Typing helpers clarify data structures used throughout the benchmark code.
#NumPy and pandas support numerical computation and data analysis, while Matplotlib enables result visualization.
#Warnings are globally suppressed to keep benchmark output concise and readable.

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
# ------------********------------ Utilities
# ------------********------------
#This small utility ensures that a directory exists before writing outputs.
#It creates the path if needed while safely ignoring cases where it already exists.

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

#This small utility ensures that a directory exists before writing outputs.
#It creates the path if needed while safely ignoring cases where it already exists.

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

#This utility saves a Matplotlib figure to disk with consistent layout and resolution.
#It closes the figure after saving to free resources and prints a confirmation message.

def save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved plot: {path}")

#This helper resolves a user-specified or default file path, prioritizing files on the Desktop.
#It supports quoted inputs, absolute and relative paths, and common spreadsheet extensions.
#If no explicit argument is given, it searches through a list of default candidate filenames.
#The function returns a validated absolute path or None if no matching file is found.

def resolve_desktop_file(user_arg: str | None,
                         default_candidates: List[str]) -> str | None:
    desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
    def exists(p): return os.path.isfile(p)

    if user_arg:
        base = user_arg.strip().strip('"').strip("'")
        if exists(base):
            return os.path.abspath(base)
        candidate = os.path.join(desktop, base)
        if exists(candidate):
            return os.path.abspath(candidate)
        exts = [".xlsx", ".xls", ".csv"]
        if not os.path.splitext(base)[1]:
            for ext in exts:
                cand = os.path.join(desktop, base + ext)
                if exists(cand):
                    return os.path.abspath(cand)
        tail = os.path.split(base)[1].lower()
        for fname in os.listdir(desktop):
            if fname.lower() == tail or (not os.path.splitext(base)[1] and any(fname.lower() == tail + e for e in exts)):
                cand = os.path.join(desktop, fname)
                if exists(cand):
                    return os.path.abspath(cand)
        return None

    for cand in default_candidates:
        if exists(cand):
            return os.path.abspath(cand)
    return None

# ------------********------------
# ------------********------------ Linear algebra primitives
# ------------********------------
#This function defines and returns the four fundamental Pauli operators used in quantum mechanics.
#Each operator is explicitly constructed as a 2×2 complex-valued matrix for reuse in simulations.

def paulis() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z

#This helper computes the Kronecker (tensor) product of a sequence of matrices.
#It iteratively combines matrices to build operators for multi-qubit systems.

def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def single_qubit_op(n: int, i: int, op: np.ndarray) -> np.ndarray:
    I, _, _, _ = paulis()
    mats = [I] * n
    mats[i] = op
    return kron_all(mats)

def two_qubit_op(n: int, i: int, j: int, op_i: np.ndarray, op_j: np.ndarray) -> np.ndarray:
    I, _, _, _ = paulis()
    mats = [I] * n
    mats[i] = op_i
    mats[j] = op_j
    return kron_all(mats)

def expm_hermitian(H: np.ndarray, dt: float) -> np.ndarray:
    w, V = np.linalg.eigh(H)
    phase = np.exp(-1j * dt * w)
    return (V * phase) @ V.conj().T

# ------------********------------
# ------------********------------ Adiabatic TFIM Hamiltonians
# ------------********------------
#This function constructs the transverse-field Ising model Hamiltonian components in dense form.
#H0 captures the transverse X-field term, while H1 includes nearest-neighbor ZZ interactions and a longitudinal Z-field.
#All operators are embedded into the full Hilbert space using Kronecker products.
#The two Hamiltonian parts are returned separately to support splitting or Trotterization schemes.

def build_tfim_components(n: int, J: float = 1.0, h: float = 0.5, Gamma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    H0 = -Gamma * sum_i X_i
    H1 = -J * sum_<i,i+1> Z_i Z_{i+1} - h * sum_i Z_i
    Returns (H0, H1) as dense matrices (2^n x 2^n)
    """
    I, X, Y, Z = paulis()
    dim = 2 ** n

    H0 = np.zeros((dim, dim), dtype=complex)
    H1 = np.zeros((dim, dim), dtype=complex)

    for i in range(n):
        H0 += -Gamma * single_qubit_op(n, i, X)

    for i in range(n - 1):
        H1 += -J * (two_qubit_op(n, i, i + 1, Z, Z))
    for i in range(n):
        H1 += -h * single_qubit_op(n, i, Z)

    return H0, H1

def init_plus_state(n: int) -> np.ndarray:
    """|+>^{⊗n}"""
    plus = (1/np.sqrt(2)) * np.array([1.0, 1.0], dtype=complex)
    state = plus
    for _ in range(n - 1):
        state = np.kron(state, plus)
    return state

# ------------********------------
# ------------********------------ Time-dependent evolution
# ------------********------------
#This function simulates adiabatic quantum evolution using a piecewise-constant approximation.
#The Hamiltonian is interpolated linearly between H0 and H1 over total time T.
#At each step, a midpoint rule is used to improve accuracy of the time discretization.
#The statevector is updated via exact exponentiation of the instantaneous Hamiltonian.

def simulate_adiabatic(H0: np.ndarray, H1: np.ndarray, T: float, m_steps: int, psi0: np.ndarray) -> np.ndarray:
   
    dt = T / m_steps
    psi = psi0.copy()
    for k in range(m_steps):
        t = (k + 0.5) * dt  # midpoint for better accuracy
        s = t / T
        Hs = (1.0 - s) * H0 + s * H1
        U = expm_hermitian(Hs, dt)
        psi = U @ psi
    return psi

#These helpers provide a high-accuracy reference solution for adiabatic evolution.
#The reference uses a much finer time discretization to approximate continuous dynamics.


def simulate_reference(H0: np.ndarray, H1: np.ndarray, T: float, m_ref: int, psi0: np.ndarray) -> np.ndarray:
    """High-resolution reference evolution (m_ref >> m_steps)."""
    return simulate_adiabatic(H0, H1, T=T, m_steps=m_ref, psi0=psi0)

#An ℓ2 norm function measures deviation between approximate and reference statevectors.
def l2_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# ------------********------------
# ------------********------------ Plot helpers
# ------------********------------
#This function plots runtime as a function of adiabatic time slices for different qubit counts.
#It uses logarithmic scaling on the x-axis to highlight scaling behavior.
#The figure is saved to disk using a shared plotting helper.

def plot_runtime_vs_steps(perf_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, sub in perf_df.groupby("n_qubits"):
        sub = sub.sort_values("steps")
        ax.plot(sub["steps"], sub["runtime_sec"], marker="o", label=f"n={n}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Adiabatic time slices (m)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs time slices (Adiabatic)")
    ax.legend()
    save_fig(fig, out_path)

#This function visualizes peak memory usage versus adiabatic time slices across system sizes.
#Logarithmic scaling helps compare memory growth as the number of slices increases.
#The plot is saved with consistent formatting for reporting.

def plot_memory_vs_steps(perf_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, sub in perf_df.groupby("n_qubits"):
        sub = sub.sort_values("steps")
        ax.plot(sub["steps"], sub["peak_mem_mb"], marker="s", label=f"n={n}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Adiabatic time slices (m)")
    ax.set_ylabel("Peak allocated (MB, tracemalloc)")
    ax.set_title("Memory vs time slices (Adiabatic)")
    ax.legend()
    save_fig(fig, out_path)


#This function plots the mean L2 error against the number of adiabatic time slices.
#Both axes are logarithmic to reveal convergence trends across qubit counts.
#It highlights the reliability of the adiabatic approximation.

def plot_error_vs_steps(rel_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, sub in rel_df.groupby("n_qubits"):
        sub = sub.sort_values("steps")
        ax.plot(sub["steps"], sub["mean_error"], marker="o", label=f"n={n}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Adiabatic time slices (m)")
    ax.set_ylabel("Mean L2 error vs reference")
    ax.set_title("Reliability: Error vs time slices (Adiabatic)")
    ax.legend()
    save_fig(fig, out_path)


#This function generates a heatmap of logarithmic error values over qubit counts and time slices.
#It aggregates raw errors to provide a global view of accuracy across parameters.
#Color encoding makes error growth patterns easy to interpret.

def plot_error_heatmap(raw_err: pd.DataFrame, out_path: str) -> None:
    piv = raw_err.pivot_table(index="n_qubits", columns="steps", values="error", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.log10(piv.values + 1e-16), aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
    ax.set_xlabel("Adiabatic time slices (m)")
    ax.set_ylabel("n_qubits")
    ax.set_title("log10(Error): n vs m (Adiabatic)")
    fig.colorbar(im, ax=ax, label="log10(L2 error)")
    save_fig(fig, out_path)


#This function plots runtime scalability against Hilbert space dimension for a fixed time-slice budget.
#It selects the closest available step count to the requested value for robustness.
#Log–log scaling reveals exponential growth trends with system size.

def plot_scaling_runtime_vs_dim(perf_df: pd.DataFrame, m_for_plot: int, out_path: str) -> None:
    avail = sorted(perf_df["steps"].unique())
    m = min(avail, key=lambda x: abs(x - m_for_plot))
    sub = perf_df[perf_df["steps"] == m].copy()
    sub["dimension"] = 2 ** sub["n_qubits"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["dimension"], sub["runtime_sec"], marker="o")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Hilbert space dimension (2^n)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Scalability: runtime vs dimension (m≈{m})")
    save_fig(fig, out_path)

#This function plots runtime scaling versus adiabatic time slices for a fixed number of qubits.
#It uses logarithmic axes to emphasize computational complexity.
#The resulting figure helps assess time-discretization costs for a given system size.

def plot_scaling_runtime_vs_steps(perf_df: pd.DataFrame, n_for_plot: int, out_path: str) -> None:
    sub = perf_df[perf_df["n_qubits"] == n_for_plot].copy().sort_values("steps")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["steps"], sub["runtime_sec"], marker="o")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Adiabatic time slices (m)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Scalability: runtime vs time slices (n={n_for_plot})")
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
    med = sub.groupby("steps")["emissions_kgCO2"].median().reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(med["steps"], med["emissions_kgCO2"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Adiabatic time slices (m)")
    ax.set_ylabel("Median emissions across countries (kg CO2e)")
    ax.set_title(f"Median emissions vs time slices (n={n_for_plot})")
    save_fig(fig, out_path)

# ------------********------------
# ------------********------------ Carbon logic
# ------------********------------

def load_country_co2(path: str, year_select: str = "latest") -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CO2 file not found: {path}")

    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    cname = next((c for c in df.columns if "country" in c or c in ("name", "nation")), None)
    if cname is None:
        raise ValueError("Could not find a 'country' column.")
    yname = next((c for c in df.columns if "year" in c), None)

    cand = [c for c in df.columns if "intensity" in c]
    if not cand: cand = [c for c in df.columns if ("co2" in c and "kwh" in c)]
    if not cand: cand = [c for c in df.columns if (("kg" in c or "g" in c) and "kwh" in c)]
    if not cand:
        raise ValueError("No intensity column found.")
    icol = cand[0]

    use_cols = [cname, icol] + ([yname] if yname else [])
    out = df[use_cols].dropna().copy()
    out.columns = ["country", "intensity_raw"] + (["year"] if yname else [])
    out["intensity_raw"] = pd.to_numeric(out["intensity_raw"], errors="coerce")
    out = out.dropna(subset=["intensity_raw"])

    if np.nanmedian(out["intensity_raw"].values) > 10.0:
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
    perf["_key"] = 1; co2["_key"] = 1
    joined = perf.merge(co2, on="_key").drop(columns="_key")
    joined["emissions_kgCO2"] = joined["energy_kwh"] * joined["kgco2_per_kwh"]
    return joined

def build_worstcase_country_table(perf_df: pd.DataFrame,
                                  co2_df: pd.DataFrame,
                                  power_watts: float,
                                  pue: float) -> Tuple[pd.DataFrame, dict]:
    n_heavy = int(perf_df["n_qubits"].max())
    m_heavy = int(perf_df["steps"].max())
    mean_runtime_s = float(perf_df[(perf_df["n_qubits"] == n_heavy) &
                                   (perf_df["steps"] == m_heavy)]["runtime_sec"].mean())
    energy_kwh = mean_runtime_s * power_watts / 3.6e6 * pue

    tbl = co2_df.copy()
    if "year" not in tbl.columns:
        tbl["year"] = pd.NA
    tbl_out = pd.DataFrame({
        "Country": tbl["country"].astype(str),
        "Year": tbl["year"],
        "Intensity": tbl["kg_per_kwh"],   # kg CO2e / kWh
        "kWh": energy_kwh,                # constant across countries here
    })
    tbl_out["kgCO2e"] = tbl_out["Intensity"] * tbl_out["kWh"]
    tbl_out = tbl_out.sort_values("kgCO2e", ascending=False).reset_index(drop=True)

    meta = {"n_heavy": n_heavy, "m_heavy": m_heavy,
            "mean_runtime_s": mean_runtime_s, "power_watts": power_watts,
            "pue": pue, "kWh_used": energy_kwh}
    return tbl_out, meta

# ------------********------------
# ------------********------------ Benchmark driver (+ multiprocessing)
# ------------********------------
#This worker function runs a single adiabatic gate-based simulation instance.
#It initializes the system in a uniform superposition, measures runtime and memory usage,
#and returns both performance metrics and the final simulated statevector.

def _one_task(args):
    n, m, r, seed_base, J, h, Gamma, T = args
    dim = 2 ** n
    H0, H1 = build_tfim_components(n, J=J, h=h, Gamma=Gamma)
    psi0 = init_plus_state(n)  # deterministic start (adiabatic)

    # Performance + Reliability vs high-res reference
    tracemalloc.start()
    t0 = time.perf_counter()
    psi_m = simulate_adiabatic(H0, H1, T=T, m_steps=m, psi0=psi0)
    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "n_qubits": n, "dimension": dim, "steps": m, "repeat": r,
        "runtime_sec": runtime, "peak_mem_mb": peak / (1024 ** 2),
    }, psi_m

#This function orchestrates the full benchmark sweep over qubit counts, time-slice counts, and repeats.
#It supports parallel execution, collects performance data, and stores intermediate states for error analysis.
#High-resolution dense reference evolutions are computed per system size to quantify approximation error.

def run_benchmarks(n_list: List[int],
                   m_list: List[int],
                   repeats: int,
                   J: float,
                   h: float,
                   Gamma: float,
                   T: float,
                   workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      perf_df: per-run performance (runtime, memory)
      err_raw: per-run reliability vs high-res reference (L2 error)
    """
    tasks = [(n, m, r, 0, J, h, Gamma, T)
             for n in n_list for m in m_list for r in range(repeats)]

    perf_rows = []
    states = []

    if workers and workers > 1:
        workers = min(workers, cpu_count())
        with Pool(processes=workers) as pool:
            for perf_row, psi_m in pool.imap_unordered(_one_task, tasks):
                perf_rows.append(perf_row); states.append((perf_row["n_qubits"], perf_row["steps"], psi_m))
    else:
        for t in tasks:
            perf_row, psi_m = _one_task(t)
            perf_rows.append(perf_row); states.append((perf_row["n_qubits"], perf_row["steps"], psi_m))

    perf_df = pd.DataFrame(perf_rows)

    # Build high-resolution references once per n
    err_rows = []
    for n in sorted(set(perf_df["n_qubits"])):
        H0, H1 = build_tfim_components(n, J=J, h=h, Gamma=Gamma)
        psi0 = init_plus_state(n)
        m_ref = int(max(m_list) * 4)  # high-resolution reference
        psi_ref = simulate_reference(H0, H1, T=T, m_ref=m_ref, psi0=psi0)

        for (nn, mm, psi_m) in [s for s in states if s[0] == n]:
            err = l2_error(psi_m, psi_ref)
            err_rows.append({"n_qubits": nn, "steps": mm, "repeat": 0, "error": float(err)})

    err_raw = pd.DataFrame(err_rows)
    return perf_df, err_raw

#This helper aggregates raw error measurements to assess reliability of the adiabatic simulation.
#It computes mean, variance, extrema, and sample counts grouped by qubit number and time slices.

def summarize_reliability(err_raw: pd.DataFrame) -> pd.DataFrame:
    g = err_raw.groupby(["n_qubits", "steps"], as_index=False)
    return g.agg(mean_error=("error", "mean"),
                 std_error=("error", "std"),
                 max_error=("error", "max"),
                 min_error=("error", "min"),
                 num_runs=("error", "count"))

#This function summarizes scalability behavior by fitting power-law models.
#It estimates how runtime scales with Hilbert space dimension and with the number of adiabatic steps.
#The resulting fit parameters provide concise indicators of computational complexity.

def summarize_scalability(perf_df: pd.DataFrame) -> pd.DataFrame:
    m_target = perf_df["steps"].max()
    sub_dim = perf_df.loc[perf_df["steps"] == m_target].copy()
    sub_dim["dimension"] = 2 ** sub_dim["n_qubits"]
    sub_dim = sub_dim.groupby(["n_qubits", "dimension"], as_index=False)["runtime_sec"].mean()
    x = np.log(sub_dim["dimension"].values + 1e-16)
    y = np.log(sub_dim["runtime_sec"].values + 1e-16)
    alpha, _ = np.polyfit(x, y, 1)

    n_target = perf_df["n_qubits"].max()
    sub_steps = perf_df.loc[perf_df["n_qubits"] == n_target].copy()
    sub_steps = sub_steps.groupby(["steps"], as_index=False)["runtime_sec"].mean()
    xb = np.log(sub_steps["steps"].values + 1e-16)
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
    parser = argparse.ArgumentParser(description="Classical Adiabatic Evolution with four benchmarks and carbon outputs.")
    # Grid / physics
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=6)
    parser.add_argument("--steps", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--J", type=float, default=1.0, help="Ising ZZ coupling in H1")
    parser.add_argument("--h", type=float, default=0.5, help="Longitudinal Z field in H1")
    parser.add_argument("--Gamma", type=float, default=1.0, help="Transverse X field in H0")
    parser.add_argument("--T", type=float, default=1.0, help="Total evolution time")

    # Carbon / files
    parser.add_argument("--excel", type=str, default=None,
                        help="CO2 file name or path (e.g., 'Filtered CO2 intensity 236 Countries.xlsx').")
    parser.add_argument("--countries-co2-file", type=str, default=None,
                        help="Alias for the same CO2 file path.")
    parser.add_argument("--device-power-watts", type=float, default=None,
                        help="Average device power draw in Watts.")
    parser.add_argument("--power-watts", type=float, default=None,
                        help="Alias of --device-power-watts.")
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", choices=["latest", "all"], default="latest")

    # Output / execution
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--combine", action="store_true",
                        help="Accepted for compatibility; no effect.")
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    # steps
    try:
        m_list = sorted(set(int(s.strip()) for s in args.steps.split(",") if s.strip()))
        m_list = [m for m in m_list if m > 0]
    except Exception:
        print("Invalid --steps; use e.g. '1,2,4,8,16,32'"); sys.exit(1)

    # qubits
    if args.n_max < args.n_min or args.n_min < 1:
        print("Invalid n range."); sys.exit(1)
    n_list = list(range(args.n_min, args.n_max + 1))

    # power
    power_watts = args.device_power_watts if args.device_power_watts is not None else (
        args.power_watts if args.power_watts is not None else 45.0
    )

    # CO2 file on Desktop
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

    # Output dirs
    root = os.path.abspath(args.outdir)
    perf_dir = os.path.join(root, "Performance")
    scal_dir = os.path.join(root, "Scalability")
    reli_dir = os.path.join(root, "Reliability")
    carb_dir = os.path.join(root, "Carbon")
    for d in (perf_dir, scal_dir, reli_dir, carb_dir): ensure_dir(d)

    print("[INFO] Running adiabatic benchmarks ...")
    t0_all = time.perf_counter()
    perf_df, err_raw = run_benchmarks(
        n_list=n_list, m_list=m_list, repeats=args.repeats,
        J=args.J, h=args.h, Gamma=args.Gamma, T=args.T,
        workers=max(1, int(args.workers))
    )

    # PERFORMANCE
    perf_summary = perf_df.groupby(["n_qubits", "steps"], as_index=False).agg(
        mean_runtime_sec=("runtime_sec", "mean"),
        std_runtime_sec=("runtime_sec", "std"),
        mean_peak_mem_mb=("peak_mem_mb", "mean"),
        std_peak_mem_mb=("peak_mem_mb", "std")
    )
    write_excel(
        {"raw": perf_df.sort_values(["n_qubits", "steps", "repeat"]),
         "summary": perf_summary.sort_values(["n_qubits", "steps"])},
        os.path.join(perf_dir, "performance.xlsx")
    )
    plot_runtime_vs_steps(perf_df, os.path.join(perf_dir, "runtime_vs_steps.png"))
    plot_memory_vs_steps(perf_df, os.path.join(perf_dir, "memory_vs_steps.png"))

    # RELIABILITY
    rel_summary = summarize_reliability(err_raw)
    write_excel(
        {"errors_raw": err_raw.sort_values(["n_qubits", "steps", "repeat"]),
         "errors_summary": rel_summary.sort_values(["n_qubits", "steps"])},
        os.path.join(reli_dir, "reliability.xlsx")
    )
    plot_error_vs_steps(rel_summary, os.path.join(reli_dir, "error_vs_steps.png"))
    plot_error_heatmap(err_raw, os.path.join(reli_dir, "error_heatmap.png"))

    # SCALABILITY
    scaling_fit = summarize_scalability(perf_df)
    write_excel(
        {"perf_raw": perf_df.sort_values(["n_qubits", "steps", "repeat"]),
         "scaling_fit": scaling_fit},
        os.path.join(scal_dir, "scalability.xlsx")
    )
    plot_scaling_runtime_vs_dim(perf_df, m_for_plot=max(m_list),
                                out_path=os.path.join(scal_dir, "runtime_vs_dimension_loglog.png"))
    plot_scaling_runtime_vs_steps(perf_df, n_for_plot=max(n_list),
                                  out_path=os.path.join(scal_dir, "runtime_vs_steps_at_largest_n.png"))

    # CARBON (Performance-only)
    if co2_path and os.path.isfile(co2_path):
        co2_df = load_country_co2(co2_path, year_select=args.year_select)
        carbon_df = compute_emissions(perf_df, co2_df, power_watts=power_watts, pue=args.pue)
        worst_table, meta = build_worstcase_country_table(perf_df, co2_df, power_watts, args.pue)

        write_excel(
            {
                "co2_by_country": (co2_df.rename(columns={"kg_per_kwh": "kgco2_per_kwh"})
                                   if "kg_per_kwh" in co2_df.columns else co2_df),
                "emissions_all": carbon_df.sort_values(["n_qubits", "steps", "country"]),
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

    print(f"[DONE] Adiabatic benchmarks completed in {time.perf_counter() - t0_all:.2f}s")
    print(f"[OUT] Folders created under: {root}")
    print("      - Performance\\performance.xlsx + 2 PNGs")
    print("      - Scalability\\scalability.xlsx + 2 PNGs")
    print("      - Reliability\\reliability.xlsx + 2 PNGs")
    print("      - Carbon\\carbon_footprint.xlsx (if CO2 provided) + PNGs:")
    print("          carbon_distribution.png, carbon_top15.png, median_emissions_vs_steps.png")

if __name__ == "__main__":
    main()
