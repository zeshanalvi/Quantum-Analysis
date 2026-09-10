#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ideal_qaoa (QUANTUM, IDEAL STATEVECTOR)
# This is the quantum (noiseless) version of classical script that i used for classical computer
# It runs QAOA on an ideal statevector simulator and produces the same deliverables:
# Performance, Scalability, Reliability, Carbon folders
# Max-Cut graph visuals (nodes colored by side, dashed cut edges)
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
IMPORTANT: Statevector simulation scales as 2^n.
The script enforces MAX_QUBITS = 16. You can edit this baseline as per your work/project.
  --perf_size 12   --rel_size 12   --sizes 8,10,12,14
"""

# ------------********------------ Imports

from __future__ import annotations

import argparse, glob, math, os, threading, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np                        # Numerical computing library
import pandas as pd                       # Data analysis library for tables/DataFrames
import matplotlib
matplotlib.use("Agg")                     # Use non-GUI backend for saving plots in headless environments
import matplotlib.pyplot as plt
import psutil                             # Provides system and process-level resource usage info
import networkx as nx                     # Graph library for Max-Cut visualizations

# Qiskit (ideal, noiseless statevector)
from qiskit import QuantumCircuit         # Core class for constructing quantum circuits
from qiskit.quantum_info import Statevector  # Represents quantum states for ideal simulations


# ------------********------------ Limits / Globals 

MAX_QUBITS = 16  # hard cap for safe, fast statevector sims
SIGN_CACHE: Dict[int, np.ndarray] = {}  # cache of Z-sign tables for expectation calc

# ------------********------------ Utility & I/O 

#These utility functions locate CO₂-related files on the Desktop, create directories safely, 
# generate timestamps, and save DataFrames to Excel.
#They support higher-level benchmarking tasks by handling filesystem access, file discovery, 
# and standardized output formatting.
#Overall, they provide reusable helpers for consistent input detection and result exporting.

def desktop_path() -> str:
    return os.path.join(os.path.expanduser("~"), "Desktop")   # Build full path to the user's Desktop folder

def auto_find_co2_on_desktop() -> Optional[str]:
    desk = desktop_path()                                     # Get Desktop directory path
    pats = ["*co2*.xlsx", "*CO2*.xlsx", "*co2*.csv", "*CO2*.csv", "Filtered CO2 intensity*.xlsx"]  # File patterns to search for
    for p in pats:
        hits = glob.glob(os.path.join(desk, p))               # Search desktop for matching files
        if hits:
            return hits[0]                                    # Return the first match found
    return None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)                          # Create directory if it doesn't already exist

def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")                     # Timestamp for filenames (YYYYMMDD_HHMMSS)

def save_df_excel(df: pd.DataFrame, outpath: str, sheet_name: str = "results") -> None:
    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:  # Use ExcelWriter to save DataFrame to .xlsx
        df.to_excel(writer, index=False, sheet_name=sheet_name)

# ------------********------------ CO2 Helpers 
#This function attempts to identify a column representing country information by checking for common country-related names.
#If no direct match is found, it falls back to returning the first column containing string data.
#If neither condition is satisfied, it returns None.

def choose_country_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("country", "name", "nation", "entity", "location"):
            return c   # Prefer common country-identifying column names
    for c in df.columns:
        if df[c].dtype == object:
            return c   # Fallback: choose first string-type column
    return None

#This function selects a CO₂ intensity column by first checking for exact matches against a list of preferred column names.
#If none match, it looks for heuristic patterns containing both “co2” and “kwh,” indicating an intensity-like field.
#As a last fallback, it returns the first numeric column available or None if none exist.

def choose_intensity_column(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "gco2_per_kwh", "co2_g_per_kwh", "grid_intensity_g_per_kwh",
        "intensity_gco2_per_kwh", "g_co2_per_kwh", "g_per_kwh", "gco2kwh"
    ]
    lower = {c: str(c).lower() for c in df.columns}   # Map lowercase names for consistent matching
    for want in preferred:
        for c in df.columns:
            if lower[c] == want:
                return c   # Exact match with preferred CO₂ intensity column
    for c in df.columns:
        lc = lower[c]
        if "co2" in lc and "kwh" in lc:
            return c   # Heuristic match: contains both CO₂ and kWh
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else None   # Final fallback: any numeric column

def choose_year_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("year", "yr"):
            return c   # Direct match to year field
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("date", "time", "period"):
            return c   # Alternative time-related column
    return None

def read_co2_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()   # Detect extension to choose CSV or Excel parser
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")

#This function returns the most recent record per country, using the year column when available and valid.
#If no usable year column exists, it simply keeps the last non-null entry for each country.
#When a year column is present, it converts years to numeric, identifies each country’s latest year, and returns those rows.

def latest_per_country(df: pd.DataFrame, year_col: Optional[str], country_col: str) -> pd.DataFrame:
    if (year_col is None) or (year_col not in df.columns):
        return df.dropna(subset=[country_col]).drop_duplicates(subset=[country_col], keep="last")
    y = pd.to_numeric(df[year_col], errors="coerce")   # Convert year values to numeric for sorting
    df = df.copy()
    df["_year_num"] = y
    idx = df.groupby(country_col)["_year_num"].idxmax()  # Pick the row with the latest year per country
    out = df.loc[idx].drop(columns=["_year_num"])
    return out


# ------------********------------ CPU / Memory Sampling

#This class continuously samples CPU usage in a background daemon thread, storing each reading for later analysis.
#It supports stopping the sampling loop and provides helper methods to compute the average and maximum CPU utilization.
#The accompanying function retrieves the current process’s memory usage in megabytes for lightweight system monitoring.

class CPUUsageSampler(threading.Thread):
    def __init__(self, interval_sec: float = 0.2):
        super().__init__(daemon=True)                     # Daemon thread so it stops automatically with the main program
        self.interval = interval_sec
        self.samples: List[float] = []                    # List to store sampled CPU usage values
        self._stop = threading.Event()                    # Event used to signal the thread to stop
    def run(self):
        psutil.cpu_percent(interval=None)                 # Prime psutil so first reading isn't incorrect
        while not self._stop.is_set():
            self.samples.append(float(psutil.cpu_percent(interval=self.interval)))  # Sample CPU usage repeatedly
    def stop(self):
        self._stop.set()                                  # Signal the sampling loop to terminate
    def avg_util(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0
    def max_util(self) -> float:
        return float(np.max(self.samples)) if self.samples else 0.0

def current_mem_mb() -> float:
    proc = psutil.Process(os.getpid())                    # Get the current process object
    return proc.memory_info().rss / (1024.0 ** 2)         # Report memory usage in megabytes


# ------------********------------ Problems & Mappings 

#This function generates a random weighted undirected graph by sampling edges with a given probability 
#and assigning each a random weight.
#It constructs a symmetric adjacency matrix by keeping only the upper triangle and mirroring it.
#Self-loops are removed by zeroing the diagonal, and the final matrix is returned as floats.

def random_weighted_graph(n: int, edge_prob: float = 0.5, w_low: int = 1, w_high: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)                       # Seeded RNG for reproducible graph generation
    A = rng.random((n, n))
    W = (A < edge_prob).astype(float)                       # Create adjacency mask based on probability
    weights = rng.integers(w_low, w_high + 1, size=(n, n))  # Random edge weights within range
    W = W * weights                                         # Apply weights to edges that exist
    W = np.triu(W, 1)                                       # Keep only upper triangle for undirected graph
    W = W + W.T                                             # Mirror to make matrix symmetric
    np.fill_diagonal(W, 0.0)
    return W.astype(float)

#The following function computes the Max-Cut objective by converting a bitstring into 
#spin form and applying the standard quadratic cost expression.


def maxcut_cost_bitstring(x: np.ndarray, W: np.ndarray) -> float:
    s = 1 - 2 * x                                           # Convert {0,1} bits to {+1,-1} spins
    return 0.25 * (np.sum(W) - np.sum(W * np.outer(s, s)))

#The following function generates a random QUBO matrix by sampling entries under a sparsity mask, 
#assigning random coefficients, and symmetrizing the result.
#Both utilities (above and down functions) produce numerical structures used as problem instances for optimization benchmarks.

def random_qubo(n: int, density: float = 0.3, q_low: int = -5, q_high: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density                     # Sparsity mask controlling QUBO density
    Q = rng.integers(q_low, q_high + 1, size=(n, n)) * mask # Apply mask to random Q values
    Q = np.triu(Q)
    Q = Q + Q.T - np.diag(np.diag(Q))                       # Symmetrize and keep diagonal once
    return Q.astype(float)

def qubo_to_ising(Q: np.ndarray) -> Tuple[float, np.ndarray, Dict[Tuple[int,int], float]]:
    """
    Map x^T Q x (x∈{0,1}) to Ising: const + sum_i a_i Z_i + sum_{i<j} b_ij Z_i Z_j
    We *maximize* -x^T Q x to align with earlier behavior -> negate all coeffs.
    Returns (const, a (n,), couplings dict {(i,j): b_ij}).
    """
    n = Q.shape[0]
    const = 0.5 * np.trace(Q) + 0.5 * np.sum(np.triu(Q, 1)) # Constant shift in QUBO→Ising mapping
    a = np.zeros(n, dtype=float)
    b: Dict[Tuple[int,int], float] = {}
    for i in range(n):
        a[i] -= 0.5 * Q[i, i]                                # Linear coefficients from diagonal terms
        for j in range(i+1, n):
            a[i] -= 0.5 * Q[i, j]                            # Contribution of Q[i,j] to spin i
            a[j] -= 0.5 * Q[i, j]                            # Contribution to spin j
            b[(i, j)] = 0.5 * Q[i, j]                        # Pairwise coupling term
    # negate for maximize(-QUBO)
    const = -const
    a = -a
    b = {k: -v for k, v in b.items()}                         # Flip coupling signs for maximization
    return const, a, b


# ------------********------------ QAOA (Quantum, ideal)  ------------********------------
#This function enforces a hard upper limit on qubit count 
#   to keep statevector simulations within memory-safe bounds.
#If the requested size exceeds the maximum allowed, it raises a clear, helpful error
#   explaining the limit and suggesting safer alternatives.
#This prevents accidental crashes or OOM errors during large-scale quantum simulations.

def _check_qubit_limit(n: int):
    if n > MAX_QUBITS:
        raise ValueError(
            f"[Quantum] Requested n={n} qubits exceeds statevector cap MAX_QUBITS={MAX_QUBITS}.\n"
            f"Please rerun with smaller sizes, e.g.: --perf_size 12 --rel_size 12 --sizes 8,10,12,14"
        )   # Prevent simulation sizes that exceed the allowed statevector limit



#This function applies the Max-Cut cost Hamiltonian by adding weighted ZZ rotations 
#   between every pair of nodes connected by a non-zero edge.
#Each interaction contributes an RZZ gate with angle −γ·w, encoding the cost term exp(−iγ H_cost).
#It iterates over all upper-triangle edges, ensuring every unique pairwise interaction is applied exactly once.

def _apply_maxcut_cost_layer(circ: QuantumCircuit, W: np.ndarray, gamma: float):
    n = W.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            w = float(W[i, j])
            if w != 0.0:
                # RZZ(θ) = exp(-i θ/2 Z⊗Z). For MaxCut, θ = -gamma * w determined by cost Hamiltonian
                circ.rzz(-gamma * w, i, j)   # Apply weighted ZZ rotation for each MaxCut edge interaction



#This function applies the QUBO (Ising-form) cost layer by adding local Z-rotations 
#   for linear terms and RZZ gates for pairwise couplings.
#Single-qubit terms use RZ(2γaᵢ), while interaction terms use RZZ(2γbᵢⱼ), 
#   matching the Ising Hamiltonian evolution operator.
#Only non-zero coefficients are applied, avoiding unnecessary quantum operations.
def _apply_qubo_cost_layer(circ: QuantumCircuit, a: np.ndarray, b: Dict[Tuple[int,int], float], gamma: float):
    n = len(a)
    # single-Z terms: exp(-i γ a_i Z) = RZ(2γ a_i)
    for i in range(n):
        ai = float(a[i])
        if ai != 0.0:
            circ.rz(2.0 * gamma * ai, i)        # Apply local Z-rotation from linear Ising term
    # ZZ terms: exp(-i γ b_ij Z_i Z_j) = RZZ(2γ b_ij)
    for (i, j), bij in b.items():
        if bij != 0.0:
            circ.rzz(2.0 * gamma * float(bij), i, j)  # Coupling term between qubits i and j



def _apply_mixer_layer(circ: QuantumCircuit, beta: float):
    # e^{-i β X} = RX(2β)
    for q in range(circ.num_qubits):
        circ.rx(2.0 * beta, q)                 # Mixer applies rotations around X for all qubits


#This function constructs a full QAOA circuit by preparing |+⟩ⁿ, 
#  then applying alternating cost and mixer layers for either Max-Cut or QUBO.
# It converts QUBO to Ising form when required and enforces qubit-count limits before circuit creation.
# The completed parameterized circuit is returned for simulation or execution.

def build_qaoa_circuit(problem: str, mat: np.ndarray,
                       p_layers: int, gamma: float, beta: float) -> QuantumCircuit:
    n = mat.shape[0]
    _check_qubit_limit(n)                      # Prevent circuits that exceed max simulator qubit count
    qc = QuantumCircuit(n)
    # Start in |+>^n
    qc.h(range(n))                             # Apply Hadamards to prepare uniform superposition
    if problem == "maxcut":
        for _ in range(p_layers):
            _apply_maxcut_cost_layer(qc, mat, gamma)
            _apply_mixer_layer(qc, beta)
    elif problem == "qubo":
        const, a, b = qubo_to_ising(mat)
        for _ in range(p_layers):
            _apply_qubo_cost_layer(qc, a, b, gamma)
            _apply_mixer_layer(qc, beta)
    else:
        raise ValueError("Unsupported problem.")
    return qc


def statevector_from_qc(qc: QuantumCircuit) -> np.ndarray:
    sv = Statevector.from_instruction(qc)                   # Simulate full statevector from the quantum circuit
    return np.asarray(sv.data, dtype=np.complex128)

def _z_signs(n: int) -> np.ndarray:
    """Return an (n, 2^n) array with entries in {+1,-1} for Z eigenvalues of each qubit across basis states."""
    if n in SIGN_CACHE:
        return SIGN_CACHE[n]                                # Use cached result if available
    M = 1 << n                                               # Total number of basis states (2^n)
    idx = np.arange(M, dtype=np.uint64)
    signs = np.empty((n, M), dtype=np.int8)
    # Qiskit qubit 0 is least significant bit
    for i in range(n):
        bit = (idx >> i) & 1                                 # Extract qubit-i bit for all basis states
        signs[i, :] = 1 - 2*bit                              # Map 0→+1, 1→−1 for Z eigenvalues
    SIGN_CACHE[n] = signs                                    # Cache result for reuse
    return signs

#This function computes the Max-Cut expectation value by evaluating ⟨ZᵢZⱼ⟩ 
# for every weighted edge and converting it into cut weight contributions.
#It uses cached Z-sign tables to efficiently compute edge correlations under the probability distribution.
#It also extracts and returns the most probable bitstring corresponding to the simulated QAOA state.

def expectation_maxcut(probs: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray]:
    n = W.shape[0]
    S = _z_signs(n)  # (n, 2^n)
    # For each edge, <Z_i Z_j> = sum_z p(z) z_i z_j
    cut = 0.0
    for i in range(n):
        for j in range(i+1, n):
            w = W[i, j]
            if w != 0.0:
                zz = float(np.dot(probs, (S[i, :] * S[j, :]).astype(np.float64)))
                cut += w * 0.5 * (1.0 - zz)
    # also return the most probable bitstring x*
    max_k = int(np.argmax(probs))
    x_bits = np.array([(max_k >> i) & 1 for i in range(n)], dtype=np.int64)  # little-endian
    return cut, x_bits

def expectation_qubo(probs: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    We report the QAOA expectation for the *maximization* target used above: maximize(-x^T Q x).
    """
    n = Q.shape[0]
    const, a, b = qubo_to_ising(Q)  # already negated for maximize(-QUBO)
    S = _z_signs(n)
    # <Z_i> and <Z_i Z_j>
    z_exp = np.array([float(np.dot(probs, S[i, :].astype(np.float64))) for i in range(n)])
    exp_val = const + float(np.dot(a, z_exp))
    for (i, j), bij in b.items():
        zz = float(np.dot(probs, (S[i, :] * S[j, :]).astype(np.float64)))
        exp_val += bij * zz
    # Most probable bitstring for visualization and a concrete cost
    max_k = int(np.argmax(probs))
    x_bits = np.array([(max_k >> i) & 1 for i in range(n)], dtype=np.int64)
    return exp_val, x_bits

#This function performs a full QAOA parameter sweep over (γ, β) 
#  using an ideal statevector simulator to evaluate every circuit exactly.
#For each parameter pair, it builds the circuit, computes the statevector, 
#   converts it to probabilities, and evaluates either Max-Cut or QUBO expectations.
#It tracks and returns the best bitstring, its objective value, and metadata describing the explored parameter grid.

def cqaoa_run(x0_unused: np.ndarray, problem: str, mat: np.ndarray,
              cfg, rng) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Quantum QAOA grid search over (gamma, beta) using ideal statevector.
    Returns:
      best_x   -> most probable bitstring from best (gamma,beta) circuit
      best_obj -> QAOA expectation (Max-Cut weight or maximize(-QUBO))
      meta     -> grid/layer info
    """
    n = mat.shape[0]
    _check_qubit_limit(n)
    best_obj = -1e308
    best_x = None

    for gamma in cfg.gamma_grid:
        for beta in cfg.beta_grid:
            qc = build_qaoa_circuit(problem, mat, cfg.p_layers, gamma, beta)
            psi = statevector_from_qc(qc)
            probs = (psi.real**2 + psi.imag**2).astype(np.float64)
            if problem == "maxcut":
                exp_val, x_bits = expectation_maxcut(probs, mat)
            else:
                exp_val, x_bits = expectation_qubo(probs, mat)
            if exp_val > best_obj:
                best_obj = exp_val
                best_x = x_bits

    return best_x, float(best_obj), {
        "gamma_candidates": len(cfg.gamma_grid),
        "beta_candidates": len(cfg.beta_grid),
        "layers": cfg.p_layers
    }

# ------------********------------ Max-Cut Graph Visualization  ------------********------------

def plot_maxcut_graph(W: np.ndarray, x: np.ndarray, out_path: str, title: str = "") -> None:
    G = nx.Graph()
    n = W.shape[0]
    for i in range(n):
        G.add_node(i, part=int(x[i]))                        # Assign each node to its partition (0 or 1)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if w > 0.0:
                G.add_edge(i, j, weight=w, cut=(x[i] != x[j]))  # Mark edge as cut if endpoints differ
    pos = nx.spring_layout(G, seed=42)                       # Compute visually stable layout
    node_colors = ["tab:blue" if G.nodes[i]["part"] == 0 else "tab:orange" for i in G.nodes()]
    cut_edges = [(u, v) for u, v, d in G.edges(data=True) if d["cut"]]  # Edges that cross the cut
    in_edges  = [(u, v) for u, v, d in G.edges(data=True) if not d["cut"]]
    all_w = [G[u][v]["weight"] for u, v in G.edges()] or [1.0] # Edge weights for scaling widths
    maxw = max(all_w)
    lw_cut = [1.0 + 3.0 * (G[u][v]["weight"] / maxw) for u, v in cut_edges]  # Thicker lines for heavier cut edges
    lw_in  = [0.5 + 2.0 * (G[u][v]["weight"] / maxw) for u, v in in_edges]
    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(G, pos, node_size=140, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=in_edges, width=lw_in, alpha=0.45)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=lw_cut, style="dashed")
    nx.draw_networkx_labels(G, pos, font_size=8)
    cut_val = maxcut_cost_bitstring(x, W)                    # Compute final Max-Cut objective value
    plt.title(title or f"Max-Cut partition (weighted cut = {cut_val:.2f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)                           # Save visualization to file
    plt.close()

# ------------********------------ Benchmarks


@dataclass
class CQAOAConfig:
    p_layers: int = 1
    gamma_grid: List[float] = None
    beta_grid: List[float] = None
    seed: int = 0
    def __post_init__(self):
        if self.gamma_grid is None:
            self.gamma_grid = [0.2, 0.4, 0.6, 0.8, 1.0]
        if self.beta_grid is None:
            self.beta_grid = [0.2, 0.4, 0.6, 0.8, 1.0]

# ------------********------------ Performance Benchmarks

def performance_benchmark(args, out_dir: str, rng: np.random.Generator) -> Dict[str, float]:
    print("[Performance][Quantum] Starting benchmark (ideal statevector)…")
    ensure_dir(out_dir)
    if args.perf_size > MAX_QUBITS:
        raise ValueError(f"[Performance] perf_size={args.perf_size} exceeds MAX_QUBITS={MAX_QUBITS}. "
                         f"Use e.g. --perf_size 12")

    records, pngs, graph_imgs = [], [], []

    sampler = CPUUsageSampler(interval_sec=0.2)
    sampler.start()
    t0 = time.perf_counter()

    n = args.perf_size
    instances = args.perf_instances

    for inst in range(instances):
        seed = args.seed + inst
        if args.problem == "maxcut":
            mat = random_weighted_graph(n=n, edge_prob=0.5, seed=seed)
        else:
            mat = random_qubo(n=n, density=0.3, seed=seed)

        x0_dummy = np.zeros(n, dtype=np.int64)  # unused in quantum path
        cfg = CQAOAConfig(p_layers=args.layers,
                          gamma_grid=[0.2, 0.4, 0.6, 0.8, 1.0],
                          beta_grid=[0.2, 0.4, 0.6, 0.8, 1.0],
                          seed=seed)

        run_t0 = time.perf_counter()
        best_x, best_obj, meta = cqaoa_run(x0_dummy, args.problem, mat, cfg, np.random.default_rng(seed))
        run_t1 = time.perf_counter()

        records.append({
            "instance_id": inst,
            "n": n,
            "runtime_sec": run_t1 - run_t0,
            "memory_mb": current_mem_mb(),
            "objective_value": best_obj,  # QAOA expectation
            "layers": meta["layers"],
            "gamma_candidates": meta["gamma_candidates"],
            "beta_candidates": meta["beta_candidates"],
            "seed": seed
        })

        if (args.problem == "maxcut" and args.perf_plot_mode == "graphs"
                and n <= args.viz_graph_maxn and len(graph_imgs) < args.viz_graph_examples):
            gpath = os.path.join(out_dir, f"perf_graph_inst{inst}_{ts()}.png")
            plot_maxcut_graph(mat, best_x, gpath, title=f"Instance {inst} (n={n})")
            graph_imgs.append(gpath)

    t1 = time.perf_counter()
    sampler.stop()
    total_runtime = t1 - t0
    avg_cpu = sampler.avg_util()
    max_cpu = sampler.max_util()

    df = pd.DataFrame.from_records(records)
    excel_path = os.path.join(out_dir, f"performance_{ts()}.xlsx")
    save_df_excel(df, excel_path, sheet_name="performance")
    print(f"[Performance] Saved Excel -> {excel_path}")

    # Plots (2–4)
    plt.figure()
    plt.bar(df["instance_id"].astype(str), df["runtime_sec"])
    plt.xlabel("Instance"); plt.ylabel("Runtime (s)")
    plt.title(f"Performance: Runtime per instance (n={n})")
    p1 = os.path.join(out_dir, f"perf_runtime_{ts()}.png")
    plt.tight_layout(); plt.savefig(p1, dpi=300); plt.close()

    plt.figure()
    plt.bar(df["instance_id"].astype(str), df["memory_mb"])
    plt.xlabel("Instance"); plt.ylabel("Memory (MB)")
    plt.title(f"Performance: Memory per instance (n={n})")
    p2 = os.path.join(out_dir, f"perf_memory_{ts()}.png")
    plt.tight_layout(); plt.savefig(p2, dpi=300); plt.close()

    pngs = [p1, p2]
    if args.perf_plot_mode == "default" or (args.problem != "maxcut"):
        plt.figure()
        plt.plot(df["instance_id"], df["objective_value"], marker="o", linestyle="-")
        plt.xlabel("Instance"); plt.ylabel("QAOA expectation (higher is better)")
        plt.title("Performance: Objective values")
        p3 = os.path.join(out_dir, f"perf_objective_{ts()}.png")
        plt.tight_layout(); plt.savefig(p3, dpi=300); plt.close()

        plt.figure()
        plt.bar(["Avg CPU", "Max CPU"], [avg_cpu, max_cpu])
        plt.ylabel("CPU Utilization (%)"); plt.title("Performance: CPU utilization during runs")
        p4 = os.path.join(out_dir, f"perf_cpu_{ts()}.png")
        plt.tight_layout(); plt.savefig(p4, dpi=300); plt.close()
        pngs.extend([p3, p4])
    else:
        pngs.extend(graph_imgs[:2] if graph_imgs else [])

    print(f"[Performance] Saved {len(pngs)} plots to {out_dir}")

    return {
        "total_runtime_sec": float(total_runtime),
        "avg_cpu_util_pct": float(avg_cpu),
        "mem_mb_mean": float(df["memory_mb"].mean()),
        "excel_path": excel_path,
        "plots": pngs
    }

# ------------********------------ Scalability Benchmarks

#This function begins the quantum scalability benchmark by preparing the output directory and parsing the target problem sizes.
#It filters out any sizes that exceed the hardware or simulator qubit limit, ensuring only feasible benchmarks are run.
#If no valid sizes remain, it raises an informative error instructing the user to provide smaller problem sizes.

def scalability_benchmark(args, out_dir: str, rng: np.random.Generator) -> None:
    print("[Scalability][Quantum] Starting benchmark…")
    ensure_dir(out_dir)
    sizes = [int(s) for s in args.sizes.split(",")]
    sizes_ok = [n for n in sizes if n <= MAX_QUBITS]
    if not sizes_ok:
        raise ValueError(f"[Scalability] All sizes in --sizes exceed MAX_QUBITS={MAX_QUBITS}. "
                         f"Use e.g. --sizes 6,8,10,12")

    records = []
    per_size = args.scal_instances

    for n in sizes_ok:
        rt, mm = [], []
        for k in range(per_size):
            seed = args.seed + 1000 + n * 10 + k
            mat = random_weighted_graph(n, 0.5, seed=seed) if args.problem == "maxcut" else random_qubo(n, 0.3, seed=seed)
            x0_dummy = np.zeros(n, dtype=np.int64)
            cfg = CQAOAConfig(p_layers=args.layers, gamma_grid=[0.2, 0.6, 1.0], beta_grid=[0.2, 0.6, 1.0], seed=seed)
            t0 = time.perf_counter()
            _ = cqaoa_run(x0_dummy, args.problem, mat, cfg, np.random.default_rng(seed))
            t1 = time.perf_counter()
            rt.append(t1 - t0)
            mm.append(current_mem_mb())
        records.append({
            "n": n,
            "avg_runtime_sec": float(np.mean(rt)),
            "std_runtime_sec": float(np.std(rt, ddof=1)) if len(rt) > 1 else 0.0,
            "avg_memory_mb": float(np.mean(mm)),
            "instances": per_size
        })

    df = pd.DataFrame(records).sort_values("n")
    excel_path = os.path.join(out_dir, f"scalability_{ts()}.xlsx")
    save_df_excel(df, excel_path, sheet_name="scalability")
    print(f"[Scalability] Saved Excel -> {excel_path}")

    plt.figure()
    plt.plot(df["n"], df["avg_runtime_sec"], marker="o")
    plt.xlabel("Problem size n"); plt.ylabel("Average runtime (s)")
    plt.title("Scalability: Runtime vs size")
    p1 = os.path.join(out_dir, f"scal_runtime_{ts()}.png")
    plt.tight_layout(); plt.savefig(p1, dpi=300); plt.close()

    plt.figure()
    plt.plot(df["n"], df["avg_memory_mb"], marker="o")
    plt.xlabel("Problem size n"); plt.ylabel("Average memory (MB)")
    plt.title("Scalability: Memory vs size")
    p2 = os.path.join(out_dir, f"scal_memory_{ts()}.png")
    plt.tight_layout(); plt.savefig(p2, dpi=300); plt.close()

    if (df["avg_runtime_sec"] > 0).all():
        logn = np.log(df["n"].values.astype(float))
        logr = np.log(df["avg_runtime_sec"].values.astype(float))
        slope = float(np.polyfit(logn, logr, 1)[0])
        plt.figure()
        plt.plot(logn, logr, marker="o")
        plt.xlabel("log(n)"); plt.ylabel("log(runtime)")
        plt.title(f"Scalability: log-log slope ≈ {slope:.2f}")
        p3 = os.path.join(out_dir, f"scal_loglog_{ts()}.png")
        plt.tight_layout(); plt.savefig(p3, dpi=300); plt.close()
        print(f"[Scalability] Estimated scaling exponent ≈ {slope:.2f}")

    plt.figure()
    plt.errorbar(df["n"], df["avg_runtime_sec"], yerr=df["std_runtime_sec"], fmt="o-")
    plt.xlabel("Problem size n"); plt.ylabel("Avg runtime (s) ± 1 std")
    plt.title("Scalability: Runtime dispersion")
    p4 = os.path.join(out_dir, f"scal_runtime_err_{ts()}.png")
    plt.tight_layout(); plt.savefig(p4, dpi=300); plt.close()

    print(f"[Scalability] Saved plots to {out_dir}")


# ------------********------------ Reliability Benchmarks

def reliability_benchmark(args, out_dir: str, rng: np.random.Generator) -> None:
    print("[Reliability][Quantum] Starting benchmark…")
    ensure_dir(out_dir)
    if args.rel_size > MAX_QUBITS:
        raise ValueError(f"[Reliability] rel_size={args.rel_size} exceeds MAX_QUBITS={MAX_QUBITS}. "
                         f"Use e.g. --rel_size 12")

    n = args.rel_size
    seed_graph = args.seed + 4242
    mat = random_weighted_graph(n, 0.5, seed=seed_graph) if args.problem == "maxcut" else random_qubo(n, 0.3, seed=seed_graph)

    runs = args.rel_runs
    results = []

    for r in range(runs):
        seed = args.seed + r
        x0_dummy = np.zeros(n, dtype=np.int64)
        cfg = CQAOAConfig(p_layers=args.layers, gamma_grid=[0.2, 0.6, 1.0], beta_grid=[0.2, 0.6, 1.0], seed=seed)
        t0 = time.perf_counter()
        _, best_obj, _ = cqaoa_run(x0_dummy, args.problem, mat, cfg, np.random.default_rng(seed))
        t1 = time.perf_counter()
        results.append({"run_id": r, "objective_value": best_obj, "runtime_sec": (t1 - t0), "seed": seed})

    df = pd.DataFrame(results)
    best = df["objective_value"].max()
    tolerances = [0.01, 0.03, 0.05, 0.10]
    succ_rows = []
    for tau in tolerances:
        thr = best * (1 - tau)
        succ_rows.append({"tolerance": tau, "success_rate": float((df["objective_value"] >= thr).mean())})
    df_succ = pd.DataFrame(succ_rows)

    excel_path = os.path.join(out_dir, f"reliability_{ts()}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="runs")
        df_succ.to_excel(writer, index=False, sheet_name="success_rates")
    print(f"[Reliability] Saved Excel -> {excel_path}")

    plt.figure()
    plt.boxplot(df["objective_value"], vert=True)
    plt.ylabel("QAOA expectation (higher is better)")
    plt.title(f"Reliability: Objective dispersion (n={n})")
    p1 = os.path.join(out_dir, f"rel_box_{ts()}.png")
    plt.tight_layout(); plt.savefig(p1, dpi=300); plt.close()

    plt.figure()
    plt.hist(df["objective_value"], bins=min(10, runs))
    plt.xlabel("QAOA expectation"); plt.ylabel("Frequency")
    plt.title("Reliability: Objective histogram")
    p2 = os.path.join(out_dir, f"rel_hist_{ts()}.png")
    plt.tight_layout(); plt.savefig(p2, dpi=300); plt.close()

    plt.figure()
    plt.plot([int(t*100) for t in df_succ["tolerance"]], df_succ["success_rate"], marker="o")
    plt.xlabel("Tolerance from best (%)"); plt.ylabel("Success rate"); plt.ylim(0, 1)
    plt.title("Reliability: Success rate vs tolerance")
    p3 = os.path.join(out_dir, f"rel_success_{ts()}.png")
    plt.tight_layout(); plt.savefig(p3, dpi=300); plt.close()

    plt.figure()
    plt.hist(df["runtime_sec"], bins=min(10, runs))
    plt.xlabel("Runtime (s)"); plt.ylabel("Frequency")
    plt.title("Reliability: Runtime distribution")
    p4 = os.path.join(out_dir, f"rel_runtime_hist_{ts()}.png")
    plt.tight_layout(); plt.savefig(p4, dpi=300); plt.close()

    print(f"[Reliability] Saved plots to {out_dir}")

# ------------********------------ Performance Based Benchmarks for Carbon

def carbon_benchmark(args, out_dir: str, perf_summary: Dict[str, float]) -> None:
    print("[Carbon] Starting benchmark…")
    ensure_dir(out_dir)
    co2_path = args.co2file or args.excel or auto_find_co2_on_desktop()
    if not co2_path or not os.path.exists(co2_path):
        print("[Carbon][Warning] CO2 file not found. Writing placeholder 0-emissions workbook.")
        df_out = pd.DataFrame({"country": ["N/A"], "gco2_per_kwh": [0.0], "year": [np.nan],
                               "energy_kWh": [0.0], "emissions_kgCO2": [0.0]})
        excel_path = os.path.join(out_dir, f"carbon_{ts()}.xlsx")
        save_df_excel(df_out, excel_path, sheet_name="carbon")
        return

    dfc = read_co2_table(co2_path)
    country_col = choose_country_column(dfc)
    intensity_col = choose_intensity_column(dfc)
    year_col = choose_year_column(dfc)
    if country_col is None or intensity_col is None:
        raise ValueError("Could not identify 'country' and intensity (gCO2/kWh) columns in the CO2 file.")

    if str(args.year_select).lower() == "latest":
        dfc_sel = latest_per_country(dfc, year_col, country_col)
    else:
        try:
            want_year = int(args.year_select)
            if year_col is None:
                print("[Carbon][Note] No year column; cannot filter to a specific year—using all rows.")
                dfc_sel = dfc
            else:
                yn = pd.to_numeric(dfc[year_col], errors="coerce")
                dfc_sel = dfc[yn == want_year]
        except Exception:
            dfc_sel = latest_per_country(dfc, year_col, country_col)

    cols = [country_col, intensity_col] + ([year_col] if (year_col and year_col in dfc_sel.columns) else [])
    df_c = dfc_sel[cols].copy()
    df_c.columns = ["country", "gco2_per_kwh"] + (["year"] if (year_col and year_col in dfc_sel.columns) else [])
    df_c["gco2_per_kwh"] = pd.to_numeric(df_c["gco2_per_kwh"], errors="coerce")
    df_c = df_c.dropna(subset=["country", "gco2_per_kwh"])

    if args.countries:
        wanted = set([c.strip().lower() for c in args.countries.split(",") if c.strip()])
        df_c = df_c[df_c["country"].str.lower().isin(wanted)].copy()

    total_sec = float(perf_summary["total_runtime_sec"])
    energy_kWh = (args.device_power_watts * args.pue) * (total_sec / 3600.0) / 1000.0

    df_c["energy_kWh"] = energy_kWh
    df_c["emissions_kgCO2"] = (df_c["gco2_per_kwh"] * df_c["energy_kWh"]) / 1000.0
    df_c = df_c.sort_values("emissions_kgCO2", ascending=False).reset_index(drop=True)

    excel_path = os.path.join(out_dir, f"carbon_{ts()}.xlsx")
    save_df_excel(df_c, excel_path, sheet_name="carbon")
    print(f"[Carbon] Saved Excel -> {excel_path}")

    top = df_c.head(15)
    plt.figure()
    plt.bar(top["country"].astype(str), top["emissions_kgCO2"])
    plt.xticks(rotation=60, ha="right"); plt.ylabel("kg CO₂")
    plt.title("Carbon: Emissions by country (top 15, latest year)")
    p1 = os.path.join(out_dir, f"carbon_top_{ts()}.png")
    plt.tight_layout(); plt.savefig(p1, dpi=300); plt.close()

    plt.figure()
    plt.scatter(df_c["gco2_per_kwh"], df_c["emissions_kgCO2"])
    plt.xlabel("Grid intensity (gCO₂/kWh)"); plt.ylabel("Emissions (kg CO₂)")
    plt.title("Carbon: Intensity vs Emissions")
    p2 = os.path.join(out_dir, f"carbon_scatter_{ts()}.png")
    plt.tight_layout(); plt.savefig(p2, dpi=300); plt.close()

    energy_Wh = (args.device_power_watts * args.pue) * (total_sec / 3600.0)
    plt.figure()
    plt.bar(["Device Power (W)", "PUE"], [args.device_power_watts, args.pue])
    plt.title("Carbon: Device & PUE (context)")
    p3 = os.path.join(out_dir, f"carbon_context1_{ts()}.png")
    plt.tight_layout(); plt.savefig(p3, dpi=300); plt.close()

    plt.figure()
    plt.bar(["Energy (Wh)"], [energy_Wh])
    plt.title("Carbon: Energy from Performance runtime")
    p4 = os.path.join(out_dir, f"carbon_context2_{ts()}.png")
    plt.tight_layout(); plt.savefig(p4, dpi=300); plt.close()

    if args.combine:
        meta = pd.DataFrame([{
            "total_runtime_sec": total_sec,
            "device_power_watts": args.device_power_watts,
            "pue": args.pue,
            "energy_kWh": energy_kWh,
            "countries_count": int(df_c.shape[0]),
            "source_file": os.path.abspath(co2_path),
            "year_mode": str(args.year_select)
        }])
        combined_path = os.path.join(out_dir, f"carbon_combined_{ts()}.xlsx")
        with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
            meta.to_excel(writer, index=False, sheet_name="summary")
            df_c.to_excel(writer, index=False, sheet_name="emissions_by_country")
        print(f"[Carbon] Saved combined workbook -> {combined_path}")

    print(f"[Carbon] Saved plots to {out_dir}")

# ================================ CLI / Main ================================= #
#This code defines a placeholder dataclass for argument grouping and 
#constructs an argument parser for the full QAOA benchmarking suite.
#The parser configures options for problem type, layers, performance tests, scalability sizes, reliability runs,
# and carbon-intensity inputs.
#It returns a populated namespace containing all CLI arguments, enabling consistent configuration across all benchmark modules.


@dataclass
class CQAOAArgs:
    pass

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ideal (noiseless) QAOA benchmarking (Performance, Scalability, Reliability, Carbon)."
    )
    # Core problem & layers
    p.add_argument("--problem", type=str, default="maxcut", choices=["maxcut", "qubo"])
    p.add_argument("--layers", type=int, default=1)

    # Performance
    p.add_argument("--perf_size", type=int, default=128)  # NOTE: must be <=16 for quantum; else error message explains
    p.add_argument("--perf_instances", type=int, default=5)
    p.add_argument("--perf_plot_mode", type=str, choices=["default", "graphs"], default="graphs")
    p.add_argument("--viz_graph_examples", type=int, default=2)
    p.add_argument("--viz_graph_maxn", type=int, default=80)

    # Scalability
    p.add_argument("--sizes", type=str, default="32,64,96,128")  # will auto-filter to <=16 or raise if none
    p.add_argument("--scal_instances", type=int, default=3)

    # Reliability
    p.add_argument("--rel_size", type=int, default=128)  # must be <=16 for quantum; else error explains
    p.add_argument("--rel_runs", type=int, default=20)

    # Carbon / CO2 inputs
    p.add_argument("--excel", type=str, default=None, help="Path to Excel/CSV with country intensities.")
    p.add_argument("--co2file", type=str, default=None, help="Alias for --excel (CSV/Excel).")
    p.add_argument("--device-power-watts", dest="device_power_watts", type=float, default=65.0)
    p.add_argument("--pue", type=float, default=1.2)
    p.add_argument("--year-select", dest="year_select", type=str, default="latest")
    p.add_argument("--countries", type=str, default=None)
    p.add_argument("--combine", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default=".", help="Root output directory (four folders created within).")
    return p.parse_args()


#This main function coordinates the entire benchmarking pipeline by parsing arguments, preparing output directories, and initializing the RNG.
#It runs the performance, scalability, reliability, and carbon benchmarks in sequence, passing intermediate results where required.
#Finally, it prints completion messages and absolute paths to all generated output folders for user reference.

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    perf_dir = os.path.join(args.outdir, "Performance")
    scal_dir = os.path.join(args.outdir, "Scalability")
    rel_dir  = os.path.join(args.outdir, "Reliability")
    carb_dir = os.path.join(args.outdir, "Carbon")
    for d in (perf_dir, scal_dir, rel_dir, carb_dir):
        ensure_dir(d)

    perf_summary = performance_benchmark(args, perf_dir, rng)
    scalability_benchmark(args, scal_dir, rng)
    reliability_benchmark(args, rel_dir, rng)
    carbon_benchmark(args, carb_dir, perf_summary)

    print("\nAll benchmarks completed.")
    print(f"Outputs saved under:\n  {os.path.abspath(perf_dir)}\n  {os.path.abspath(scal_dir)}\n  {os.path.abspath(rel_dir)}\n  {os.path.abspath(carb_dir)}")

if __name__ == "__main__":
    main()
