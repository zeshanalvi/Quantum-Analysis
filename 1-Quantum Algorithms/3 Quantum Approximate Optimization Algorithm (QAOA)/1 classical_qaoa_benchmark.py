# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# classical_qaoa_benchmark.py
# QAOA-inspired classical heuristic (cQAOA)
# on binary optimization (Max-Cut / QUBO).


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
Run example from Desktop:
  python classical_qaoa_benchmark.py --problem maxcut ^
    --perf_plot_mode graphs --viz_graph_examples 2 --viz_graph_maxn 70 ^
    --excel "Filtered CO2 intensity 236 Countries.xlsx" ^
    --device-power-watts 65 --pue 1.2 --year-select latest ^
    --outdir carbon_by_country --combine
"""

# ------------********------------ Imports

from __future__ import annotations     # Enables postponed evaluation of type hints for forward references
import argparse                        # Used for parsing command-line arguments
import glob                            # Provides file path pattern matching (e.g., *.txt)
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np                     # Numerical computing library used for arrays and math operations
import pandas as pd                    # Data analysis library for working with DataFrames
import matplotlib
matplotlib.use("Agg")  # ensure headless save on Windows/servers  # Prevents GUI usage; allows saving plots in headless environments
import matplotlib.pyplot as plt
import psutil                          # Library for retrieving system and process information (CPU, RAM, etc.)
import networkx as nx                  # Library for creating and analyzing complex networks/graphs



# ------------********------------ Paths & I/O 

def desktop_path() -> str:
    return os.path.join(os.path.expanduser("~"), "Desktop")   # Build full path to user's Desktop folder


#This function retrieves the Desktop path and prepares several filename patterns for CO₂-related files.
#It scans the Desktop for matches to these patterns and returns the first file found.
#If no matching files exist across all patterns, the function returns None.

def auto_find_co2_on_desktop() -> Optional[str]:
    desk = desktop_path()                                   
    pats = [
        "*co2*.xlsx", "*CO2*.xlsx", "*co2*.csv", "*CO2*.csv",
        "Filtered CO2 intensity*.xlsx"
    ]
    for p in pats:                                           
        hits = glob.glob(os.path.join(desk, p))           
        if hits:
            return hits[0]                                  
    return None

#In the following three functions, the first ensures a directory exists by creating it if necessary.
#The second generates a timestamp string in a fixed “YYYYMMDD_HHMMSS” format.
#The third saves a pandas DataFrame to an Excel file using openpyxl, writing it to the specified sheet without index values.

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def save_df_excel(df: pd.DataFrame, outpath: str, sheet_name: str = "results") -> None:
    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:   # Create an Excel writer using openpyxl
        df.to_excel(writer, index=False, sheet_name=sheet_name)



# ------------********------------  CO2 Schema Inference 


#This function attempts to identify a column in the DataFrame that represents a country by checking for common country-related names.
#If none of the expected names match, it falls back to returning the first column that contains string data.
#If no suitable column is found, it returns None.

def choose_country_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("country", "name", "nation", "entity", "location"):
            return c  
    for c in df.columns:
        if df[c].dtype == object:     
            return c
    return None

#This function tries to identify the CO₂-intensity column by first checking for exact matches against a list of preferred column names.
#If no exact match is found, it then looks for heuristic matches containing both “co2” and “kwh” in the column name.
#As a final fallback, it returns the first numeric column, or None if no numeric columns exist.

def choose_intensity_column(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "gco2_per_kwh", "co2_g_per_kwh", "grid_intensity_g_per_kwh",
        "intensity_gco2_per_kwh", "g_co2_per_kwh", "g_per_kwh", "gco2kwh"
    ]
    lower = {c: str(c).lower() for c in df.columns}  
    for want in preferred:
        for c in df.columns:
            if lower[c] == want:     
                return c
    for c in df.columns:
        lc = lower[c]
        if "co2" in lc and "kwh" in lc:   
            return c
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else None    

#This function attempts to locate a year column by first checking for direct matches such as “year” or “yr.”
#If not found, it falls back to broader time-related names like “date,” “time,” or “period.”
#If no suitable column is detected, it returns None.

def choose_year_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("year", "yr"):
            return c  
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("date", "time", "period"):  
            return c
    return None

def read_co2_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()  
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")



#This function extracts the latest available record per country, using the year column when present.
#If a valid year column doesn’t exist, it simply drops missing country values and keeps the last occurrence for each country.
#When a year column is available, it converts it to numeric, identifies the maximum year per country, and returns those rows.

def latest_per_country(df: pd.DataFrame, year_col: Optional[str], country_col: str) -> pd.DataFrame:
    if (year_col is None) or (year_col not in df.columns):
        return df.dropna(subset=[country_col]).drop_duplicates(subset=[country_col], keep="last")
    y = pd.to_numeric(df[year_col], errors="coerce")   
    df = df.copy()
    df["_year_num"] = y
    idx = df.groupby(country_col)["_year_num"].idxmax() 
    out = df.loc[idx].drop(columns=["_year_num"])
    return out

# ------------********------------ CPU / Memory Sampling 

#The following class runs a background thread that periodically samples CPU usage and stores the collected values.
#It supports starting, stopping, and computing statistics such as average and maximum CPU utilization from the samples.
#The function below the class code, retrieves the current process’s memory consumption in megabytes.

class CPUUsageSampler(threading.Thread):
    def __init__(self, interval_sec: float = 0.2):
        super().__init__(daemon=True)                    
        self.interval = interval_sec
        self.samples: List[float] = []                   
        self._stop = threading.Event()                   
    def run(self):
        psutil.cpu_percent(interval=None)                
        while not self._stop.is_set():
            self.samples.append(float(psutil.cpu_percent(interval=self.interval))) 
    def stop(self):
        self._stop.set()                                
    def avg_util(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0
    def max_util(self) -> float:
        return float(np.max(self.samples)) if self.samples else 0.0

def current_mem_mb() -> float:
    proc = psutil.Process(os.getpid())                   
    return proc.memory_info().rss / (1024.0 ** 2)      



# ------------********------------ Problem Generation & Costs


#This function generates a random weighted undirected graph using a reproducible RNG seed.
#Edges are created based on a probability mask, assigned random weights, and mirrored to form a symmetric adjacency matrix.
#The diagonal is zeroed to remove self-loops, and the final matrix is returned as floats.

def random_weighted_graph(n: int, edge_prob: float = 0.5, w_low: int = 1, w_high: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)                         
    A = rng.random((n, n))
    W = (A < edge_prob).astype(float)                         
    weights = rng.integers(w_low, w_high + 1, size=(n, n))    
    W = W * weights                                          
    W = np.triu(W, 1)                                        
    W = W + W.T                                              
    np.fill_diagonal(W, 0.0)
    return W.astype(float)

def maxcut_cost(x: np.ndarray, W: np.ndarray) -> float:
    s = 1 - 2 * x  # x in {0,1} -> s in {+1,-1}                # Convert binary vector to ±1 spin representation
    return 0.25 * (np.sum(W) - np.sum(W * np.outer(s, s)))     # Standard MaxCut cost formula

def delta_cost_if_flip_maxcut(i: int, x: np.ndarray, W: np.ndarray) -> float:
    s = 1 - 2 * x                                              # Convert bitstring to ±1 form
    return float(s[i] * np.sum(W[i, :] * s))                   # Compute cost change if node i is flipped

def random_qubo(n: int, density: float = 0.3, q_low: int = -5, q_high: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)                          # Initialize random generator
    mask = rng.random((n, n)) < density                        # Create sparsity mask for QUBO matrix
    Q = rng.integers(q_low, q_high + 1, size=(n, n)) * mask    # Apply mask to random Q entries
    Q = np.triu(Q)                                             # Keep only upper triangle
    Q = Q + Q.T - np.diag(np.diag(Q))                          # Symmetrize and remove duplicated diagonal
    return Q.astype(float)

def qubo_cost(x: np.ndarray, Q: np.ndarray) -> float:
    return float(x @ Q @ x)

def delta_cost_if_flip_qubo(i: int, x: np.ndarray, Q: np.ndarray) -> float:
    x_new = x.copy()
    x_new[i] = 1 - x[i]
    return qubo_cost(x_new, Q) - qubo_cost(x, Q)


# ------------********------------  cQAOA Heuristic

#This dataclass stores configuration parameters for a QAOA-like algorithm, including layers, parameter grids, and RNG seed.
#If gamma or beta grids are not provided, the post-initialization step fills them with default evenly spaced values.
#This ensures the configuration is always initialized with usable optimization parameter ranges.

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

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# This function performs a grid-search CQAOA run, selecting cost and delta functions 
# based on whether the task is MaxCut or QUBO.
# For each combination of γ and β, it iteratively applies probabilistic flip and mixing operations across p layers, 
# guided by normalized cost deltas.
# It tracks and returns the best solution found, its objective value, and metadata describing the parameter sweep.

def cqaoa_run(x0: np.ndarray, problem: str, mat: np.ndarray, cfg: CQAOAConfig, rng: np.random.Generator
             ) -> Tuple[np.ndarray, float, Dict[str, float]]:
    if problem == "maxcut":
        cost_fn = lambda x: maxcut_cost(x, mat)
        delta_fn = lambda i, x: delta_cost_if_flip_maxcut(i, x, mat)
    elif problem == "qubo":
        cost_fn = lambda x: -qubo_cost(x, mat) 
        delta_fn = lambda i, x: -delta_cost_if_flip_qubo(i, x, mat)
    else:
        raise ValueError("Unsupported problem type.")

    n = len(x0)
    best_x = x0.copy()
    best_c = cost_fn(best_x)

    for gamma in cfg.gamma_grid:
        for beta in cfg.beta_grid:
            x = x0.copy()
            for _ in range(cfg.p_layers):
                deltas = np.array([delta_fn(i, x) for i in range(n)], dtype=float)
                nd = (deltas - deltas.mean()) / (deltas.std() + 1e-9) if deltas.std() > 1e-12 else deltas * 0.0
                p_flip = sigmoid(gamma * nd)
                flips = rng.random(n) < p_flip
                x = (x ^ flips.astype(np.int64))
                p_mix = (math.sin(beta) ** 2)
                mix_mask = rng.random(n) < p_mix
                x = (x ^ mix_mask.astype(np.int64))
            cval = cost_fn(x)
            if cval > best_c:
                best_c, best_x = cval, x
    meta = {"gamma_candidates": len(cfg.gamma_grid), "beta_candidates": len(cfg.beta_grid), "layers": cfg.p_layers}
    return best_x, float(best_c), meta


# ------------********------------  Max-Cut Graph Visualization 

# This function visualizes a Max-Cut solution by constructing a graph 
# where nodes belong to two partitions and edges indicate whether they are cut.
# It draws nodes, cut edges, and internal edges with widths scaled by weight, 
# then annotates and saves the resulting plot to file.
# The displayed title includes the computed weighted cut value for the given partition.

def plot_maxcut_graph(W: np.ndarray, x: np.ndarray, out_path: str, title: str = "") -> None:
    G = nx.Graph()
    n = W.shape[0]
    for i in range(n):
        G.add_node(i, part=int(x[i]))                                
    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if w > 0.0:
                G.add_edge(i, j, weight=w, cut=(x[i] != x[j]))        
    pos = nx.spring_layout(G, seed=42)                                
    node_colors = ["tab:blue" if G.nodes[i]["part"] == 0 else "tab:orange" for i in G.nodes()]
    cut_edges = [(u, v) for u, v, d in G.edges(data=True) if d["cut"]] 
    in_edges  = [(u, v) for u, v, d in G.edges(data=True) if not d["cut"]]
    all_w = [G[u][v]["weight"] for u, v in G.edges()] or [1.0]
    maxw = max(all_w)
    lw_cut = [1.0 + 3.0 * (G[u][v]["weight"] / maxw) for u, v in cut_edges] 
    lw_in  = [0.5 + 2.0 * (G[u][v]["weight"] / maxw) for u, v in in_edges]
    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(G, pos, node_size=140, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=in_edges, width=lw_in, alpha=0.45)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=lw_cut, style="dashed")
    nx.draw_networkx_labels(G, pos, font_size=8)
    cut_val = maxcut_cost(x, W)                                       
    plt.title(title or f"Max-Cut partition (weighted cut = {cut_val:.2f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)                                   
    plt.close()

# ------------********------------ Benchmarks

# ------------********------------ Performance Benchmarks

#This function runs a performance benchmark by generating problem instances, executing the CQAOA solver, 
# and recording metrics such as runtime, memory, and objective value.
#It also samples CPU usage during execution, optionally produces Max-Cut graph visualizations, 
# and aggregates all run data into a DataFrame.
#Finally, it saves the benchmark results to an Excel file and reports the output location.

def performance_benchmark(args, out_dir: str, rng: np.random.Generator) -> Dict[str, float]:
    print("[Performance] Starting benchmark...")
    ensure_dir(out_dir)

    records = []
    pngs = []
    graph_imgs = []

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

        x0 = np.random.default_rng(seed).integers(0, 2, size=n).astype(np.int64)
        cfg = CQAOAConfig(p_layers=args.layers,
                          gamma_grid=[0.2, 0.4, 0.6, 0.8, 1.0],
                          beta_grid=[0.2, 0.4, 0.6, 0.8, 1.0],
                          seed=seed)

        run_t0 = time.perf_counter()
        best_x, best_c, meta = cqaoa_run(x0, args.problem, mat, cfg, np.random.default_rng(seed))
        run_t1 = time.perf_counter()

        records.append({
            "instance_id": inst,
            "n": n,
            "runtime_sec": run_t1 - run_t0,
            "memory_mb": current_mem_mb(),
            "objective_value": best_c,
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

    # Plots (2–4 total)
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
        plt.xlabel("Instance"); plt.ylabel("Objective (higher is better)")
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
        if graph_imgs:
            pngs.extend(graph_imgs[:2])
        else:
            plt.figure()
            plt.plot(df["instance_id"], df["objective_value"], marker="o", linestyle="-")
            plt.xlabel("Instance"); plt.ylabel("Objective (higher is better)")
            plt.title("Performance: Objective values")
            p3 = os.path.join(out_dir, f"perf_objective_{ts()}.png")
            plt.tight_layout(); plt.savefig(p3, dpi=300); plt.close()

            plt.figure()
            plt.bar(["Avg CPU", "Max CPU"], [avg_cpu, max_cpu])
            plt.ylabel("CPU Utilization (%)"); plt.title("Performance: CPU utilization during runs")
            p4 = os.path.join(out_dir, f"perf_cpu_{ts()}.png")
            plt.tight_layout(); plt.savefig(p4, dpi=300); plt.close()
            pngs.extend([p3, p4])

    print(f"[Performance] Saved {len(pngs)} plots to {out_dir}")

    return {
        "total_runtime_sec": float(total_runtime),
        "avg_cpu_util_pct": float(avg_cpu),   # for reporting only (not used in Carbon calc)
        "mem_mb_mean": float(df["memory_mb"].mean()),
        "excel_path": excel_path,
        "plots": pngs
    }


# ------------********------------ Scalability Benchmarks

# This function benchmarks how runtime and memory scale with problem size by running multiple CQAOA instances 
# for each specified n and recording performance statistics.
# It saves the aggregated data to Excel, generates several plots—runtime, memory, log–log scaling, 
# and runtime dispersion—and stores them in the output directory.
# The function also estimates a scaling exponent from log–log regression to characterize algorithmic growth behavior.

def scalability_benchmark(args, out_dir: str, rng: np.random.Generator) -> None:
    print("[Scalability] Starting benchmark...")
    ensure_dir(out_dir)
    records = []
    sizes = [int(s) for s in args.sizes.split(",")]
    per_size = args.scal_instances

    for n in sizes:
        rt, mm = [], []
        for k in range(per_size):
            seed = args.seed + 1000 + n * 10 + k
            mat = random_weighted_graph(n, 0.5, seed=seed) if args.problem == "maxcut" else random_qubo(n, 0.3, seed=seed)
            x0 = np.random.default_rng(seed).integers(0, 2, size=n).astype(np.int64)
            cfg = CQAOAConfig(p_layers=args.layers, gamma_grid=[0.2, 0.6, 1.0], beta_grid=[0.2, 0.6, 1.0], seed=seed)
            t0 = time.perf_counter()
            _, best_c, _ = cqaoa_run(x0, args.problem, mat, cfg, np.random.default_rng(seed))
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

#This function evaluates solver reliability by running many trials on the same graph 
# and recording each run’s objective value and runtime.
#It computes success rates at multiple tolerances from the best solution, 
# saves detailed and summary results to Excel, 
# and produces dispersion plots.
#Additional visualizations—boxplots, histograms, and success-rate curves—help reveal stability and variability across repeated runs.

def reliability_benchmark(args, out_dir: str, rng: np.random.Generator) -> None:
    print("[Reliability] Starting benchmark...")
    ensure_dir(out_dir)
    n = args.rel_size
    seed_graph = args.seed + 4242
    mat = random_weighted_graph(n, 0.5, seed=seed_graph) if args.problem == "maxcut" else random_qubo(n, 0.3, seed=seed_graph)

    runs = args.rel_runs
    results = []

    for r in range(runs):
        seed = args.seed + r
        x0 = np.random.default_rng(seed).integers(0, 2, size=n).astype(np.int64)
        cfg = CQAOAConfig(p_layers=args.layers, gamma_grid=[0.2, 0.6, 1.0], beta_grid=[0.2, 0.6, 1.0], seed=seed)
        t0 = time.perf_counter()
        _, best_c, _ = cqaoa_run(x0, args.problem, mat, cfg, np.random.default_rng(seed))
        t1 = time.perf_counter()
        results.append({"run_id": r, "objective_value": best_c, "runtime_sec": (t1 - t0), "seed": seed})

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
    plt.ylabel("Objective (higher is better)")
    plt.title(f"Reliability: Objective dispersion (n={n})")
    p1 = os.path.join(out_dir, f"rel_box_{ts()}.png")
    plt.tight_layout(); plt.savefig(p1, dpi=300); plt.close()

    plt.figure()
    plt.hist(df["objective_value"], bins=min(10, runs))
    plt.xlabel("Objective"); plt.ylabel("Frequency")
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

# ------------********------------ Carbon Benchmarks
# =============================== Carbon (CO₂) 

def carbon_benchmark(args, out_dir: str, perf_summary: Dict[str, float]) -> None:
    """
    Emissions use ONLY Performance's total runtime:
      Energy (kWh) = device_power_watts * PUE * (total_runtime_sec / 3600) / 1000
      Emissions (kg CO2) = Energy(kWh) * (gCO2/kWh) / 1000
    """
    print("[Carbon] Starting benchmark...")
    ensure_dir(out_dir)

    # Resolve input file
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

    # Select latest or specific year
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

    # Clean/select columns
    cols = [country_col, intensity_col] + ([year_col] if (year_col and year_col in dfc_sel.columns) else [])
    df_c = dfc_sel[cols].copy()
    df_c.columns = ["country", "gco2_per_kwh"] + (["year"] if (year_col and year_col in dfc_sel.columns) else [])
    df_c["gco2_per_kwh"] = pd.to_numeric(df_c["gco2_per_kwh"], errors="coerce")
    df_c = df_c.dropna(subset=["country", "gco2_per_kwh"])

    # Optional country subset
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

    # Plots (2–4)
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


# ------------********------------ CLI / Main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="cQAOA benchmarking (Performance, Scalability, Reliability, Carbon) with optional Max-Cut visuals and latest-year CO2."
    )
    # Core problem
    p.add_argument("--problem", type=str, default="maxcut", choices=["maxcut", "qubo"])
    p.add_argument("--layers", type=int, default=1)

    # Performance
    p.add_argument("--perf_size", type=int, default=128)
    p.add_argument("--perf_instances", type=int, default=5)
    p.add_argument("--perf_plot_mode", type=str, choices=["default", "graphs"], default="graphs",
                   help="In Performance: 'graphs' swaps in up to 2 Max-Cut visuals (if problem=maxcut).")
    p.add_argument("--viz_graph_examples", type=int, default=2)
    p.add_argument("--viz_graph_maxn", type=int, default=80)

    # Scalability
    p.add_argument("--sizes", type=str, default="32,64,96,128")
    p.add_argument("--scal_instances", type=int, default=3)

    # Reliability
    p.add_argument("--rel_size", type=int, default=128)
    p.add_argument("--rel_runs", type=int, default=20)

    # Carbon / CO2 inputs (with my preferred flags)
    p.add_argument("--excel", type=str, default=None, help="Path to Excel/CSV with country intensities.")
    p.add_argument("--co2file", type=str, default=None, help="Alias for --excel (CSV/Excel).")
    p.add_argument("--device-power-watts", dest="device_power_watts", type=float, default=65.0,
                   help="device power in watts (default 65).")
    p.add_argument("--pue", type=float, default=1.2, help="Power Usage Effectiveness multiplier.")
    p.add_argument("--year-select", dest="year_select", type=str, default="latest",
                   help="'latest' or a specific year, e.g., 2021.")
    p.add_argument("--countries", type=str, default=None, help="Comma-separated list to filter countries.")
    p.add_argument("--combine", action="store_true", help="Write a combined Carbon workbook (summary + results).")

    # Misc
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default=".", help="Root output directory (four folders created within).")
    return p.parse_args()


#This main function orchestrates the entire benchmarking pipeline 
# by parsing arguments, seeding RNG, creating output directories, and running all benchmark types.
#It executes performance, scalability, reliability, and carbon benchmarks in sequence, passing outputs where needed 
# (e.g., carbon uses performance results).
#After completion, it prints the locations of all generated benchmark outputs for user reference.

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)                         # RNG seeded for reproducible benchmark data

    # Prepare metric-specific folders under outdir
    perf_dir = os.path.join(args.outdir, "Performance")
    scal_dir = os.path.join(args.outdir, "Scalability")
    rel_dir  = os.path.join(args.outdir, "Reliability")
    carb_dir = os.path.join(args.outdir, "Carbon")
    for d in (perf_dir, scal_dir, rel_dir, carb_dir):
        ensure_dir(d)                                              # Create each directory if not present

    perf_summary = performance_benchmark(args, perf_dir, rng)      # Run performance benchmark and collect summary
    scalability_benchmark(args, scal_dir, rng)
    reliability_benchmark(args, rel_dir, rng)
    carbon_benchmark(args, carb_dir, perf_summary)                 # Carbon calculations depend on performance results

    print("\nAll benchmarks completed.")
    print(f"Outputs saved under:\n  {os.path.abspath(perf_dir)}\n  {os.path.abspath(scal_dir)}\n  {os.path.abspath(rel_dir)}\n  {os.path.abspath(carb_dir)}")

if __name__ == "__main__":
    main()                                                          # Entry point when script is executed directly

