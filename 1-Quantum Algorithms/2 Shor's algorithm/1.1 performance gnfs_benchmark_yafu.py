# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# shor_algorithm_benchmarks_for_classical_computer
# Empirical evaluation of Shor's algorithm

# Code focuses on Performance


# ------------********------------   Imports

import argparse
import math
import os
import sys
import time
import statistics
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout  # For parallel execution & timeout handling

import pandas as pd
import matplotlib
matplotlib.use("Agg")            # Use non-interactive backend for saving plots in scripts/servers
import matplotlib.pyplot as plt   # Plotting interface

# ------------********------------ PRIME PAIRS (given)

PRIME_PAIRS = [
    (1000003, 1000033),   # ~1.0e12 composite numbers for Shor's algorithm tests
    (1000003, 1000037),
    (999983,  1000003),
    (1065023, 1065079),
    (1299709, 1300033),
]

# ------------********------------ Qiskit import robustness

def _import_qiskit_primitives():
   
    # Return (backend, ShorCtor, env_note). Tries multiple Qiskit import paths
    # to maximize compatibility across versions/installations.
   
    try:
        from qiskit_aer import Aer
        backend = Aer.get_backend("aer_simulator")      # Attempt Aer (modern layout)
        try:
            from qiskit_algorithms import Shor          # Preferred Shor class
            return backend, Shor, "qiskit_algorithms + qiskit_aer"
        except Exception:
            pass
    except Exception:
        backend = None

    try:
        try:
            from qiskit import Aer as AerOld            # Older Qiskit layout
        except Exception:
            from qiskit_aer import Aer as AerOld
        backend2 = AerOld.get_backend("aer_simulator")
        try:
            from qiskit.algorithms import Shor          # Legacy namespace for algorithms
            return backend2, Shor, "qiskit.algorithms + Aer"
        except Exception:
            pass
    except Exception:
        pass

    try:
        from qiskit.providers.aer import AerSimulator   # Newer AerSimulator class
        backend3 = AerSimulator()
        try:
            from qiskit.algorithms import Shor
            return backend3, Shor, "qiskit.algorithms + AerSimulator"
        except Exception:
            pass
        try:
            from qiskit_algorithms import Shor
            return backend3, Shor, "qiskit_algorithms + AerSimulator"
        except Exception:
            pass
    except Exception:
        pass

    raise ImportError(
        "No usable combination of Qiskit Aer + Shor found.\n"
        "Install with:  pip install qiskit qiskit-aer qiskit-algorithms\n"
        "or:            pip install 'qiskit[algorithms]'"
    )


# ------------********------------ Helpers

def human_ts():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")  # Produce UTC timestamp in ISO format

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)  # Create directory if it doesn't already exist
    return path

def desktop_path():
    return os.path.join(os.path.expanduser("~"), "Desktop")  # Resolve user's desktop folder

def build_semiprimes_from_pairs(pairs):
    return [p*q for (p, q) in pairs]  # Multiply prime pairs -> semiprimes for Shor tests

def digits_bits(n: int):
    return (int(math.log10(n)) + 1, n.bit_length())  # Return (decimal digits, binary bits)

def save_plot(fig, path):
    fig.tight_layout()             # Adjust spacing before saving
    fig.savefig(path, dpi=220)     # Save figure with high resolution
    plt.close(fig)                 # Free memory by closing the figure


# ------------********------------ Shor execution (subprocess)

def _run_one_shor(N: int, seed: int = None, shots: int = 2048):
    
    # One Shor factorization on ideal simulator with broad API compatibility.
    # Returns dict: ok, wall_s, factors, detail, env_note
    
    t0 = time.perf_counter()
    backend, ShorCtor, env_note = _import_qiskit_primitives()
    shor = ShorCtor()
    result = None
    try:
        # Modern API
        try:
            result = shor.factor(N, a=None, backend=backend)
        except TypeError:
            result = None

        # QuantumInstance API
        if result is None:
            try:
                from qiskit.utils import QuantumInstance
                qi = QuantumInstance(backend=backend, shots=shots, seed_simulator=seed, seed_transpiler=seed)
                result = shor.factor(N, a=None, quantum_instance=qi)
            except Exception:
                result = None

        # Legacy fallback
        if result is None:
            try:
                if hasattr(shor, "quantum_instance"):
                    shor.quantum_instance = backend
                result = shor.factor(N, a=None)
            except Exception:
                result = None

        factors = []
        if result is not None:
            if isinstance(result, dict):
                if "factors" in result and result["factors"]:
                    if isinstance(result["factors"], dict) and N in result["factors"]:
                        factors = result["factors"][N]
                    elif isinstance(result["factors"], (list, tuple)):
                        inner = result["factors"][0] if result["factors"] else []
                        factors = list(inner) if inner else []
            else:
                if hasattr(result, "factors") and result.factors:
                    if isinstance(result.factors, dict) and N in result.factors:
                        factors = result.factors[N]
                    elif isinstance(result.factors, (list, tuple)):
                        inner = result.factors[0] if result.factors else []
                        factors = list(inner) if inner else []
        ok = bool(factors) and math.prod(factors) == N
        wall_s = time.perf_counter() - t0
        return dict(ok=ok, wall_s=wall_s, factors=factors, detail=str(result), env_note=env_note)
    except Exception as e:
        wall_s = time.perf_counter() - t0
        return dict(ok=False, wall_s=wall_s, factors=[], detail=f"Exception: {e}", env_note=env_note)


# ------------********------------  Benchmarks

def run_performance(Ns, repeats, timeout_s):
    rows = []
    env_used = None
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as pool:   # Run Shor experiments sequentially but safely isolated in a subprocess
        for n in Ns:
            di, bi = digits_bits(n)                    # Get size characteristics (digits, bits)
            ok_times = []
            for r in range(1, repeats + 1):
                start = time.perf_counter()
                fut = pool.submit(_run_one_shor, n, int(time.time() * 1000) % 2**32)  # Launch Shor factoring attempt in worker
                try:
                    res = fut.result(timeout=timeout_s)   # Kill attempt if it exceeds allowed time
                except FuturesTimeout:
                    res = dict(ok=False, wall_s=time.perf_counter() - start,
                               factors=[], detail="timeout", env_note="n/a")

                env_used = env_used or res.get("env_note", "")  # Record which Qiskit environment worked
                status = "ok" if res["ok"] else ("timeout" if "timeout" in str(res.get("detail","")).lower() else "fail")

                if res["ok"]:
                    ok_times.append(res["wall_s"])          # Store successful runtimes

                rows.append({
                    "Timestamp": human_ts(),
                    "N": n,
                    "Digits": di,
                    "Bits": bi,
                    "Run_Index": r,
                    "Wall_Time_s": round(res["wall_s"], 6),
                    "Status": status,
                    "Factors_Found": "x".join(map(str, res.get("factors", []))),
                    "Env": env_used or "",
                })

            if ok_times:
                print(f"[Performance] N={n} avg={statistics.mean(ok_times):.3f}s "   # Report statistics for successful runs
                      f"std={statistics.pstdev(ok_times) if len(ok_times)>1 else 0:.3f}s "
                      f"succ={len(ok_times)}/{repeats}")
            else:
                print(f"[Performance] N={n} : no successful runs")
    return pd.DataFrame(rows)  # Convert collected results to DataFrame


def run_scalability(perf_df):
    ok = perf_df[perf_df["Status"] == "ok"].copy()
    grp = ok.groupby(["N", "Digits", "Bits"])["Wall_Time_s"].agg(["mean", "std", "count"]).reset_index()
    grp.rename(columns={"mean": "Avg_Time_s", "std": "Std_Time_s", "count": "Success_Count"}, inplace=True)
    return grp

def run_reliability(perf_df, repeats):
    rel = perf_df.copy()
    sr = rel.groupby(["N", "Digits", "Bits"])["Status"].apply(lambda s: (s == "ok").mean()).reset_index(name="Success_Rate")
    ok = rel[rel["Status"] == "ok"]
    disp = ok.groupby(["N", "Digits", "Bits"])["Wall_Time_s"].agg(
        Avg_Time_s="mean",
        Std_Time_s="std",
        Min_Time_s="min",
        Max_Time_s="max",
        Median_Time_s="median"
    ).reset_index()
    out = pd.merge(sr, disp, on=["N", "Digits", "Bits"], how="left")
    out["Repeats"] = repeats
    return out


# ------------********------------ CO₂ handling (performance-only)

def _find_co2_file(default_name="Filtered CO2 intensity 236 Countries"):
    
    # Tries Desktop (with/without extension) and CWD; supports .csv/.xlsx.
    
    desk = desktop_path()
    base_candidates = [
        default_name,
        default_name + ".csv",
        default_name + ".xlsx",
    ]
    for b in base_candidates:
        p = os.path.join(desk, b)
        if os.path.isfile(p):
            return p
    # fallback: CWD
    for b in base_candidates:
        p = os.path.join(os.getcwd(), b)
        if os.path.isfile(p):
            return p
    return None

def _read_co2_table(path):
    
    # Reads country carbon intensity. Expected columns (case-insensitive subset):
    #  - Country 
    #  - Year (optional)
    #  - kg_co2_per_kwh  OR  gco2_per_kwh  
    
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Standardize
    cols_lower = {c.lower(): c for c in df.columns}
    # country column
    country_col = cols_lower.get("country")
    if not country_col:
        for c in df.columns:
            if "country" in c.lower():
                country_col = c
                break
    if not country_col:
        raise ValueError("CO₂ file must include a 'Country' column.")

    # intensity column
    kg_col = None
    for c in df.columns:
        if "kg" in c.lower() and "co2" in c.lower() and "kwh" in c.lower():
            kg_col = c
            break
    g_col = None
    if kg_col is None:
        for c in df.columns:
            if "g" in c.lower() and "co2" in c.lower() and "kwh" in c.lower():
                g_col = c
                break
    if kg_col is None and g_col is None:
        raise ValueError("CO₂ file must contain 'kg_co2_per_kwh' or 'gco2_per_kwh'.")

    # year column (optional)
    year_col = None
    for c in df.columns:
        if c.lower() == "year" or "year" in c.lower():
            year_col = c
            break

    # Normalize
    if kg_col:
        df["kg_co2_per_kwh"] = pd.to_numeric(df[kg_col], errors="coerce")
    else:
        df["kg_co2_per_kwh"] = pd.to_numeric(df[g_col], errors="coerce") / 1000.0

    df.rename(columns={country_col: "Country"}, inplace=True)
    if year_col:
        df.rename(columns={year_col: "Year"}, inplace=True)

    return df[["Country", "kg_co2_per_kwh"] + (["Year"] if "Year" in df.columns else [])].dropna(subset=["kg_co2_per_kwh"])

def _select_year(df, year_select):
    if "Year" not in df.columns:
        return df
    if year_select == "latest":
        # pick most recent record per country
        idx = df.groupby("Country")["Year"].idxmax()
        return df.loc[idx].reset_index(drop=True)
    # numeric year
    try:
        y = int(year_select)
        out = df[df["Year"] == y].copy()
        if out.empty:
            # fallback to latest if not present
            idx = df.groupby("Country")["Year"].idxmax()
            return df.loc[idx].reset_index(drop=True)
        return out.reset_index(drop=True)
    except Exception:
        idx = df.groupby("Country")["Year"].idxmax()
        return df.loc[idx].reset_index(drop=True)

def compute_co2(perf_df, watts, pue, co2_df, combine=True):
    
    # Compute emissions using *only* successful performance wall-times.
    # energy_kwh = (wall_time_sum_s * watts * PUE) / 3.6e6
    # If combine=False, returns per-N totals (wide table).
    
    ok = perf_df[perf_df["Status"] == "ok"].copy()
    if combine:
        total_time_s = ok["Wall_Time_s"].sum()
        energy_kwh = (watts * pue * total_time_s) / (1000.0 * 3600.0)
        out = co2_df.copy()
        out["Emissions_kgCO2"] = out["kg_co2_per_kwh"] * energy_kwh
        out.sort_values("Emissions_kgCO2", ascending=False, inplace=True)
        totals = dict(Total_Wall_Time_s=total_time_s, Watts=watts, PUE=pue, Energy_kWh=energy_kwh)
        return out, totals
    else:
        perN = ok.groupby("N")["Wall_Time_s"].sum().reset_index(name="Wall_Time_s")
        perN["Energy_kWh"] = (watts * pue * perN["Wall_Time_s"]) / (1000.0 * 3600.0)
        # Cartesian join with countries
        cross = perN.assign(_key=1).merge(co2_df.assign(_key=1), on="_key").drop("_key", axis=1)
        cross["Emissions_kgCO2"] = cross["kg_co2_per_kwh"] * cross["Energy_kWh"]
        cross.sort_values(["N", "Emissions_kgCO2"], ascending=[True, False], inplace=True)
        totals = dict(Total_Wall_Time_s=perN["Wall_Time_s"].sum(), Watts=watts, PUE=pue,
                      Energy_kWh=perN["Energy_kWh"].sum())
        return cross, totals


# ------------********------------  Plots

def plots_performance(perf_df, out_dir):
    ok = perf_df[perf_df["Status"] == "ok"].copy()

    # 1) Runtime per N (mean ± std)

    g = ok.groupby(["N"])["Wall_Time_s"].agg(["mean", "std"]).reset_index()
    fig = plt.figure()
    plt.errorbar(g["N"], g["mean"], yerr=g["std"].fillna(0.0), fmt="o-")
    plt.xlabel("Semiprime N")
    plt.ylabel("Average wall time (s)")
    plt.title("Shor (Ideal Simulator) — Runtime per N")
    plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir, "runtime_per_N.png"))

    # 2) Runtime vs digits

    g2 = ok.groupby(["Digits"])["Wall_Time_s"].mean().reset_index()
    fig = plt.figure()
    plt.plot(g2["Digits"], g2["Wall_Time_s"], "o-")
    plt.xlabel("Digits of N")
    plt.ylabel("Average wall time (s)")
    plt.title("Runtime vs Digits")
    plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir, "runtime_vs_digits.png"))

    # 3) Success rate per N

    sr = perf_df.groupby(["N"])["Status"].apply(lambda s: (s == "ok").mean()).reset_index(name="Success_Rate")
    fig = plt.figure()
    plt.bar(sr["N"].astype(str), sr["Success_Rate"])
    plt.ylim(0, 1.05)
    plt.ylabel("Success rate")
    plt.xlabel("N")
    plt.title("Success Rate per N (Performance runs)")
    save_plot(fig, os.path.join(out_dir, "success_rate_per_N.png"))

# ------------********------------
def plots_scalability(scal_df, out_dir):
    # 1) Avg runtime per N

    fig = plt.figure()
    plt.plot(scal_df["N"], scal_df["Avg_Time_s"], "o-")
    plt.xlabel("Semiprime N")
    plt.ylabel("Average wall time (s)")
    plt.title("Scalability — Average Runtime per N")
    plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir, "scalability_avg_runtime_per_N.png"))

    # 2) Runtime vs digits

    fig = plt.figure()
    plt.plot(scal_df["Digits"], scal_df["Avg_Time_s"], "o-")
    plt.xlabel("Digits of N")
    plt.ylabel("Average wall time (s)")
    plt.title("Scalability — Runtime vs Digits")
    plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir, "scalability_runtime_vs_digits.png"))

    # 3) Variability (error bars)

    fig = plt.figure()
    plt.errorbar(scal_df["Digits"], scal_df["Avg_Time_s"], yerr=scal_df["Std_Time_s"].fillna(0.0), fmt="o-")
    plt.xlabel("Digits of N")
    plt.ylabel("Avg time ± std (s)")
    plt.title("Scalability — Variability across Repeats")
    plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir, "scalability_variability.png"))

# ------------********------------
def plots_reliability(rel_df, out_dir):

    # 1) Success rate per N
    fig = plt.figure()
    plt.bar(rel_df["N"].astype(str), rel_df["Success_Rate"])
    plt.ylim(0, 1.05)
    plt.xlabel("N")
    plt.ylabel("Success rate")
    plt.title("Reliability — Success Rate per N")
    save_plot(fig, os.path.join(out_dir, "reliability_success_rate.png"))

    # 2) Time dispersion
    fig = plt.figure()
    xs = rel_df["N"].astype(str)
    plt.plot(xs, rel_df["Min_Time_s"], "o-", label="Min")
    plt.plot(xs, rel_df["Median_Time_s"], "o-", label="Median")
    plt.plot(xs, rel_df["Max_Time_s"], "o-", label="Max")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Wall time (s)")
    plt.title("Reliability — Time Dispersion")
    plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir, "reliability_time_dispersion.png"))

# ------------********------------
def plots_carbon(co2_df, totals, out_dir, combine=True):
    if combine:
        top = co2_df.sort_values("Emissions_kgCO2", ascending=False).head(20)

        # 1) Top20 horizontal bar
        fig = plt.figure(figsize=(8, 6))
        plt.barh(top["Country"], top["Emissions_kgCO2"])   # Horizontal bar chart of top emitters
        plt.gca().invert_yaxis()                           # Largest values appear at the top
        plt.xlabel("Emissions (kg CO₂) for workload")
        plt.title("CO₂ by Country — Shor (Ideal Simulator)")
        save_plot(fig, os.path.join(out_dir, "co2_by_country_top20.png"))  # Save plot to file


        # 2) Cumulative
        fig = plt.figure()
        cum = top["Emissions_kgCO2"].cumsum()
        plt.plot(range(1, len(cum)+1), cum, "o-")
        plt.xlabel("Top-k countries")
        plt.ylabel("Cumulative kg CO₂")
        plt.title("Cumulative CO₂ — Top 20 Countries")
        plt.grid(True, alpha=0.4)
        save_plot(fig, os.path.join(out_dir, "co2_cumulative_top20.png"))
    else:

        # Per-N: show one plot for the largest-N breakdown
        lastN = co2_df["N"].iloc[-1]
        sub = co2_df[co2_df["N"] == lastN].sort_values("Emissions_kgCO2", ascending=False).head(15)
        fig = plt.figure(figsize=(8, 6))
        plt.barh(sub["Country"], sub["Emissions_kgCO2"])
        plt.gca().invert_yaxis()
        plt.xlabel(f"Emissions for N={lastN} (kg CO₂)")
        plt.title("CO₂ by Country (per N example)")
        save_plot(fig, os.path.join(out_dir, f"co2_by_country_top15_N{lastN}.png"))

    # 3) Text summary
    fig = plt.figure()
    txt = (f"Total successful wall time: {totals['Total_Wall_Time_s']:.2f} s\n"
           f"Assumed device power: {totals['Watts']} W\n"
           f"PUE: {totals['PUE']}\n"
           f"Total energy: {totals['Energy_kWh']:.6f} kWh")
    plt.axis("off")   # Hide axes to produce a clean text-only summary image
    plt.text(0.02, 0.8, "Carbon Model Summary", fontsize=12, weight="bold")  # Title text
    plt.text(0.02, 0.6, txt, fontsize=10)   # Detailed summary text block
    save_plot(fig, os.path.join(out_dir, "co2_summary.png"))  # Save summary as an image



# ------------********------------  Excel writer (one file per metric)

def write_excel_one(df, out_path, sheet_name="Data"):
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        df.to_excel(xl, index=False, sheet_name=sheet_name)


# ------------********------------ Main Function

def main():
    parser = argparse.ArgumentParser(
        description="Shor on ideal simulator — performance, scalability, reliability, and carbon."
    )
    parser.add_argument("--repeats", type=int, default=5, help="Repetitions per N.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-run timeout (s).")
    parser.add_argument("--Ns", type=str, default="", help="Optional comma-separated semiprimes to override PRIME_PAIRS.")

    # Carbon controls

    parser.add_argument("--year-select", type=str, default="latest",
                        help="'latest' or a specific year (if Year exists).")  # Controls which CO₂ dataset year to use
    parser.add_argument("--device-power-watts", type=float, default=65.0,
                        help="Average device/system power in Watts.")           # Used for energy estimates
    parser.add_argument("--pue", type=float, default=1.2,
                        help="Power Usage Effectiveness factor.")               # Accounts for cooling + overhead
    parser.add_argument("--outdir", type=str, default="carbon_by_country",
                        help="Subfolder inside CarbonFootprints/")
    parser.add_argument("--combine", action="store_true",
                        help="If set, aggregate CO₂ across all runs (recommended).")  # Enables aggregated carbon report
    args = parser.parse_args()


    # Build N set

    if args.Ns.strip():
        try:
            Ns = [int(x.strip()) for x in args.Ns.split(",") if x.strip()]
        except Exception:
            Ns = build_semiprimes_from_pairs(PRIME_PAIRS)
    else:
        Ns = build_semiprimes_from_pairs(PRIME_PAIRS)

    base = os.path.abspath(os.getcwd())
    perf_dir = ensure_dir(os.path.join(base, "Performance"))
    scal_dir = ensure_dir(os.path.join(base, "Scalability"))
    rel_dir  = ensure_dir(os.path.join(base, "Reliability"))
    carb_root = ensure_dir(os.path.join(base, "CarbonFootprints"))
    carb_dir = ensure_dir(os.path.join(carb_root, args.outdir))

    # ------------********------------ Performance

    print("\n=== PERFORMANCE ===")
    perf_df = run_performance(Ns, args.repeats, args.timeout)
    write_excel_one(perf_df, os.path.join(perf_dir, "performance.xlsx"))
    plots_performance(perf_df, perf_dir)
    print(f"Performance → Excel + PNGs in {perf_dir}")

   # ------------********------------ Scalability

    print("\n=== SCALABILITY ===")
    scal_df = run_scalability(perf_df)
    write_excel_one(scal_df, os.path.join(scal_dir, "scalability.xlsx"))
    plots_scalability(scal_df, scal_dir)
    print(f"Scalability → Excel + PNGs in {scal_dir}")

    # ------------********------------ Reliability

    print("\n=== RELIABILITY ===")
    rel_df = run_reliability(perf_df, args.repeats)
    write_excel_one(rel_df, os.path.join(rel_dir, "reliability.xlsx"))
    plots_reliability(rel_df, rel_dir)
    print(f"Reliability → Excel + PNGs in {rel_dir}")

    # ------------********------------ Carbon (performance-only)

    print("\n=== CARBON FOOTPRINTS ===")
    co2_path = _find_co2_file()
    if not co2_path:
        print("WARNING: Could not locate 'Filtered CO2 intensity 236 Countries' on Desktop or CWD.")
        print("         Place the file (CSV/XLSX) on Desktop and re-run.")
    else:
        print(f"Using CO₂ table: {co2_path}")
        co2_raw = _read_co2_table(co2_path)
        co2_tbl = _select_year(co2_raw, args.year_select)
        co2_emissions_df, totals = compute_co2(
            perf_df, watts=args.device_power_watts, pue=args.pue,
            co2_df=co2_tbl, combine=args.combine or True  # combine by default
        )
        write_excel_one(co2_emissions_df, os.path.join(carb_dir, "carbon_footprints.xlsx"))
        plots_carbon(co2_emissions_df, totals, carb_dir, combine=True)

    # ------------********------------ lightweight text summary

        with open(os.path.join(carb_dir, "carbon_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Total successful wall time (s): {totals['Total_Wall_Time_s']:.6f}\n")
            f.write(f"Assumed device power (W): {totals['Watts']}\n")
            f.write(f"PUE: {totals['PUE']}\n")
            f.write(f"Total energy (kWh): {totals['Energy_kWh']:.9f}\n")
        print(f"Carbon → Excel + PNGs in {carb_dir}")

    print("\nAll done. Generated folders:")
    print("  Performance, Scalability, Reliability, CarbonFootprints\\" + args.outdir)

if __name__ == "__main__":

    # The Windows guard is required for multiprocessing safety.
    main()
