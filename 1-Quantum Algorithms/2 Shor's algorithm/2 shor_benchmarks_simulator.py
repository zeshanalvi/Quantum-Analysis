# shor_benchmarks_simulator.py
# Small, approachable Shor evaluation
# fallback that simulates an ideal quantum order-finding oracle.
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

# Outputs (one .xlsx per metric + PNGs):
#   Performance\performance.xlsx
#   Scalability\scalability.xlsx
#   Reliability\reliability.xlsx
#   CarbonFootprints\<outdir>\carbon_footprints.xlsx (+ PNGs)
#
# Default Ns: 15, 21, 33, 35, 39
# Works with Qiskit 0.41.x on Python 3.10 
# ----------------------------------------------------------------------------------------------------------------


# ------------********------------   Imports

import argparse, math, os, time, random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout  # For parallel execution with timeout control
import pandas as pd
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt

DEFAULT_NS = [15, 21, 33, 35, 39]   # Default composite numbers (small semiprimes) for factoring tests


# ------------********------------   Qiskit plumbing 

def _import_qiskit_qasm():
    
    # Prefer BasicAer qasm_simulator; fallback to Aer qasm_simulator.
    # Returns (backend, ShorCtor, note) or raises ImportError.
    
    try:
        from qiskit import BasicAer
        backend = BasicAer.get_backend("qasm_simulator")  
        from qiskit.algorithms import Shor                 # Shor implementation
        return backend, Shor, "BasicAer qasm_simulator"
    except Exception:
        pass

    try:
        from qiskit import Aer
        backend = Aer.get_backend("qasm_simulator")        # Fallback: Aer QASM simulator
        from qiskit.algorithms import Shor
        return backend, Shor, "Aer qasm_simulator"
    except Exception:
        pass

    raise ImportError(
        "QASM simulator with Shor not found (use Qiskit 0.41.x on Python 3.10)."
    )   


# ------------********------------ helpers

def human_ts(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")  
    # Produce UTC timestamp in ISO format

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True); return p  
    # Create directory if missing; return path

def desktop_path(): 
    return os.path.join(os.path.expanduser("~"), "Desktop")  
    # Resolve absolute path to Desktop

def digits_bits(n): 
    return (int(math.log10(n)) + 1, n.bit_length())  
    # Return (number of decimal digits, number of bits)

def save_plot(fig, path): 
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)  
    # Neatly format and save a figure, then release resources


# ------------********------------ Idealized fallback

def _order_mod(a, N):
    # Return the multiplicative order r of a (mod N), brute-forced.
    # This simulates a perfect period-finding subroutine.

    if math.gcd(a, N) != 1: return 0          # Order only defined when a and N are coprime
    x, r = 1, 1
    while True:
        x = (x * a) % N                       # Compute a^r mod N iteratively
        if x == 1:
            return r                          # Found the order
        r += 1
        if r > 10_000:                        # Safety guard for pathological loops
            return 0

 # Deterministic classical emulation of Shor’s post-processing 
 #  using an ideal order oracle.

def shor_ideal_fallback(N, max_tries=25, rng=None):
   
    rng = rng or random.Random()
    for _ in range(max_tries):
        a = rng.randrange(2, N-1)
        g = math.gcd(a, N)
        if g > 1:                              # Lucky case: a shares factor with N
            return True, sorted([g, N // g]), f"gcd-shortcut (a={a})"

        r = _order_mod(a, N)                   # Perfect “oracle” for the order r
        if r % 2 == 0 and pow(a, r//2, N) != N-1:
            p = math.gcd(pow(a, r//2, N) - 1, N)
            q = math.gcd(pow(a, r//2, N) + 1, N)
            if 1 < p < N and 1 < q < N and p*q == N:   # Validate extracted factors
                return True, sorted([p, q]), f"ideal-oracle (a={a}, r={r})"

    return False, [], "ideal-oracle failed"              # No success within max_tries

# ---------- One attempt (Qiskit then ideal fallback) ----------
def _run_one_attempt(N: int, shots: int, seed: int, allow_fallback: bool):
    t0 = time.perf_counter()
    method = "qiskit"
    try:
        backend, ShorCtor, note = _import_qiskit_qasm()
        from qiskit.utils import QuantumInstance
        qi = QuantumInstance(backend=backend, shots=shots, seed_simulator=seed, seed_transpiler=seed)
        shor = ShorCtor()
        res = shor.factor(N, a=None, quantum_instance=qi)

        # Parse robustly
        factors = []
        if res is not None:
            if isinstance(res, dict):
                f = res.get("factors", None)
                if isinstance(f, dict) and N in f: factors = f[N]
                elif isinstance(f, (list, tuple)) and f: factors = list(f[0])
            else:
                if hasattr(res, "factors") and res.factors:
                    if isinstance(res.factors, dict) and N in res.factors: factors = res.factors[N]
                    elif isinstance(res.factors, (list, tuple)) and res.factors: factors = list(res.factors[0])
        ok = bool(factors) and math.prod(factors) == N
        if ok:
            wall = time.perf_counter() - t0
            return dict(ok=True, wall_s=wall, factors=sorted(factors), reason="", method=method, backend_note=note)

        # If Qiskit path didn’t yield factors, optionally can use ideal fallback
        if allow_fallback:
            method = "ideal_fallback"
            ok2, fac2, why = shor_ideal_fallback(N, rng=random.Random(seed))
            wall = time.perf_counter() - t0
            return dict(ok=ok2, wall_s=wall, factors=fac2, reason=why, method=method, backend_note=note)

        wall = time.perf_counter() - t0
        return dict(ok=False, wall_s=wall, factors=[], reason="no non-trivial factors", method=method, backend_note=note)
    except Exception as e:
        if allow_fallback:
            method = "ideal_fallback"
            ok2, fac2, why = shor_ideal_fallback(N, rng=random.Random(seed))
            wall = time.perf_counter() - t0
            return dict(ok=ok2, wall_s=wall, factors=fac2, reason=f"qiskit_error: {e}; {why}", method=method, backend_note="fallback")
        wall = time.perf_counter() - t0
        return dict(ok=False, wall_s=wall, factors=[], reason=f"qiskit_error: {e}", method=method, backend_note="error")

# ------------********------------ Benchmarks 
# ------------********------------ Performance Benchmarks 
def run_performance(Ns, repeats, timeout_s, shots, allow_fallback):
    rows = []
    with ProcessPoolExecutor(max_workers=1) as pool:
        for n in Ns:
            di, bi = digits_bits(n)
            for r in range(1, repeats+1):
                seed = (int(time.time()*1000) + 31*r + n) % (2**32-1)
                fut = pool.submit(_run_one_attempt, n, shots, seed, allow_fallback)
                tstart = time.perf_counter()
                try:
                    res = fut.result(timeout=timeout_s)
                except FuturesTimeout:
                    res = dict(ok=False, wall_s=time.perf_counter()-tstart, factors=[], reason="timeout", method="qiskit", backend_note="timeout")
                status = "ok" if res["ok"] else ("timeout" if "timeout" in res["reason"].lower() else "fail")
                rows.append({
                    "Timestamp": human_ts(),
                    "N": n, "Digits": di, "Bits": bi, "Run_Index": r,
                    "Wall_Time_s": round(res["wall_s"], 6),
                    "Status": status, "Method": res.get("method",""),
                    "Fail_Reason": res.get("reason",""), "Backend": res.get("backend_note",""),
                    "Factors_Found": "x".join(map(str, res.get("factors", []))),
                })
            ok_cnt = sum(1 for row in rows if row["N"]==n and row["Status"]=="ok")
            print(f"[Performance] N={n} successes={ok_cnt}/{repeats}")
    return pd.DataFrame(rows)

# ------------********------------ Scalability Benchmarks 
def run_scalability(perf_df):
    ok = perf_df[perf_df["Status"]=="ok"]              # Keep only successful runs
    if ok.empty:
        g = perf_df.groupby(["N","Digits","Bits"])["Wall_Time_s"].mean().reset_index()
        g.rename(columns={"Wall_Time_s":"Avg_Time_s"}, inplace=True)
        g["Std_Time_s"]=0.0; g["Success_Count"]=0    
        return g

    g = ok.groupby(["N","Digits","Bits"])["Wall_Time_s"].agg(["mean","std","count"]).reset_index()
    g.rename(columns={
        "mean":"Avg_Time_s",
        "std":"Std_Time_s",
        "count":"Success_Count"
    }, inplace=True)                                    # Standard scalability metrics
    return g

# ------------********------------ Reliability Benchmarks 

def run_reliability(perf_df, repeats):
    sr = perf_df.groupby(["N","Digits","Bits"])["Status"].apply(lambda s:(s=="ok").mean()).reset_index(name="Success_Rate")
    base = perf_df[perf_df["Status"]=="ok"]
    if base.empty: base = perf_df
    disp = base.groupby(["N","Digits","Bits"])["Wall_Time_s"].agg(
        Avg_Time_s="mean", Std_Time_s="std", Min_Time_s="min", Max_Time_s="max", Median_Time_s="median"
    ).reset_index()
    out = pd.merge(sr, disp, on=["N","Digits","Bits"], how="left")
    out["Repeats"]=repeats
    return out

# ------------********------------ Carbon handling Benchmarks

def _find_co2_file(default="Filtered CO2 intensity 236 Countries"):
    desk = desktop_path()
    for nm in (default, default+".xlsx", default+".csv"):
        p = os.path.join(desk, nm)
        if os.path.isfile(p): return p
    for nm in (default, default+".xlsx", default+".csv"):
        p = os.path.join(os.getcwd(), nm)
        if os.path.isfile(p): return p
    return None

def _norm(s):
    import re; return re.sub(r'[^a-z0-9]', '', str(s).strip().lower())

def _read_co2_table(path):
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in (".xlsx",".xls") else pd.read_csv(path)
    norm = {_norm(c): c for c in df.columns}
    # country
    country_col = None
    for k in ["country","countryname","countryorarea","entity","name","location","area","nation","economy"]:
        if k in norm: country_col = norm[k]; break
    if not country_col:
        obj = [c for c in df.columns if df[c].dtype==object]
        if not obj: raise ValueError("Could not detect a country column in the CO₂ file.")
        country_col = obj[0]
    # intensity
    kg_col = next((orig for k,orig in norm.items() if "kwh" in k and "co2" in k and "kg" in k), None)
    g_col  = next((orig for k,orig in norm.items() if "kwh" in k and "co2" in k and "g"  in k), None)
    if not kg_col and not g_col:
        raise ValueError("Need intensity column (kgCO2/kWh or gCO2/kWh).")
    # year (optional)
    year_col = next((orig for k,orig in norm.items() if k in ("year","yr")), None)

    if kg_col:
        df["kg_co2_per_kwh"] = pd.to_numeric(df[kg_col], errors="coerce")
    else:
        df["kg_co2_per_kwh"] = pd.to_numeric(df[g_col], errors="coerce")/1000.0
    df.rename(columns={country_col:"Country"}, inplace=True)
    keep = ["Country","kg_co2_per_kwh"]
    if year_col:
        df.rename(columns={year_col:"Year"}, inplace=True); keep.append("Year")
    return df[keep].dropna(subset=["kg_co2_per_kwh"])

def _select_year(df, year_select):
    if "Year" not in df.columns: return df.reset_index(drop=True)
    if str(year_select).lower()=="latest":
        idx = df.groupby("Country")["Year"].idxmax()
        return df.loc[idx].reset_index(drop=True)
    try:
        y=int(year_select); sel=df[df["Year"]==y]
        return sel.reset_index(drop=True) if not sel.empty else df.loc[df.groupby("Country")["Year"].idxmax()].reset_index(drop=True)
    except:
        return df.loc[df.groupby("Country")["Year"].idxmax()].reset_index(drop=True)

def compute_co2(perf_df, watts, pue, co2_df, mode="successful"):
    base = perf_df if mode=="all" else perf_df[perf_df["Status"]=="ok"]
    total_time_s = float(base["Wall_Time_s"].sum())
    energy_kwh = (watts*pue*total_time_s)/(1000.0*3600.0)
    out = co2_df.copy()
    out["Emissions_kgCO2"] = out["kg_co2_per_kwh"]*energy_kwh
    out.sort_values("Emissions_kgCO2", ascending=False, inplace=True)
    return out, dict(Total_Wall_Time_s=total_time_s, Watts=watts, PUE=pue, Energy_kWh=energy_kwh, Mode=mode)

# ------------********------------ plots

def plots_performance(perf_df, out_dir):
    ok = perf_df[perf_df["Status"]=="ok"]; src = ok if not ok.empty else perf_df

    g = src.groupby("N")["Wall_Time_s"].agg(["mean","std"]).reset_index()
    fig=plt.figure(); plt.errorbar(g["N"], g["mean"], yerr=g["std"].fillna(0.0), fmt="o-")
    plt.xlabel("Semiprime N"); plt.ylabel("Average wall time (s)")
    plt.title("Runtime per N"+(" (all runs)" if ok.empty else "")); plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir,"runtime_per_N.png"))

    g2 = src.groupby("Digits")["Wall_Time_s"].mean().reset_index()
    fig=plt.figure(); plt.plot(g2["Digits"], g2["Wall_Time_s"], "o-")
    plt.xlabel("Digits of N"); plt.ylabel("Average wall time (s)")
    plt.title("Runtime vs Digits"+(" (all runs)" if ok.empty else "")); plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir,"runtime_vs_digits.png"))

    sr = perf_df.groupby("N")["Status"].apply(lambda s:(s=="ok").mean()).reset_index(name="Success_Rate")
    fig=plt.figure(); bars=plt.bar(sr["N"].astype(str), sr["Success_Rate"]); plt.ylim(0,1.05)
    for b,v in zip(bars,sr["Success_Rate"]): plt.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Success rate"); plt.xlabel("N"); plt.title("Success Rate per N")
    save_plot(fig, os.path.join(out_dir,"success_rate_per_N.png"))

    counts = perf_df.pivot_table(index="N", columns="Status", values="Run_Index", aggfunc="count").fillna(0)
    for c in ("ok","fail","timeout"):
        if c not in counts.columns: counts[c]=0
    counts = counts[["ok","fail","timeout"]]
    fig=plt.figure(); bottom=None
    for col in counts.columns:
        plt.bar(counts.index.astype(str), counts[col], bottom=0 if bottom is None else bottom, label=col)
        bottom = counts[col] if bottom is None else bottom + counts[col]
    plt.legend(title="Status"); plt.xlabel("N"); plt.ylabel("Count"); plt.title("Run Status Counts per N")
    save_plot(fig, os.path.join(out_dir,"status_counts_per_N.png"))

    # ------------********------------ Method breakdown
    meth = perf_df.pivot_table(index="N", columns="Method", values="Run_Index", aggfunc="count").fillna(0)
    fig=plt.figure(); bottom=None
    for col in meth.columns:
        plt.bar(meth.index.astype(str), meth[col], bottom=0 if bottom is None else bottom, label=col)
        bottom = meth[col] if bottom is None else bottom + meth[col]
    plt.legend(title="Method"); plt.xlabel("N"); plt.ylabel("Count"); plt.title("OK/Fail Method Mix per N")
    save_plot(fig, os.path.join(out_dir,"method_mix_per_N.png"))

def plots_scalability(scal_df, out_dir):
    fig=plt.figure(); plt.plot(scal_df["N"], scal_df["Avg_Time_s"], "o-")
    plt.xlabel("Semiprime N"); plt.ylabel("Average wall time (s)"); plt.title("Scalability — Average Runtime per N"); plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir,"scalability_avg_runtime_per_N.png"))
    fig=plt.figure(); plt.plot(scal_df["Digits"], scal_df["Avg_Time_s"], "o-")
    plt.xlabel("Digits of N"); plt.ylabel("Average wall time (s)"); plt.title("Scalability — Runtime vs Digits"); plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir,"scalability_runtime_vs_digits.png"))
    fig=plt.figure(); plt.errorbar(scal_df["Digits"], scal_df["Avg_Time_s"], yerr=scal_df["Std_Time_s"].fillna(0.0), fmt="o-")
    plt.xlabel("Digits of N"); plt.ylabel("Avg time ± std (s)"); plt.title("Scalability — Variability across Repeats"); plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir,"scalability_variability.png"))

def plots_reliability(rel_df, out_dir):
    fig=plt.figure(); bars=plt.bar(rel_df["N"].astype(str), rel_df["Success_Rate"]); plt.ylim(0,1.05)
    for b,v in zip(bars,rel_df["Success_Rate"]): plt.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xlabel("N"); plt.ylabel("Success rate"); plt.title("Reliability — Success Rate per N")
    save_plot(fig, os.path.join(out_dir,"reliability_success_rate.png"))
    if {"Min_Time_s","Median_Time_s","Max_Time_s"}.issubset(rel_df.columns):
        fig=plt.figure(); xs=rel_df["N"].astype(str)
        plt.plot(xs, rel_df["Min_Time_s"], "o-", label="Min")
        plt.plot(xs, rel_df["Median_Time_s"], "o-", label="Median")
        plt.plot(xs, rel_df["Max_Time_s"], "o-", label="Max")
        plt.legend(); plt.xlabel("N"); plt.ylabel("Wall time (s)"); plt.title("Reliability — Time Dispersion"); plt.grid(True, alpha=0.4)
        save_plot(fig, os.path.join(out_dir,"reliability_time_dispersion.png"))

def plots_carbon(co2_df, totals, out_dir):
    top = co2_df.head(20)
    fig=plt.figure(figsize=(8,6)); plt.barh(top["Country"], top["Emissions_kgCO2"]); plt.gca().invert_yaxis()
    plt.xlabel("Emissions (kg CO₂) for workload"); plt.title("CO₂ by Country — Shor (Ideal Simulator)")
    save_plot(fig, os.path.join(out_dir,"co2_by_country_top20.png"))
    fig=plt.figure(); cum=top["Emissions_kgCO2"].cumsum(); plt.plot(range(1,len(cum)+1), cum, "o-")
    plt.xlabel("Top-k countries"); plt.ylabel("Cumulative kg CO₂"); plt.title("Cumulative CO₂ — Top 20 Countries"); plt.grid(True, alpha=0.4)
    save_plot(fig, os.path.join(out_dir,"co2_cumulative_top20.png"))
    fig=plt.figure(); txt=(f"CO₂ mode: {totals['Mode']}\nTotal wall time: {totals['Total_Wall_Time_s']:.2f} s\n"
                           f"Device power: {totals['Watts']} W, PUE: {totals['PUE']}\nTotal energy: {totals['Energy_KWh']:.6f} kWh")
    plt.axis("off"); plt.text(0.02,0.8,"Carbon Model Summary",fontsize=12,weight="bold"); plt.text(0.02,0.6,txt,fontsize=10)
    save_plot(fig, os.path.join(out_dir,"co2_summary.png"))

# ------------********------------ Excel writer
def write_excel_one(df, out_path, sheet_name="Data"):
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        df.to_excel(xl, index=False, sheet_name=sheet_name)

# ------------********------------ main

#def main defines the CLI entry point for running repeated YAFU-based factoring 
#  benchmarks with configurable timeouts, sizes, and carbon settings.
#It parses command-line arguments, resolves the list of integers to factor, 
#  and sets up output directories for performance, scalability, reliability, and carbon analysis.
#The function prepares the experiment environment before launching the full benchmarking workflow.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--Ns", type=str, default="")
    ap.add_argument("--allow-ideal-fallback", action="store_true", help="Use ideal order-finding fallback if Qiskit returns no factors.")
    ap.add_argument("--year-select", type=str, default="latest")
    ap.add_argument("--device-power-watts", type=float, default=65.0)
    ap.add_argument("--pue", type=float, default=1.2)
    ap.add_argument("--outdir", type=str, default="carbon_by_country")
    ap.add_argument("--co2-mode", type=str, default="successful", choices=["successful","all"])
    args = ap.parse_args()

    Ns = [int(x) for x in args.Ns.split(",")] if args.Ns.strip() else DEFAULT_NS

    base = os.path.abspath(os.getcwd())
    perf_dir = ensure_dir(os.path.join(base, "Performance"))
    scal_dir = ensure_dir(os.path.join(base, "Scalability"))
    rel_dir  = ensure_dir(os.path.join(base, "Reliability"))
    carb_dir = ensure_dir(os.path.join(base, "CarbonFootprints", args.outdir))

    # ------------********------------ PERFORMANCE
    print("\n=== PERFORMANCE ===")
    perf_df = run_performance(Ns, args.repeats, args.timeout, args.shots, allow_fallback=args.allow_ideal_fallback)
    write_excel_one(perf_df, os.path.join(perf_dir, "performance.xlsx"))
    plots_performance(perf_df, perf_dir)
    print(f"[Saved] Performance → {perf_dir}")

    # ------------********------------ SCALABILITY
    print("\n=== SCALABILITY ===")
    scal_df = run_scalability(perf_df)
    write_excel_one(scal_df, os.path.join(scal_dir, "scalability.xlsx"))
    plots_scalability(scal_df, scal_dir)
    print(f"[Saved] Scalability → {scal_dir}")

    # ------------********------------ RELIABILITY
    print("\n=== RELIABILITY ===")
    rel_df = run_reliability(perf_df, args.repeats)
    write_excel_one(rel_df, os.path.join(rel_dir, "reliability.xlsx"))
    plots_reliability(rel_df, rel_dir)
    print(f"[Saved] Reliability → {rel_dir}")

    # ------------********------------ CARBON
    print("\n=== CARBON FOOTPRINTS ===")
    co2_path = _find_co2_file()
    if not co2_path:
        print("WARNING: CO₂ file not found on Desktop/CWD (Filtered CO2 intensity 236 Countries.[xlsx|csv]).")
    else:
        print(f"Using CO₂ file: {co2_path}")
        co2_raw = _read_co2_table(co2_path)
        co2_tbl = _select_year(co2_raw, args.year_select)
        co2_emissions_df, totals = compute_co2(perf_df, args.device_power_watts, args.pue, co2_tbl, mode=args.co2_mode)
        write_excel_one(co2_emissions_df, os.path.join(carb_dir, "carbon_footprints.xlsx"))
        plots_carbon(co2_emissions_df, totals, carb_dir)
        with open(os.path.join(carb_dir, "carbon_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"CO2 mode: {args.co2_mode}\n")
            f.write(f"Total wall time (s): {totals['Total_Wall_Time_s']:.6f}\n")
            f.write(f"Device power (W): {totals['Watts']}\n")
            f.write(f"PUE: {totals['PUE']}\n")
            f.write(f"Energy (kWh): {totals['Energy_KWh']:.9f}\n")
        print(f"[Saved] Carbon → {carb_dir}")

    print("\nAll done.\n"
          f"  {perf_dir}\n  {scal_dir}\n  {rel_dir}\n  {carb_dir}")

if __name__ == "__main__":
    main()
