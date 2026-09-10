#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------
# gnfs_scaling_yafu.py
# Strong-scaling benchmark for YAFU GNFS

        # Notes:
# Sweeps thread counts and measures wall time per input
# Forces GNFS via -xover to keep method consistent
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
#These imports gather standard libraries for process execution, timing, statistics, filesystem operations, and platform inspection.
#They support logging, CSV handling, date/time stamping, and environment introspection for benchmarking workflows.
#Matplotlib is included to generate and save visualizations of collected performance or system metrics.

import subprocess
import time
import csv
import statistics
import platform
import shutil
import sys
import os
import re
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


# ------------********------------    CONFIG (Can be overridden via CLI flags below as per need)

DEFAULT_THREADS_LIST = [1, 2, 4]             
REPEATS = 2                                  # per N per thread
TIMEOUT_SEC = 180                            # per-run timeout
XOVER_DIGITS = "10"                          # force GNFS on small inputs
PLAN = "noecm"                               # reduce pretest variability: 'noecm'|'normal'|...
OUTDIR = "gnfs_scaling"
LABEL = "GNFS Scaling (YAFU)"

# Small-to-medium semiprimes (quick to factor; can be adjusted up later)
PRIME_PAIRS = [
    (1000003, 1000033),   # ~1.0000e12
    (1000003, 1000037),   # ~1.0000e12
    (999983,  1000003),   # ~9.9999e11
    (1065023, 1065079),   # ~1.1343e12
    (1299709, 1300033),   # ~1.6897e12
]
NUMBERS = [p*q for (p, q) in PRIME_PAIRS]


# ------------********------------ YAFU PATH DISCOVERY

def find_yafu_exe():
    here = Path(__file__).resolve().parent
    desktop = Path.home() / "Desktop"
    candidates = [
        shutil.which("yafu-x64.exe"),
        shutil.which("yafu.exe"),
        shutil.which("yafu"),
        str(here / "yafu-x64.exe"),
        str(here / "yafu.exe"),
        str(here / "yafu"),
        str(desktop / "yafu-x64.exe"),
        str(desktop / "yafu.exe"),
        str(desktop / "yafu"),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.is_file() and p.suffix.lower() == ".exe":     # Prefer valid .exe executable
            return str(p)
        # if found bare "yafu", prefer ...exe next to it

        if p.is_file() and (p.with_suffix(".exe")).is_file():  # If plain 'yafu' exists, choose nearby .exe version
            return str(p.with_suffix(".exe"))
    return None                                              # No usable executable found

def get_yafu_version(yafu_path):
    try:
        out = subprocess.run(
            [yafu_path, "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=15)   # Query YAFU for its version string
        m = re.search(r"yafu.*?version\s*([\d\.]+)", out.stdout or "", re.IGNORECASE)
        return m.group(1) if m else (out.stdout.strip().splitlines()[0] if out.stdout else "unknown")
    except Exception:
        return "unknown"                                     # Return fallback if command fails



# ------------********------------  PARSERS
#def parse_yafu_method inspects YAFU output text to infer the factoring 
#  algorithm used (GNFS or SIQS) from log keywords.

def parse_yafu_method(output):
    for line in (output or "").splitlines():
        low = line.lower()
        if "gnfs" in low or "nfs" in low: return "GNFS"
        if "siqs" in low or "qs"  in low: return "SIQS"
    return "unknown"

#def parse_yafu_total_time extracts the reported total factoring runtime in seconds using a regex match.

def parse_yafu_total_time(output):
    m = re.search(r"Total factoring time\s*=\s*([0-9.]+)\s*seconds", output or "")
    return float(m.group(1)) if m else None

#def parse_factors parses discovered integer factors from YAFU output lines, 
#  handling multiple output formats and returning unique sorted factors.

def parse_factors(output):
    facs = []
    for line in (output or "").splitlines():
        m = re.search(r"P\d+\s*=\s*([0-9]+)", line)
        if m:
            try: facs.append(int(m.group(1)))
            except: pass
        m2 = re.search(r"ans:\s*\d+\s*=\s*(\d+)\s*\*\s*(\d+)", line)
        if m2:
            try:
                facs.extend([int(m2.group(1)), int(m2.group(2))])
            except: pass
    return sorted(set(facs))


# ------------********------------ RUNNER
#def run_yafu_factor invokes the YAFU command-line tool to factor an integer n using controlled GNFS-related settings.
#It measures wall-clock runtime, captures stdout for parsing, and extracts method, total factoring time, and discovered factors.
#The function handles timeouts gracefully and returns a structured result dictionary with status, timings, and raw output.

def run_yafu_factor(n, threads, yafu_path):
    # Force GNFS selection and keep pretesting light/consistent
    cmd = [
        yafu_path, f"factor({n})",
        "-v", "-xover", XOVER_DIGITS,
        "-plan", PLAN,
        "-threads", str(threads),
        "-lathreads", str(threads),
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT_SEC)
        wall = time.perf_counter() - start
        out = proc.stdout or ""
        return {
            "ok": proc.returncode == 0,
            "wall_time_s": wall,
            "yafu_total_time_s": parse_yafu_total_time(out),
            "method": parse_yafu_method(out),
            "factors": parse_factors(out),
            "raw": out
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "wall_time_s": None, "yafu_total_time_s": None, "method": "timeout", "factors": [], "raw": "[timeout]"}


# ------------********------------ MAIN

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    yafu_path = find_yafu_exe()
    if not yafu_path:
        print("[ERROR] Could not find yafu-x64.exe / yafu.exe. Put it next to this .py file or on my Desktop.")
        print('Test: ".\\yafu-x64.exe" "factor(8051)"')
        sys.exit(1)

    host = platform.node()
    cpu = platform.processor() or platform.machine()
    os_name = f"{platform.system()} {platform.release()}"
    yafu_ver = get_yafu_version(yafu_path)

    print(f"Using YAFU: {yafu_path} (v={yafu_ver})")
    print(f"Host: {host} | CPU: {cpu} | OS: {os_name}")
    print(f"Numbers: {len(NUMBERS)} | Repeats: {REPEATS}")
    print(f"Threads sweep: {DEFAULT_THREADS_LIST}")
    print(f"Forcing GNFS at >= {XOVER_DIGITS} digits; plan={PLAN}\n")

    rows = []
    # Strong-scaling: same workload for all thread counts

    for t in DEFAULT_THREADS_LIST:
        print(f"=== Threads = {t} ===")
        for n in NUMBERS:
            digits = int(math.floor(math.log10(n))) + 1
            bits = n.bit_length()
            for r in range(1, REPEATS + 1):
                print(f"  N={digits} digits ({bits} bits) — run {r}/{REPEATS}")
                res = run_yafu_factor(n, t, yafu_path)
                status = "ok" if (res["ok"] and res["wall_time_s"] is not None) else "fail_or_timeout"
                if status == "ok":
                    print(f"    wall={res['wall_time_s']:.3f}s method={res['method']}")
                else:
                    print(f"    (failed/timeout)")
                rows.append({
                    "Timestamp": datetime.now(timezone.utc).isoformat(),
                    "Label": LABEL,
                    "Host": host,
                    "OS": os_name,
                    "CPU": cpu,
                    "Backend": "YAFU",
                    "Backend_Version": yafu_ver,
                    "Threads": t,
                    "Number": str(n),
                    "Digits": digits,
                    "Bits": bits,
                    "Run_Index": r,
                    "Wall_Time_s": round(res["wall_time_s"], 6) if res["wall_time_s"] is not None else None,
                    "YAFU_Reported_Time_s": round(res["yafu_total_time_s"], 6) if res["yafu_total_time_s"] is not None else None,
                    "Method_Reported": res["method"],
                    "Factors_Found": "x".join(str(x) for x in (res["factors"] or [])),
                    "Status": status,
                    "Notes": f"-xover {XOVER_DIGITS}, plan={PLAN}"
                })

    # Write per-run CSV

    per_run_csv = os.path.join(OUTDIR, "gnfs_scaling_results.csv")
    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-run → {per_run_csv}")

    # Aggregate per thread

    # Use sum of wall times across all Ns and repeats (successful runs only)

    agg = {}
    for r in rows:
        if r["Status"] != "ok" or r["Wall_Time_s"] is None: continue
        agg.setdefault(r["Threads"], []).append(r["Wall_Time_s"])

    threads_sorted = sorted(agg.keys())
    base_time = sum(agg[threads_sorted[0]]) if threads_sorted else None

    summary_rows = []
    for t in threads_sorted:
        total_t = sum(agg[t])
        speedup = (base_time / total_t) if (base_time and total_t > 0) else None
        efficiency = (speedup / t) if (speedup and t > 0) else None
        summary_rows.append({
            "Label": LABEL,
            "Threads": t,
            "Total_Wall_Time_s": round(total_t, 6),
            "Speedup_vs_1T": round(speedup, 6) if speedup else None,
            "Efficiency": round(efficiency, 6) if efficiency else None,
            "Num_Successful_Runs": len(agg[t])
        })

    summary_csv = os.path.join(OUTDIR, "gnfs_scaling_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary → {summary_csv}")

    # ------------********------------ Plots ---------

    # 1) Runtime vs Threads (lower is better)

    totals = [s["Total_Wall_Time_s"] for s in summary_rows]   # Extract total runtimes for each thread count
    plt.figure()
    plt.plot(threads_sorted, totals, marker="o")              # Plot runtime vs number of threads
    plt.xlabel("Threads")
    plt.ylabel("Total wall time (s) — sum over all Ns & repeats")
    plt.title("GNFS (YAFU) — Strong Scaling Runtime")
    plt.grid(True)
    plt.tight_layout()                                        # Improve layout before saving
    fig1 = os.path.join(OUTDIR, "scaling_runtime_vs_threads.png")
    plt.savefig(fig1, dpi=200)                                # Save scaling plot


    # 2) Speedup vs Threads

    su = [s["Speedup_vs_1T"] for s in summary_rows]     # Extract speedup relative to 1-thread baseline
    plt.figure()
    plt.plot(threads_sorted, su, marker="o")            # Plot actual speedup curve
    plt.plot(threads_sorted, threads_sorted, linestyle="--", label="Ideal")  # Ideal linear scaling reference
    plt.xlabel("Threads")
    plt.ylabel("Speedup (× vs 1 thread)")
    plt.title("GNFS (YAFU) — Strong Scaling Speedup")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig2 = os.path.join(OUTDIR, "scaling_speedup.png")
    plt.savefig(fig2, dpi=200)                          # Save speedup plot


    # 3) Efficiency vs Threads

    eff = [s["Efficiency"] for s in summary_rows]    # Extract parallel efficiency values
    plt.figure()
    plt.plot(threads_sorted, eff, marker="o")        # Plot efficiency vs thread count
    plt.xlabel("Threads")
    plt.ylabel("Parallel efficiency")
    plt.title("GNFS (YAFU) — Strong Scaling Efficiency")
    plt.grid(True)
    plt.tight_layout()
    fig3 = os.path.join(OUTDIR, "scaling_efficiency.png")
    plt.savefig(fig3, dpi=200)                       # Save efficiency plot

    print(f"Saved plots → {fig1}, {fig2}, {fig3}")   # Report file paths of generated plots
    print("Done.")


if __name__ == "__main__":
    main()
