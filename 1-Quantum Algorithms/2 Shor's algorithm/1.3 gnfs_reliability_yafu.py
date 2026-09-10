#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# gnfs_reliability_yafu.py
# Reliability benchmarking for YAFU GNFS on Windows.

# - Repeats factorization runs over a mix of deterministic + random semiprimes
# - Verifies correctness (p*q == N), classifies failures, enforces timeouts
# - Forces GNFS via -xover (configurable) for method consistency
# - Outputs per-run and per-thread/per-number summaries + reliability plots

# ----------------------------------------------------------------------------------------------------------------

# Notes: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Code focuses on Reliability

# ------------********------------   Imports
import argparse
import csv
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

# ------------********------------    Helpers: prime generation (small ranges => fast)
def _is_probable_prime(n: int) -> bool:
    if n < 2: return False                         # Primes are >= 2
    small_primes = [2,3,5,7,11,13,17,19,23,29]      # Quick checks using small primes
    for p in small_primes:
        if n == p: return True                     # Direct match → prime
        if n % p == 0: return n == p               # If divisible by small prime, composite unless equal


    # Miller–Rabin deterministic for 32-bit with bases [2, 7, 61] is enough;
    # for our ~7-digit selection this is overkill (but safe).
    d = n - 1
    s = 0
    while d % 2 == 0:              # Factor n−1 into 2^s · d  (d becomes odd)
        s += 1
        d //= 2
    for a in (2, 7, 61):           # Deterministic Miller–Rabin bases valid for 32-bit integers
        if a % n == 0:
            continue
        x = pow(a, d, n)           # Compute a^d mod n
        if x == 1 or x == n - 1:   # Strong probable prime check (first condition)
            continue
        skip_a = False
        for _ in range(s - 1):     # Repeated squaring to test additional conditions
            x = (x * x) % n
            if x == n - 1:
                skip_a = True
                break
        if skip_a:
            continue
        return False               # Composite if no condition is satisfied
    return True                     # Passed all bases → prime with high certainty


def _next_prime_geq(rng: random.Random, lo: int) -> int:
    n = lo | 1      # Ensure starting point is odd (primes > 2 are odd)
    while not _is_probable_prime(n):   # Increment by 2 until a probable prime is found
        n += 2
    return n        # Return the next prime ≥ lo


def gen_random_semiprimes(count:int, digits_min:int, digits_max:int, seed:int) -> list[int]:
    """Generate 'count' semiprimes p*q from primes around the requested digit range."""
    rng = random.Random(seed)
    outs = []
    for _ in range(count):
        d = rng.randint(digits_min, digits_max)
        lo = 10**(d-1)
        hi = 10**d - 1
        # choose two primes in [~sqrt(lo), ~sqrt(hi)] so product ~ d digits
        # pick a center ~ 10**(d/2)

        p_lo = int(10**(d/2 - 1)) if d >= 2 else 10
        p_lo = max(p_lo, 100000)        # stay around 6-digit primes (fast to factor)
        p_hi = min(int(10**(d/2)), 2000000)
        a = rng.randint(p_lo, p_hi)
        b = rng.randint(p_lo, p_hi)
        p = _next_prime_geq(rng, a | 1)
        q = _next_prime_geq(rng, b | 1)
        N = p * q
        # If N falls out of range too far, it's fine; runtime stays quick in this regime.
        outs.append(N)
    return outs

# ------------********------------ This section is for YAFU discovery on computer
def find_yafu_exe() -> str | None:
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
        if p.is_file() and p.suffix.lower() == ".exe":      # Prefer a valid .exe executable
            return str(p)
        if p.is_file() and (p.with_suffix(".exe")).is_file():  # If bare 'yafu' exists, prefer adjacent .exe version
            return str(p.with_suffix(".exe"))
    return None                                              # Return None if nothing usable found


def get_yafu_version(yafu_path:str) -> str:
    try:
        out = subprocess.run(
            [yafu_path, "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=15)   # Run YAFU with -v to query version info
        m = re.search(r"yafu.*?version\s*([\d\.]+)", out.stdout or "", re.IGNORECASE)  # Extract version number
        return m.group(1) if m else (out.stdout.strip().splitlines()[0] if out.stdout else "unknown")
    except Exception:
        return "unknown"   # Fallback if process fails or output unreadable


# ------------********------------ Output parsers ----------

def parse_yafu_method(output:str) -> str:
    for line in (output or "").splitlines():
        low = line.lower()
        if "gnfs" in low or "nfs" in low: return "GNFS"   # Detect GNFS (General Number Field Sieve)
        if "siqs" in low or "qs"  in low: return "SIQS"   # Detect SIQS / Quadratic Sieve
    return "unknown"

def parse_yafu_total_time(output:str):
    m = re.search(r"Total factoring time\s*=\s*([0-9.]+)\s*seconds", output or "")
    return float(m.group(1)) if m else None              # Extract total runtime if present

def parse_factors(output:str) -> list[int]:
    facs = []
    for line in (output or "").splitlines():
        m = re.search(r"P\d+\s*=\s*([0-9]+)", line)       # Match lines like: Pxx = factor
        if m:
            try: facs.append(int(m.group(1)))
            except: pass
        m2 = re.search(r"ans:\s*\d+\s*=\s*(\d+)\s*\*\s*(\d+)", line)  # Match ans: n = p * q
        if m2:
            try:
                facs.extend([int(m2.group(1)), int(m2.group(2))])
            except: pass

    facs = sorted(set(facs))                              # Deduplicate and sort factors
    return facs

def factors_ok(n:int, facs:list[int]) -> bool:
    if len(facs) < 2: return False                        # At least two factors needed
    # allow more than 2 lines; test all pairwise products
    for i in range(len(facs)):
        for j in range(i+1, len(facs)):
            if facs[i] * facs[j] == n:                    # Check whether any pair multiplies to n
                return True
    return False


# ------------********------------ Runner
def run_yafu_factor(n:int, yafu_path:str, threads:int, xover:str, plan:str, timeout:int, session_tag:str):
    cmd = [
        yafu_path, f"factor({n})",
        "-v",
        "-xover", str(xover),          # Force GNFS selection threshold (ensures consistent method)
        "-plan", plan,                 # Pretesting plan used by YAFU
        "-threads", str(threads),      # Sieving worker threads
        "-lathreads", str(threads),    # Linear algebra worker threads
        "-session", f"session_{session_tag}",    # Unique session name for logs
        "-logfile", f"log_{session_tag}.log"     # Log output file name
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout
        )                                  # Run YAFU with timeout and capture full output
        wall = time.perf_counter() - start
        out = proc.stdout or ""
        rc_ok = (proc.returncode == 0)      # Check exit success
        return {
            "return_ok": rc_ok,
            "wall": wall,                   # Measured wall-clock time
            "yafu_total": parse_yafu_total_time(out),  # Parse time reported by YAFU
            "method": parse_yafu_method(out),          # Detect factoring method used
            "factors": parse_factors(out),             # Extract discovered factors
            "raw": out
        }
    except subprocess.TimeoutExpired:
        return {
            "return_ok": False,
            "wall": None,
            "yafu_total": None,
            "method": "timeout",            # Mark timeout condition
            "factors": [],
            "raw": "[timeout]"
        }


# ------------********------------ Summaries ----------
def pct(values, q):
    if not values: return None
    s = sorted(values)
    k = (len(s)-1)*q
    f = math.floor(k); c = math.ceil(k)
    if f == c: return s[int(k)]
    return s[f] + (s[c]-s[f])*(k-f)

# ------------********------------  Main ----------
def main():
    ap = argparse.ArgumentParser(description="Reliability benchmarking for YAFU GNFS")
    ap.add_argument("--repeats", type=int, default=3, help="Runs per N")
    ap.add_argument("--threads", type=int, default=1, help="Threads for YAFU")
    ap.add_argument("--timeout", type=int, default=180, help="Per-run timeout (s)")
    ap.add_argument("--xover", default="10", help="QS/NFS crossover to force GNFS")
    ap.add_argument("--plan", default="noecm", help="Pretesting plan: noecm|normal|light|deep|custom")
    ap.add_argument("--outdir", default="gnfs_reliability", help="Output directory")
    ap.add_argument("--label", default="Reliability GNFS (YAFU)", help="Run label")
    ap.add_argument("--seed", type=int, default=1337, help="Seed for random semiprimes")
    ap.add_argument("--num-random", type=int, default=8, help="Number of random semiprimes")
    ap.add_argument("--digits-min", type=int, default=12, help="Min digits for random semiprimes")
    ap.add_argument("--digits-max", type=int, default=14, help="Max digits for random semiprimes")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    yafu_path = find_yafu_exe()
    if not yafu_path:
        print("[ERROR] Could not find yafu-x64.exe / yafu.exe on Desktop or PATH.")
        print('Quick test: ".\\yafu-x64.exe" "factor(8051)"')
        sys.exit(1)

    # Deterministic quick semiprimes + random set
    fixed_pairs = [
        (1000003, 1000033),
        (1000003, 1000037),
        (999983,  1000003),
        (1065023, 1065079),
        (1299709, 1300033),
    ]
    deterministic = [p*q for (p,q) in fixed_pairs]
    random_ns = gen_random_semiprimes(args.num_random, args.digits_min, args.digits_max, args.seed)
    all_numbers = deterministic + random_ns

    host = platform.node()
    cpu = platform.processor() or platform.machine()
    os_name = f"{platform.system()} {platform.release()}"
    yafu_ver = get_yafu_version(yafu_path)
    print(f"Using YAFU: {yafu_path} (v={yafu_ver})")
    print(f"Host: {host} | CPU: {cpu} | OS: {os_name}")
    print(f"Numbers: {len(all_numbers)} (fixed {len(deterministic)} + random {len(random_ns)})")
    print(f"Repeats per N: {args.repeats} | Threads: {args.threads} | Timeout: {args.timeout}s")
    print(f"GNFS force: -xover {args.xover} | plan={args.plan}")
    print()

    per_run_rows = []
    # Run experiments
    for idx, n in enumerate(all_numbers, 1):
        digits = int(math.floor(math.log10(n))) + 1
        bits = n.bit_length()
        for r in range(1, args.repeats + 1):
            tag = f"{idx}_{r}_{args.threads}"
            print(f"[{idx}/{len(all_numbers)}] N={digits} digits ({bits} bits), run {r}/{args.repeats}")
            res = run_yafu_factor(n, yafu_path, args.threads, args.xover, args.plan, args.timeout, tag)
            status = "ok"
            fail_reason = ""
            if res["method"] == "timeout" or res["wall"] is None:
                status = "fail"
                fail_reason = "timeout"
            elif not res["return_ok"]:
                status = "fail"
                fail_reason = "nonzero_exit"
            elif not factors_ok(n, res["factors"]):
                status = "fail"
                fail_reason = "wrong_factors"
            elif res["method"].upper() != "GNFS":
                # still count as success but flag method mismatch for reliability
                status = "ok"
                fail_reason = "method_mismatch"

            per_run_rows.append({
                "Timestamp": datetime.now(timezone.utc).isoformat(),
                "Label": args.label,
                "Host": host,
                "OS": os_name,
                "CPU": cpu,
                "Backend": "YAFU",
                "Backend_Version": yafu_ver,
                "Threads": args.threads,
                "Number": str(n),
                "Digits": digits,
                "Bits": bits,
                "Run_Index": r,
                "Wall_Time_s": round(res["wall"], 6) if res["wall"] is not None else None,
                "YAFU_Reported_Time_s": round(res["yafu_total"], 6) if res["yafu_total"] is not None else None,
                "Method_Reported": res["method"],
                "Factors_Found": "x".join(str(x) for x in res["factors"]),
                "Status": status,
                "Failure_Reason": fail_reason,  # "", "timeout", "nonzero_exit", "wrong_factors", or "method_mismatch"
                "Notes": f"-xover {args.xover}, plan={args.plan}"
            })

    # Save per-run CSV
    per_run_csv = outdir / "reliability_runs.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))  # Write CSV using dynamic column names
        writer.writeheader()                                                # Write header row
        writer.writerows(per_run_rows)                                      # Write all run-level records
    print(f"\nSaved per-run → {per_run_csv}")                                # Notify where file was saved


   # ------------********------------ Per-number summary ----------
    by_number = {}
    for row in per_run_rows:
        key = row["Number"]
        by_number.setdefault(key, {"times": [], "ok":0, "fail":0, "method_gnfs":0, "wrong_factors":0, "timeouts":0, "nonzero_exit":0, "mismatch":0})
        if row["Status"] == "ok" and row["Wall_Time_s"] is not None:
            by_number[key]["ok"] += 1
            by_number[key]["times"].append(row["Wall_Time_s"])
            if (row["Failure_Reason"] or "") == "method_mismatch":
                by_number[key]["mismatch"] += 1
            if (row["Method_Reported"] or "").upper() == "GNFS":
                by_number[key]["method_gnfs"] += 1
        else:
            by_number[key]["fail"] += 1
            fr = (row["Failure_Reason"] or "")
            if fr == "timeout": by_number[key]["timeouts"] += 1
            if fr == "nonzero_exit": by_number[key]["nonzero_exit"] += 1
            if fr == "wrong_factors": by_number[key]["wrong_factors"] += 1
            if fr == "method_mismatch": by_number[key]["mismatch"] += 1

    num_summary_rows = []
    for key, stats in by_number.items():
        times = stats["times"]
        p50 = median(times) if times else None
        p90 = pct(times, 0.90) if times else None
        p95 = pct(times, 0.95) if times else None
        p99 = pct(times, 0.99) if times else None
        n_ok = stats["ok"]; n_fail = stats["fail"]
        total = n_ok + n_fail if (n_ok + n_fail)>0 else 1
        success_rate = n_ok / total
        method_consistency = (stats["method_gnfs"] / max(n_ok,1))
        flakiness = 1.0 if (n_ok>0 and n_fail>0) else 0.0
        num_summary_rows.append({
            "Number": key,
            "Threads": args.threads,
            "Success_Rate": round(success_rate, 6),
            "Method_GNFS_Rate": round(method_consistency, 6),
            "Flaky": int(flakiness),
            "OK_Runs": n_ok,
            "Fail_Runs": n_fail,
            "Timeout_Fails": stats["timeouts"],
            "NonzeroExit_Fails": stats["nonzero_exit"],
            "WrongFactor_Fails": stats["wrong_factors"],
            "Method_Mismatch_Count": stats["mismatch"],
            "p50_s": round(p50, 6) if p50 is not None else None,
            "p90_s": round(p90, 6) if p90 is not None else None,
            "p95_s": round(p95, 6) if p95 is not None else None,
            "p99_s": round(p99, 6) if p99 is not None else None,
        })

    per_number_csv = outdir / "reliability_summary_by_number.csv"
    with per_number_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(num_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(num_summary_rows)
    print(f"Saved per-number summary → {per_number_csv}")

    # ------------********------------ Global / thread-level summary ----------
    ok_runs = [r for r in per_run_rows if r["Status"] == "ok" and r["Wall_Time_s"] is not None]
    times_all = [r["Wall_Time_s"] for r in ok_runs]
    fr_all = [r["Failure_Reason"] for r in per_run_rows if r["Status"] != "ok" or r["Failure_Reason"]]
    success_rate_global = len(ok_runs) / len(per_run_rows) if per_run_rows else 0.0
    method_ok = sum(1 for r in ok_runs if (r["Method_Reported"] or "").upper()=="GNFS")
    method_consistency_global = method_ok / len(ok_runs) if ok_runs else 0.0
    p50 = median(times_all) if times_all else None
    p95 = pct(times_all, 0.95) if times_all else None
    p99 = pct(times_all, 0.99) if times_all else None

    fail_counts = {}
    for fr in fr_all:
        key = fr or "ok"
        fail_counts[key] = fail_counts.get(key, 0) + 1

    summary_row = {
        "Label": args.label,
        "Threads": args.threads,
        "Total_Runs": len(per_run_rows),
        "Success_Rate": round(success_rate_global, 6),
        "Method_GNFS_Rate": round(method_consistency_global, 6),
        "Median_s": round(p50, 6) if p50 is not None else None,
        "P95_s": round(p95, 6) if p95 is not None else None,
        "P99_s": round(p99, 6) if p99 is not None else None,
        "Timeout_Fails": fail_counts.get("timeout", 0),
        "NonzeroExit_Fails": fail_counts.get("nonzero_exit", 0),
        "WrongFactor_Fails": fail_counts.get("wrong_factors", 0),
        "Method_Mismatch": fail_counts.get("method_mismatch", 0),
    }
    thread_summary_csv = outdir / "reliability_summary_overall.csv"
    with thread_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    print(f"Saved overall summary → {thread_summary_csv}")

    # ------------********------------ Plots

    # 1) Histogram of runtimes (ok runs)
    if times_all:
        plt.figure()
        plt.hist(times_all, bins=min(30, max(5, int(len(times_all)/2))))   # Adaptive bin count based on sample size
        plt.xlabel("Wall time (s)")
        plt.ylabel("Count")
        plt.title("Runtime distribution (OK runs)")
        plt.tight_layout()
        plt.savefig(outdir / "reliability_runtime_hist.png", dpi=200)       # Save histogram

    # 2) CDF of runtimes
    if times_all:
        s = sorted(times_all)
        y = [i/len(s) for i in range(1, len(s)+1)]                         # Empirical CDF values
        plt.figure()
        plt.plot(s, y)
        plt.xlabel("Wall time (s)")
        plt.ylabel("Empirical CDF")
        plt.title("Runtime CDF (OK runs)")
        plt.tight_layout()
        plt.savefig(outdir / "reliability_runtime_cdf.png", dpi=200)       # Save CDF plot

    # 3) Failure reasons bar
    if fail_counts:
        labels = list(fail_counts.keys())
        vals = [fail_counts[k] for k in labels]
        plt.figure()
        plt.bar(labels, vals)                                              # Bar chart of failure categories
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.title("Failure / anomaly counts")
        plt.tight_layout()
        plt.savefig(outdir / "reliability_fail_categories.png", dpi=200)   # Save failure-count plot

    print("Saved plots →",
          outdir / "reliability_runtime_hist.png", ",",
          outdir / "reliability_runtime_cdf.png", ",",
          outdir / "reliability_fail_categories.png")
    print("Done.")

if __name__ == "__main__":
    main()    # Execute main routine when invoked as script

