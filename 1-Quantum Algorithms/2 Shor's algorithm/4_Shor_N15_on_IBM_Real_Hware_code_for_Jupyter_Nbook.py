#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Shor_N15_on_IBM_Real_Hware_code_for_Jupyter_Nbook.py
# === Shor on real hardware (ibm_fez) for DEFAULT_NS; four workbooks + PNGs ===
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



# ------------********------------   Imports

import os, io, time, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from fractions import Fraction

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate   # QFTGate (no inverse= kwarg)
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


# ------------********------------  Configuration

DEFAULT_NS = [15, 21, 33, 35, 39]
SHOTS = 2048
A_FOR_15 = 2
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# Note: Carbon footprint (These are the flags specified by me on my devices, you can change to yours or defualts)
ASSUMED_DEVICE_POWER_W = 65.0
PUE_FACTOR = 1.2
GRID_INTENSITY_KG_PER_KWH = 0.45

# Output folders/files, you can change as per your project/ requirements
# Will output four folders

OUT_PERF_DIR = "Performance"
OUT_SCAL_DIR = "Scalability"
OUT_RELI_DIR = "Reliability"
OUT_CARB_DIR = os.path.join("CarbonFootprints", TS)
os.makedirs(OUT_PERF_DIR, exist_ok=True)
os.makedirs(OUT_SCAL_DIR, exist_ok=True)
os.makedirs(OUT_RELI_DIR, exist_ok=True)
os.makedirs(OUT_CARB_DIR, exist_ok=True)

PERF_XLS = os.path.join(OUT_PERF_DIR, "performance.xlsx")
SCAL_XLS = os.path.join(OUT_SCAL_DIR, "scalability.xlsx")
RELI_XLS = os.path.join(OUT_RELI_DIR, "reliability.xlsx")
CARB_XLS = os.path.join(OUT_CARB_DIR, "carbon_footprints.xlsx")


# ------------********------------  Backend use 
# It depends on your region, availability, jobs pending, credits in account
# I chose IBM FEZ For this Shor Job, The rest I did with IBM Torino
# You can chose your selected 'backend' if it exists

try:
    backend
except NameError:
    service = QiskitRuntimeService()  # or QiskitRuntimeService(instance="open-instance")
    backend = service.backend("ibm_fez") # I chose IBM FEZ For this Shor Job, The rest I did with IBM Torino
print(f"Using backend: {backend.name} | qubits: {backend.num_qubits} | queue: {backend.status().pending_jobs}")

sampler = Sampler(mode=backend)  # job-mode on real IBM hardware


# ------------********------------ Utility helpers

def twoq_count(qc: QuantumCircuit):
    c = 0
    for inst in qc.data:
        op = inst.operation
        try:
            if op.num_qubits == 2:    # Count only two-qubit operations (CNOT, CZ, etc.)
                c += 1
        except Exception:
            pass                      # Ignore operations that don't expose num_qubits
    return c                           # Total number of 2-qubit gates

def bs_phase_to_order(bitstr, num_control, N):
    dec = int(bitstr, 2)                           # Convert measured bitstring → integer
    theta = dec / (2**num_control)                 # Convert to phase estimate in [0,1)
    r = Fraction(theta).limit_denominator(N).denominator  
        # Recover the order r by rational approximation of the phase
    return theta, r


def try_factors(a, N, r):
    if r % 2 != 0:
        return None                       # Order must be even for Shor's factor extraction

    x = pow(a, r // 2, N)                  # Compute a^(r/2) mod N
    if x in (1, N - 1):
        return None                       # Trivial cases give no useful factors

    f1 = math.gcd(x - 1, N)               # Candidate factor from (x − 1)
    f2 = math.gcd(x + 1, N)               # Candidate factor from (x + 1)

    cand = sorted({f for f in (f1, f2) if f not in (1, N)})  # Discard trivial factors
    return cand if cand else None


def backend_calibration_snapshot(backend):
    snap = {"avg_readout_error": None, "avg_t1_us": None, "avg_t2_us": None}
    try:
        props = backend.properties()                      # Retrieve calibration data from backend
        if props:
            ro = []
            t1s, t2s = [], []
            for qdat in props.qubits:                     # Loop over qubits to extract T1/T2
                try:
                    for item in qdat:
                        if item.name == "T1":
                            t1s.append(item.value * (1e6 if item.unit == "s" else 1.0))  # Convert T1 to microseconds
                        if item.name == "T2":
                            t2s.append(item.value * (1e6 if item.unit == "s" else 1.0))  # Convert T2 to microseconds
                except Exception:
                    pass

            for gate in props.gates:                      # Extract measurement/readout error
                if gate.gate == "measure":
                    for param in gate.parameters:
                        if param.name in ("gate_error", "readout_error"):
                            ro.append(param.value)

            snap["avg_readout_error"] = float(np.mean(ro)) if ro else None
            snap["avg_t1_us"] = float(np.mean(t1s)) if t1s else None
            snap["avg_t2_us"] = float(np.mean(t2s)) if t2s else None
    except Exception:
        pass                                               # Fail silently if backend lacks properties
    return snap                                             # Return calibration summary


# ------------********------------ Hand-compiled blocks for N=15 (hardware-friendly)

def M2mod15():
    U = QuantumCircuit(4)
    U.swap(2, 3); U.swap(1, 2); U.swap(0, 1)     # Implements multiplication by 2 mod 15 as a permutation on 4 qubits
    return U.to_gate(label="M_2")                # Convert circuit to a reusable gate

def M4mod15():
    U = QuantumCircuit(4)
    U.swap(1, 3); U.swap(0, 2)                   # Implements multiplication by 4 mod 15 as a permutation on 4 qubits
    return U.to_gate(label="M_4")

def a2kmodN(a, k, N):
    for _ in range(k):
        a = (a * a) % N                          # Repeated squaring: compute a^(2^k) mod N
    return a

def shor_order_circuit_15(a=2):
    """Order-finding PEA for N=15 using compact permutations; 8 control + 4 target."""
    N = 15
    num_target = 4
    num_control = 8

    control = QuantumRegister(num_control, "C")
    target  = QuantumRegister(num_target,  "T")
    creg    = ClassicalRegister(num_control, "out")
    qc = QuantumCircuit(control, target, creg)

    # |1> on target register
    qc.x(target[0])

    # H on control + controlled modular multiplications
    for k in range(num_control):
        qc.h(control[k])
        b = a2kmodN(a, k, N)
        if b == 2:
            qc.append(M2mod15().control(), [control[k], *target])
        elif b == 4:
            qc.append(M4mod15().control(), [control[k], *target])
        else:
            pass  # identity

    # Inverse QFT over control + measure  (use .inverse(), not inverse=)
    qc.append(QFTGate(num_control).inverse(), control)
    qc.measure(control, creg)
    return qc, num_control

# A small probe circuit so every N submits a real job (when full Shor is impractical)
def probe_circuit(n_qubits=2):
    q = QuantumCircuit(n_qubits, n_qubits)
    if n_qubits >= 2:
        q.h(0); q.cx(0, 1)
    q.measure_all()
    return q


# ------------********------------  Run across DEFAULT_NS on hardware

calib = backend_calibration_snapshot(backend)

perf_rows, scal_rows, reli_rows, carb_rows = [], [], [], []

for N in DEFAULT_NS:
    attempted = False
    job_id = ""
    elapsed_s = None
    counts = {}
    status = ""
    depth = None
    twoq = None
    width = None
    a_used = None
    num_control = None

    t0 = time.time()
    if N == 15:
        # Real Shor order-finding run (N=15, a=2)

        qc, num_control = shor_order_circuit_15(a=A_FOR_15)   # Build the known Shor circuit for N=15
        width = qc.num_qubits
        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
        tqc = pm.run(qc)                                      # Compile circuit for target hardware
        depth = tqc.depth()
        twoq = twoq_count(tqc)                                # Count two-qubit gates (error-critical)
        attempted = True
        a_used = A_FOR_15

        job = sampler.run([tqc], shots=SHOTS)                 # Submit compiled circuit to hardware/simulator
        job_id = job.job_id()
        res = job.result()                                    # Fetch results (blocking)
        counts = res[0].join_data().get_counts()              # Measurement outcomes
        status = "ok"
    else:
        # Generic Shor is impractically deep for these N; submit a probe so every N has data.

        qn = min(backend.num_qubits, 2) or 1                   # Use small dummy circuit respecting backend limits
        qc = probe_circuit(qn)                                 # Lightweight circuit to get hardware metrics
        width = qc.num_qubits
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        tqc = pm.run(qc)                                       # Compile probe circuit
        depth = tqc.depth()
        twoq = twoq_count(tqc)
        job = sampler.run([tqc], shots=SHOTS)                  # Submit probe job
        job_id = job.job_id()
        res = job.result()
        counts = res[0].join_data().get_counts()
        status = "probe_only (Shor not implemented on hardware for this N)"


    t1 = time.time()
    elapsed_s = round(t1 - t0, 3)

    # Reliability post-processing (try to extract factors for N=15 only)
    factors = ""
    max_frac = None
    if counts:
        total = sum(counts.values())
        if total > 0:
            max_frac = max(counts.values()) / total

    if N == 15 and attempted and counts:
        best = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        fac = None
        for bitstr, _c in best:
            theta, r = bs_phase_to_order(bitstr, num_control, N)
            fac = try_factors(A_FOR_15, N, r)
            if fac:
                break
        factors = str(fac) if fac else ""

    # ------------********------------ Collect metric rows
    perf_rows.append({
        "N": N,
        "attempted_shor": attempted,          # Whether a real Shor run was attempted (only N=15)
        "backend": backend.name,
        "shots": SHOTS,
        "job_id": job_id,                     # Hardware job identifier
        "elapsed_wall_s": elapsed_s,          # Total runtime for this N
        "compiled_depth": depth,              # Circuit depth after transpilation
        "compiled_twoq": twoq,                # Number of two-qubit gates (dominant error source)
        "qubits_used": width,                 # Actual circuit width
        "status": status
    })

    num_target = int(math.ceil(math.log2(N)))  # Minimum qubits needed to represent N
    scal_rows.append({
        "N": N,
        "target_qubits(ceil(log2 N))": num_target,
        "control_qubits(heuristic)": 2 * num_target,       # Approximate number of control qubits for Shor
        "total_qubits_heuristic": 3 * num_target,          # Rough total qubit requirement
        "compiled_depth": depth,
        "compiled_twoq": twoq,
        "attempted_shor": attempted,
        "backend_qubits": backend.num_qubits               # Hardware capacity information
    })

    # ---- Carbon (includes PUE)
    energy_kwh = (elapsed_s / 3600.0) * ((ASSUMED_DEVICE_POWER_W * PUE_FACTOR) / 1000.0)
        # Convert runtime + power draw → estimated energy usage in kWh (accounts for PUE)

    co2_kg = energy_kwh * GRID_INTENSITY_KG_PER_KWH
        # Convert energy consumption into CO₂ emissions using grid intensity

    carb_rows.append({
        "N": N,
        "job_id": job_id,
        "elapsed_wall_s": elapsed_s,
        "assumed_power_W": ASSUMED_DEVICE_POWER_W,
        "pue": PUE_FACTOR,
        "facility_power_W": round(ASSUMED_DEVICE_POWER_W * PUE_FACTOR, 3),  # Effective power incl. PUE
        "energy_kWh_est": round(energy_kwh, 6),
        "grid_intensity_kg_per_kWh": GRID_INTENSITY_KG_PER_KWH,
        "CO2e_kg_est": round(co2_kg, 6),                                    # Estimated CO₂ emissions
        "status": status
    })

    # ---- Save a counts histogram PNG per N (Reliability)
    if counts:
        bs, cs = zip(*sorted(counts.items(), key=lambda kv: kv[0]))  # Sort bitstrings for consistent plotting
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(bs)), cs)                                 # Histogram of measurement counts
        plt.xticks(range(len(bs)), bs, rotation=90)
        plt.title(f"Counts (N={N}) on {backend.name}")
        plt.tight_layout()
        png_path = os.path.join(OUT_RELI_DIR, f"counts_N{N}_{TS}.png")
        plt.savefig(png_path, dpi=150)                               # Save reliability histogram
        plt.close()

# -------------------------
# ------------********------------ Write the Excel files (1 per metric)
# -------------------------
def write_xlsx(path, df_dict):
    with pd.ExcelWriter(path, engine="xlsxwriter") as wr:
        for sheet, df in df_dict.items():
            df.to_excel(wr, sheet_name=sheet, index=False)    # Write each DataFrame to its own sheet

perf_df = pd.DataFrame(perf_rows)
scal_df = pd.DataFrame(scal_rows)
reli_df = pd.DataFrame(reli_rows)
carb_df = pd.DataFrame(carb_rows)

write_xlsx(PERF_XLS, {"Performance": perf_df})               # Save performance results
write_xlsx(SCAL_XLS, {"Scalability": scal_df})               # Save scalability results
write_xlsx(RELI_XLS, {"Reliability": reli_df})               # Save reliability results
write_xlsx(CARB_XLS, {"CarbonFootprints": carb_df})          # Save carbon impact results


# -------------------------
# ------------********------------ Metric summary PNGs (robust to missing columns)
# -------------------------


# ------------********------------ Performance: elapsed vs N

if not perf_df.empty and {"N","elapsed_wall_s"}.issubset(perf_df.columns):
    plt.figure(figsize=(6, 3))
    plt.plot(perf_df["N"], perf_df["elapsed_wall_s"], marker="o")
    plt.xlabel("N"); plt.ylabel("Elapsed (s)"); plt.title("Performance: Elapsed vs N")
    plt.tight_layout()
    p_png = os.path.join(OUT_PERF_DIR, f"performance_elapsed_{TS}.png")
    plt.savefig(p_png, dpi=150); plt.close()
else:
    p_png = os.path.join(OUT_PERF_DIR, f"performance_elapsed_{TS}.png")
    plt.figure(figsize=(6,3)); plt.text(0.5,0.5,"No Performance data",ha="center",va="center"); plt.axis("off")
    plt.savefig(p_png, dpi=150); plt.close()

# ------------********------------ Scalability: compiled depth vs N

if not scal_df.empty and {"N","compiled_depth"}.issubset(scal_df.columns):
    plt.figure(figsize=(6, 3))
    plt.plot(scal_df["N"], scal_df["compiled_depth"], marker="o")
    plt.xlabel("N"); plt.ylabel("Compiled depth"); plt.title("Scalability: Depth vs N")
    plt.tight_layout()
    s_png = os.path.join(OUT_SCAL_DIR, f"scalability_depth_{TS}.png")
    plt.savefig(s_png, dpi=150); plt.close()
else:
    s_png = os.path.join(OUT_SCAL_DIR, f"scalability_depth_{TS}.png")
    plt.figure(figsize=(6,3)); plt.text(0.5,0.5,"No Scalability data",ha="center",va="center"); plt.axis("off")
    plt.savefig(s_png, dpi=150); plt.close()

# ------------********------------ Reliability: max outcome fraction vs N

if not reli_df.empty and {"N","max_outcome_fraction"}.issubset(reli_df.columns):
    plt.figure(figsize=(6, 3))
    plt.plot(reli_df["N"], reli_df["max_outcome_fraction"], marker="o")
    plt.xlabel("N"); plt.ylabel("Max outcome fraction"); plt.title("Reliability proxy")
    plt.tight_layout()
    r_png = os.path.join(OUT_RELI_DIR, f"reliability_proxy_{TS}.png")
    plt.savefig(r_png, dpi=150); plt.close()
else:
    r_png = os.path.join(OUT_RELI_DIR, f"reliability_proxy_{TS}.png")
    plt.figure(figsize=(6,3)); plt.text(0.5,0.5,"No Reliability data",ha="center",va="center"); plt.axis("off")
    plt.savefig(r_png, dpi=150); plt.close()

# ------------********------------ Carbon: CO2e vs N

if not carb_df.empty and {"N","CO2e_kg_est"}.issubset(carb_df.columns):
    plt.figure(figsize=(6, 3))
    plt.bar(carb_df["N"].astype(str), carb_df["CO2e_kg_est"])      # Plot CO₂ emissions per N
    plt.xlabel("N"); plt.ylabel("CO2e (kg, est)"); plt.title("Carbon Footprints (incl. PUE)")
    plt.tight_layout()
    c_png = os.path.join(OUT_CARB_DIR, f"carbon_footprints_{TS}.png")
    plt.savefig(c_png, dpi=150); plt.close()                       # Save carbon footprint plot
else:
    c_png = os.path.join(OUT_CARB_DIR, f"carbon_footprints_{TS}.png")
    plt.figure(figsize=(6,3))
    plt.text(0.5,0.5,"No Carbon data", ha="center", va="center")   # Placeholder when no data is available
    plt.axis("off")
    plt.savefig(c_png, dpi=150); plt.close()


print("=== Done ===")
print("Excel:")
print(" -", PERF_XLS)
print(" -", SCAL_XLS)
print(" -", RELI_XLS)
print(" -", CARB_XLS)
print("PNGs:")
for p in [os.path.join(OUT_PERF_DIR, f"performance_elapsed_{TS}.png"),
          os.path.join(OUT_SCAL_DIR, f"scalability_depth_{TS}.png"),
          os.path.join(OUT_RELI_DIR, f"reliability_proxy_{TS}.png"),
          os.path.join(OUT_CARB_DIR, f"carbon_footprints_{TS}.png")]:
    print(" -", p)
