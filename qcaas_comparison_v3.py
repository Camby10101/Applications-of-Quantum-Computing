"""
Quantum-Computing-as-a-Service (QCaaS) Provider Comparison -- v3
====================================================================
Rebuilt with each provider's CURRENT flagship chip (checked against
live sources), replacing outdated chips used in earlier versions:
  IBM:        Heron r2 (2024)   -> Nighthawk / ibm_miami (Jan 2026)
  Quantinuum: "H2" data was actually mislabeled 2021 H1-2 data
                                -> Helios (Nov 2025, Nature-published 2026)
  Rigetti:    Ankaa-3 (Dec 2024) -> Cepheus-1-108Q (Apr 2026 GA)
  IonQ:       Forte retained (still commercially live);
              Tempo retained (published spec, not yet fully GA)

SOURCES:
  IBM Nighthawk    : arXiv:2606.20017 -- median 2Q error 2.69e-3 (sourced)
                     1Q fidelity & gate timing: no Nighthawk-specific figure
                     found; Heron-generation proxy used (ESTIMATE)
  IonQ Forte       : ionq.com/quantum-systems/compare (sourced fidelity)
                     arXiv:2506.07866 (sourced gate timing: 130us/950us)
  IonQ Tempo       : ionq.com/quantum-systems/compare (sourced fidelity;
                     system not yet fully GA as of this check)
                     Timing: Forte proxy (ESTIMATE)
  Quantinuum Helios: Nature (2026) DOI:10.1038/s41586-026-10676-4 --
                     1Q infidelity 2.5e-5, 2Q infidelity 7.9e-4 (sourced)
                     Gate timing: no Helios-specific ns/us figure found;
                     general trapped-ion range (10-200us 2Q) used as
                     ESTIMATE, taking IonQ Forte's sourced timing as proxy
  Rigetti Cepheus-1-108Q: Rigetti/AWS GA announcements (Apr 2026) --
                     99.1% median 2Q fidelity, ~60ns 2Q gate time,
                     99.9% median 1Q fidelity (all sourced)
                     1Q gate timing: no exact ns figure found; ESTIMATE
                     based on typical 1Q/2Q ratio for this modality

NOTE: figures marked ESTIMATE are proxies, not measurements of the
named device itself. This is a snapshot as of August 2026 -- given
how fast this field moves, re-verify before using in a final report.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

providers = {
    "IBM Nighthawk": {
        "modality": "Superconducting",
        "qubits": 120,
        "fidelity_1q": 0.9995,          # ESTIMATE (Heron-gen proxy)
        "fidelity_2q": 0.99731,         # sourced (arXiv:2606.20017)
        "gate_time_1q_ns": 35.5,        # ESTIMATE (Heron-gen proxy)
        "gate_time_2q_ns": 64,          # ESTIMATE (Heron-gen proxy)
    },
    "IonQ Forte": {
        "modality": "Trapped ion",
        "qubits": 36,
        "fidelity_1q": 0.9998,          # sourced
        "fidelity_2q": 0.996,           # sourced
        "gate_time_1q_ns": 130_000,     # sourced
        "gate_time_2q_ns": 950_000,     # sourced
    },
    "IonQ Tempo": {
        "modality": "Trapped ion",
        "qubits": 100,
        "fidelity_1q": 0.9999,          # sourced
        "fidelity_2q": 0.999,           # sourced
        "gate_time_1q_ns": 130_000,     # ESTIMATE (Forte proxy)
        "gate_time_2q_ns": 950_000,     # ESTIMATE (Forte proxy)
    },
    "Quantinuum Helios": {
        "modality": "Trapped ion",
        "qubits": 98,
        "fidelity_1q": 0.999975,        # sourced (Nature 2026)
        "fidelity_2q": 0.99921,         # sourced (Nature 2026)
        "gate_time_1q_ns": 130_000,     # ESTIMATE (IonQ Forte proxy)
        "gate_time_2q_ns": 950_000,     # ESTIMATE (IonQ Forte proxy)
    },
    "Rigetti Cepheus-1-108Q": {
        "modality": "Superconducting",
        "qubits": 108,
        "fidelity_1q": 0.999,           # sourced
        "fidelity_2q": 0.991,           # sourced
        "gate_time_1q_ns": 40,          # ESTIMATE (typical 1Q/2Q ratio)
        "gate_time_2q_ns": 60,          # sourced
    },
}

def project_provider(spec, n_qubits, depth):
    n_pairs = max(1, n_qubits // 2)
    n_1q_gates = n_qubits * depth
    n_2q_gates = n_pairs * depth
    fidelity = (spec["fidelity_1q"] ** n_1q_gates) * (spec["fidelity_2q"] ** n_2q_gates)
    time_s = (n_1q_gates * spec["gate_time_1q_ns"] +
              n_2q_gates * spec["gate_time_2q_ns"]) * 1e-9
    return fidelity, time_s

N_QUBITS = 20
DEPTHS = np.arange(1, 101, 2)

results = {name: {"fidelity": [], "time_s": []} for name in providers}
for depth in DEPTHS:
    for name, spec in providers.items():
        fid, t = project_provider(spec, N_QUBITS, depth)
        results[name]["fidelity"].append(fid)
        results[name]["time_s"].append(t)

sample_depth = 50
print(f"Projected results for a {N_QUBITS}-qubit circuit, depth={sample_depth} layers:\n")
print(f"{'Provider':<24}{'Modality':<16}{'Qubits':>7}{'Fidelity':>12}{'Time (ms)':>14}")
for name, spec in providers.items():
    fid, t = project_provider(spec, N_QUBITS, sample_depth)
    print(f"{name:<24}{spec['modality']:<16}{spec['qubits']:>7}"
          f"{fid:>12.4f}{t*1000:>14.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for name in providers:
    ax1.plot(DEPTHS, results[name]["fidelity"], label=name, marker=".", markersize=3)
ax1.set_xlabel("Circuit depth (layers)")
ax1.set_ylabel("Projected fidelity")
ax1.set_title(f"Projected Fidelity vs Circuit Depth ({N_QUBITS}-qubit circuit)\n"
              "Current flagship chips (Aug 2026)")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

for name in providers:
    ax2.plot(DEPTHS, np.array(results[name]["time_s"]) * 1000, label=name, marker=".", markersize=3)
ax2.set_xlabel("Circuit depth (layers)")
ax2.set_ylabel("Projected execution time (ms)")
ax2.set_yscale("log")
ax2.set_title(f"Projected Execution Time vs Circuit Depth ({N_QUBITS}-qubit circuit)\n"
              "Current flagship chips (Aug 2026)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("qcaas_comparison_v3.png", dpi=150)
print("\nSaved chart to qcaas_comparison_v3.png")

import csv
with open("qcaas_comparison_v3.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["provider", "depth", "fidelity", "time_s"])
    for name in providers:
        for d, fid, t in zip(DEPTHS, results[name]["fidelity"], results[name]["time_s"]):
            writer.writerow([name, d, fid, t])
print("Saved full sweep data to qcaas_comparison_v3.csv")
