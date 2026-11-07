"""
Theoretical Quantum Advantage Projection
==========================================
This does NOT simulate a quantum computer (that's exponentially expensive
classically, as shown in benchmark.py). Instead it PROJECTS how long a
computation would take on real quantum hardware vs classical hardware,
using known complexity formulas and realistic per-operation timings.
This is the standard way quantum advantage is estimated/argued in the
literature before/without access to large fault-tolerant hardware.

Two case studies:
  1. Grover's search:  classical O(N)      vs  quantum O(sqrt(N))
  2. Shor's factoring: classical O(exp(...)) [GNFS] vs quantum O((log N)^3)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. GROVER'S SEARCH vs CLASSICAL LINEAR SEARCH
# ---------------------------------------------------------------
# Realistic per-operation timings (order-of-magnitude, widely cited):
#   classical CPU op   ~ 1 nanosecond   (a few GHz clock)
#   quantum gate op    ~ 100 nanoseconds (superconducting qubit gate time)
# Grover needs ~ (pi/4)*sqrt(N) queries/oracle calls.

t_classical_op = 1e-9     # seconds per classical operation
t_quantum_gate = 100e-9   # seconds per quantum gate/oracle call

N = np.logspace(1, 24, 200)  # database size from 10 up to 10^24 items

T_classical_search = N * t_classical_op
T_quantum_search = (np.pi / 4) * np.sqrt(N) * t_quantum_gate

# Find crossover point (where quantum becomes faster)
crossover_idx = np.argmax(T_quantum_search < T_classical_search)
N_crossover = N[crossover_idx]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N, T_classical_search, label="Classical linear search: O(N)")
ax.plot(N, T_quantum_search, label="Grover's algorithm: O(sqrt(N))")
ax.axvline(N_crossover, color="gray", linestyle="--", alpha=0.6,
           label=f"Crossover ~ N = {N_crossover:.1e}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Search space size N (number of items)")
ax.set_ylabel("Projected time (seconds, log scale)")
ax.set_title("Projected Time: Grover's Search vs Classical Search")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("grover_advantage.png", dpi=150)
print(f"Grover crossover point: N ~ {N_crossover:.2e} items "
      f"(quantum becomes faster beyond this)")

# ---------------------------------------------------------------
# 2. SHOR'S FACTORING vs CLASSICAL GNFS
# ---------------------------------------------------------------
# Classical: General Number Field Sieve (best known classical factoring algo)
#   T_classical(n) = exp( c * (ln N)^(1/3) * (ln ln N)^(2/3) ), c ~ 1.923
# Quantum: Shor's algorithm
#   T_quantum(n) = k * n^3  gate operations, n = number of bits of N
# We express both in terms of n = bit-length of the number being factored.

def gnfs_operations(n_bits):
    """Estimated number of operations for GNFS as a function of bit length."""
    ln_N = n_bits * np.log(2)
    c = 1.923
    return np.exp(c * (ln_N) ** (1/3) * (np.log(ln_N)) ** (2/3))

def shor_gate_count(n_bits):
    """Estimated gate count for Shor's algorithm ~ O(n^3) (with small constant)."""
    return 100 * n_bits ** 3   # constant factor is a rough engineering estimate

bit_lengths = np.linspace(32, 4096, 200)

T_classical_factor = gnfs_operations(bit_lengths) * t_classical_op
T_quantum_factor = shor_gate_count(bit_lengths) * t_quantum_gate

crossover_idx2 = np.argmax(T_quantum_factor < T_classical_factor)
n_crossover = bit_lengths[crossover_idx2]

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(bit_lengths, T_classical_factor, label="Classical GNFS (best known)")
ax2.plot(bit_lengths, T_quantum_factor, label="Shor's algorithm (quantum)")
ax2.axvline(n_crossover, color="gray", linestyle="--", alpha=0.6,
            label=f"Crossover ~ {n_crossover:.0f}-bit number")
ax2.set_yscale("log")
ax2.set_xlabel("Number size (bits) — e.g. RSA-2048 = 2048 bits")
ax2.set_ylabel("Projected time (seconds, log scale)")
ax2.set_title("Projected Time: Shor's Algorithm vs Classical Factoring")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("shor_advantage.png", dpi=150)
print(f"Shor crossover point: ~ {n_crossover:.0f}-bit numbers "
      f"(quantum becomes faster beyond this)")
