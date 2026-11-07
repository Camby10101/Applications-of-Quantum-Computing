"""
Method 3 (revised): A fairer classical vs quantum kernel "competition"
=========================================================================
Instead of relying on a single dataset specifically engineered to favour
a quantum feature map, this runs the SAME classical SVM (RBF kernel) vs
quantum kernel SVM (ZZFeatureMap) comparison across FIVE different
datasets -- a mix of real-world and standard synthetic benchmarks, plus
the original engineered dataset as one case among several. This gives
an honest picture of where quantum shows an advantage, where it doesn't,
and where the two are comparable.

Datasets used:
  1. Iris (real-world, 2-class subset)      -- classic, simple, well-separated
  2. Breast Cancer Wisconsin (real-world)   -- real medical diagnostic data
  3. make_moons (synthetic, non-linear)     -- standard sklearn benchmark
  4. make_circles (synthetic, non-linear)   -- standard sklearn benchmark
  5. ad_hoc_data (engineered)               -- quantum-favoured, for comparison

Each dataset is reduced to 2 features (PCA where needed) so it's
compatible with a 2-qubit ZZFeatureMap, and each comparison is repeated
across multiple trials with different train/test splits for reliability.
"""

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

N_TRIALS = 30           # matched to original single-dataset script
FEATURE_DIM = 2
TRAINING_SIZE = 30
TEST_SIZE = 15
GAP = 0.3
feature_map = zz_feature_map(feature_dimension=FEATURE_DIM, reps=2)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)


def prep_features(X, y, n_features=2):
    """Reduce to 2 features and scale to [0, 2*pi) for quantum encoding."""
    if X.shape[1] > n_features:
        X = PCA(n_components=n_features, random_state=0).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit_transform(X)
    return X, y


def run_trial(X, y, seed=0):
    """Matched to original script: fixed TRAINING_SIZE/TEST_SIZE draw per trial."""
    rng = np.random.default_rng(seed)
    n_total = TRAINING_SIZE + TEST_SIZE
    idx = rng.choice(len(X), size=min(n_total, len(X)), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, train_size=TRAINING_SIZE,
        test_size=min(TEST_SIZE, len(X_sub) - TRAINING_SIZE),
        random_state=seed, stratify=y_sub
    )

    classical = SVC(kernel="rbf")
    classical.fit(X_train, y_train)
    classical_acc = accuracy_score(y_test, classical.predict(X_test))

    quantum = SVC(kernel=quantum_kernel.evaluate)
    quantum.fit(X_train, y_train)
    quantum_acc = accuracy_score(y_test, quantum.predict(X_test))

    return classical_acc, quantum_acc


def build_datasets():
    datasets = {}

    iris = load_iris()
    mask = iris.target != 2
    datasets["Iris (real)"] = (iris.data[mask], iris.target[mask])

    bc = load_breast_cancer()
    datasets["Breast Cancer (real)"] = (bc.data, bc.target)

    X, y = make_moons(n_samples=150, noise=0.25, random_state=0)
    datasets["Moons (synthetic)"] = (X, y)

    X, y = make_circles(n_samples=150, noise=0.15, factor=0.4, random_state=0)
    datasets["Circles (synthetic)"] = (X, y)

    train_f, train_l, test_f, test_l = ad_hoc_data(
        training_size=TRAINING_SIZE, test_size=TEST_SIZE, n=FEATURE_DIM, gap=GAP, one_hot=False
    )
    X = np.vstack([train_f, test_f])
    y = np.concatenate([train_l, test_l])
    datasets["Ad Hoc (engineered)"] = (X, y)

    return datasets


def main():
    datasets = build_datasets()
    results = {}

    for name, (X, y) in datasets.items():
        X, y = prep_features(X, y)
        classical_accs, quantum_accs = [], []
        print(f"\nRunning {name} ({N_TRIALS} trials)...")
        for trial in range(N_TRIALS):
            c_acc, q_acc = run_trial(X, y, seed=trial)
            classical_accs.append(c_acc)
            quantum_accs.append(q_acc)
        results[name] = {
            "classical_mean": np.mean(classical_accs),
            "classical_std": np.std(classical_accs),
            "quantum_mean": np.mean(quantum_accs),
            "quantum_std": np.std(quantum_accs),
        }
        print(f"  Classical: {np.mean(classical_accs):.3f} +/- {np.std(classical_accs):.3f}  "
              f"Quantum: {np.mean(quantum_accs):.3f} +/- {np.std(quantum_accs):.3f}")

    print("\n=== Summary: Classical vs Quantum Kernel Across 5 Datasets ===")
    print(f"{'Dataset':<24}{'Classical acc':>16}{'Quantum acc':>16}{'Winner':>12}")
    for name, r in results.items():
        winner = "Quantum" if r["quantum_mean"] > r["classical_mean"] else \
                 "Classical" if r["classical_mean"] > r["quantum_mean"] else "Tie"
        print(f"{name:<24}{r['classical_mean']:>10.3f}+/-{r['classical_std']:.2f}  "
              f"{r['quantum_mean']:>8.3f}+/-{r['quantum_std']:.2f}  {winner:>12}")

    import csv
    with open("qml_fair_competition_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "classical_mean", "classical_std",
                          "quantum_mean", "quantum_std"])
        for name, r in results.items():
            writer.writerow([name, r["classical_mean"], r["classical_std"],
                              r["quantum_mean"], r["quantum_std"]])
    print("\nSaved results to qml_fair_competition_results.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    classical_means = [results[n]["classical_mean"] for n in names]
    classical_stds = [results[n]["classical_std"] for n in names]
    quantum_means = [results[n]["quantum_mean"] for n in names]
    quantum_stds = [results[n]["quantum_std"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width/2, classical_means, width, yerr=classical_stds, capsize=4,
           label="Classical SVM (RBF)", color="#4C72B0")
    ax.bar(x + width/2, quantum_means, width, yerr=quantum_stds, capsize=4,
           label="Quantum Kernel SVM (ZZFeatureMap)", color="#C44E52")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random guess baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Test set accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Classical vs Quantum Kernel SVM Across Five Datasets\n"
                 f"(mean \u00b1 std over {N_TRIALS} trials each)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("qml_fair_competition.png", dpi=150)
    print("Saved chart to qml_fair_competition.png")


if __name__ == "__main__":
    main()
