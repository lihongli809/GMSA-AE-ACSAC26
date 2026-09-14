#!/usr/bin/env python3
"""Claim 2 verifier - Adaptive-MSA canonical-result statistical verification.

Reads the bundled _final CSVs and independently recomputes Final-5 Acc, TPR,
FPR, and LSA Gain against the documented rebuttal targets (tolerance 1e-4).
This is canonical-result statistical verification, NOT full retraining.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _verify_common import RESULTS, to_float, mean  # noqa: E402

TARGETS = {
    "cifar": {"gain": 0.0413, "tpr": 1.0000, "fpr": 0.0111},
    "femnist": {"gain": 0.0751, "tpr": 0.9617, "fpr": 0.0192},
}
FILES = {
    "cifar": "cifar_adaptive_msa_gmsa_r0.20_final.csv",
    "femnist": "femnist_adaptive_msa_gmsa_r0.20_final.csv",
}
TOL = 1e-4


def verify(label):
    path = os.path.join(RESULTS, "adaptive_msa", FILES[label])
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    accs = [to_float(r["Acc"]) for r in rows]
    final5 = [a for a in accs[-5:] if a is not None]
    final5_acc = mean(final5)

    present = []
    for r in rows:
        orig = to_float(r.get("Original_LSA", ""))
        best = to_float(r.get("Best_LSA", ""))
        gain_csv = to_float(r.get("LSA_Gain", ""))
        if orig is not None and gain_csv is not None:
            present.append((r, orig, best, gain_csv))

    tpr = mean([to_float(r["TPR"]) for r, _, _, _ in present])
    fpr = mean([to_float(r["FPR"]) for r, _, _, _ in present])
    gain_csv_mean = mean([g for _, _, _, g in present])
    recomp_rows = [p for p in present if p[2] is not None]
    gain_recomp = mean([b - o for _, o, b, _ in recomp_rows])

    tgt = TARGETS[label]
    diffs = {
        "gain": abs(gain_csv_mean - tgt["gain"]),
        "tpr": abs(tpr - tgt["tpr"]),
        "fpr": abs(fpr - tgt["fpr"]),
    }
    status = {k: ("MATCH" if d < TOL else "MISMATCH") for k, d in diffs.items()}
    consistent = gain_recomp is not None and abs(gain_csv_mean - gain_recomp) < TOL

    print("  %s (attack-present rows: %d/%d):" % (label, len(present), len(rows)))
    print("    Final-5 Acc            : %.17g" % final5_acc)
    print("    LSA Gain  actual=%.17g target=%.4f diff=%.3g -> %s"
          % (gain_csv_mean, tgt["gain"], diffs["gain"], status["gain"]))
    print("    LSA Gain  recomputed=%.17g  CSV-vs-recomputed=%s"
          % (gain_recomp if gain_recomp is not None else float("nan"), "CONSISTENT" if consistent else "INCONSISTENT"))
    print("    TPR       actual=%.17g target=%.4f diff=%.3g -> %s"
          % (tpr, tgt["tpr"], diffs["tpr"], status["tpr"]))
    print("    FPR       actual=%.17g target=%.4f diff=%.3g -> %s"
          % (fpr, tgt["fpr"], diffs["fpr"], status["fpr"]))

    return all(s == "MATCH" for s in status.values()) and consistent


def main():
    ok = True
    for label in ("cifar", "femnist"):
        if not verify(label):
            ok = False
    if ok:
        print("Adaptive-MSA canonical-result statistical verification: PASS")
        return 0
    print("Adaptive-MSA canonical-result statistical verification: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())