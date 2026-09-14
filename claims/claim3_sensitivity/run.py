#!/usr/bin/env python3
"""Claim 3 verifier - Hyperparameter Sensitivity file integrity check.

Verification type: Canonical Result Verification (file integrity only). This
does NOT reproduce Table 8 paper values.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _verify_common import RESULTS  # noqa: E402

FILES = [
    "cifar_msa_sens_center_r0.2.csv",
    "cifar_msa_sens_am0.1_r0.2.csv",
    "cifar_msa_sens_am0.5_r0.2.csv",
    "cifar_msa_sens_k1.0_r0.2.csv",
    "cifar_msa_sens_k2.0_r0.2.csv",
    "cifar_msa_sens_m2.0_r0.2.csv",
    "cifar_msa_sens_m4.0_r0.2.csv",
]
EXPECTED_HEADER = ["Round", "Acc", "Loss", "Rej", "TPR", "FPR"]


def check(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return False, "empty file"
    header, data = rows[0], rows[1:]
    if header != EXPECTED_HEADER:
        return False, "bad header %s" % header
    if len(data) != 20:
        return False, "rows=%d expected=20" % len(data)
    rounds = []
    for r in data:
        try:
            rounds.append(int(float(r[0])))
        except ValueError:
            return False, "bad round value %r" % r[0]
    if rounds != list(range(rounds[0], rounds[0] + len(rounds))):
        return False, "non-contiguous round sequence"
    for r in data:
        for cell in r[1:]:
            v = cell.strip()
            if v == "":
                return False, "empty cell"
            if v.lower() in ("nan", "none"):
                return False, "nan cell"
    return True, "ok (20 contiguous rounds)"


def main():
    base = os.path.join(RESULTS, "sensitivity")
    passed = 0
    failed = 0
    for name in FILES:
        path = os.path.join(base, name)
        if not os.path.exists(path):
            print("  FAIL %-38s missing" % name)
            failed += 1
            continue
        ok, msg = check(path)
        print("  %s %-38s %s" % ("PASS" if ok else "FAIL", name, msg))
        if ok:
            passed += 1
        else:
            failed += 1
    print("%d/%d files PASS" % (passed, passed + failed))
    if failed:
        print("Sensitivity integrity check: FAIL")
        return 1
    print("Sensitivity integrity check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())