#!/usr/bin/env python3
"""Claim 1 verifier - Main MSA Defense canonical-result integrity check.

Verification type: Canonical Result Verification (file presence, round count,
CSV columns, NPY presence). This is NOT a full experiment reproduction.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _verify_common import RESULTS  # noqa: E402

EXPECTED_ROWS = {"cifar": 100, "femnist": 50}
REQUIRED_COLS = ["Round", "Acc", "Loss", "Rej", "TPR", "FPR"]


def check_csv(path, expected_rows):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return False, "empty file"
    header, data = rows[0], rows[1:]
    if header != REQUIRED_COLS:
        return False, "bad header %s" % header
    if len(data) != expected_rows:
        return False, "rows=%d expected=%d" % (len(data), expected_rows)
    return True, "ok (%d rounds)" % len(data)


def main():
    base = os.path.join(RESULTS, "main_msa")
    failures = []
    for ds, n in EXPECTED_ROWS.items():
        for kind in ("def", "fedavg"):
            path = os.path.join(base, "%s_msa_%s_r0.20.csv" % (ds, kind))
            if not os.path.exists(path):
                print("  FAIL %-40s missing" % os.path.basename(path))
                failures.append(path)
                continue
            ok, msg = check_csv(path, n)
            print("  %s %-40s %s" % ("PASS" if ok else "FAIL", os.path.basename(path), msg))
            if not ok:
                failures.append(path)
        npy = os.path.join(base, "%s_msa_def_r0.20_scores.npy" % ds)
        exists = os.path.exists(npy)
        print("  %s %-40s %s" % ("PASS" if exists else "FAIL", os.path.basename(npy), "exists" if exists else "missing"))
        if not exists:
            failures.append(npy)

    if failures:
        print("Claim 1 canonical-result integrity verification: FAIL (%d issue(s))" % len(failures))
        return 1
    print("Claim 1 canonical-result integrity verification: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())