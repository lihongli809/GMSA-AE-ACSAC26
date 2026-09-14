#!/usr/bin/env python3
"""AE reproduction launcher for Claim 1 - Main MSA Defense.

Runs the FROZEN snapshot artifact/src/main.py without modifying it. This is a
scaled/full reproduction entry, NOT the canonical-result verifier (run.py).
Outputs go under artifact/src/results/ (AE-side only).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "..", "artifact", "src"))

from _ae_common import SRC_DIR, install_compat_stubs, load_entry  # noqa: E402

ALLOWED = {
    "dataset": {"cifar", "femnist"},
    "defense": {"gm_raf", "fedavg"},
    "custom_attack": {"msa"},
    "model": {"cnn"},
}


def pre_validate(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset")
    p.add_argument("--defense")
    p.add_argument("--custom_attack")
    p.add_argument("--model")
    known, _ = p.parse_known_args(argv)
    for key, allowed in ALLOWED.items():
        val = getattr(known, key)
        if val is not None and val not in allowed:
            raise SystemExit(
                "AE Claim 1 repro launcher: --%s=%r is outside this claim scope "
                "(allowed: %s)." % (key, val, ", ".join(sorted(allowed)))
            )


def main():
    pre_validate(sys.argv[1:])
    install_compat_stubs()
    os.chdir(SRC_DIR)
    module = load_entry("main.py", "_ae_claim1_main")
    module.main()


if __name__ == "__main__":
    main()