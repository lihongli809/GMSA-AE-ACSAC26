"""GMSA-AE shared helpers for canonical-result verifiers (stdlib only)."""
import os

ARTIFACT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
RESULTS = os.path.join(ARTIFACT_ROOT, "artifact", "results")


def to_float(s):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def mean(vals):
    return sum(vals) / len(vals) if vals else None