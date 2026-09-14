# GMSA AE - Official Compliance Audit (stage 7)

This is a packaging-level compliance snapshot, not a badge claim.

## Required layout
- README.txt          : present
- license.txt         : present (MIT, same content as LICENSE)
- use.txt             : present
- artifact/           : present (src/ + results/ + data/)
- infrastructure/     : present (README.txt + METADATA_GUIDE.md + RELEASE_AUDIT.md)
- claims/             : present (3 claim folders)

## Claims
- claim1_main_msa     : claim.txt + run.sh + run.py + expected/expected_output.txt
- claim2_adaptive_msa : claim.txt + run.sh + run.py + expected/expected_output.txt
- claim3_sensitivity  : claim.txt + run.sh + run.py + expected/expected_output.txt
- Each claim also has run_repro.py / run_repro.sh (optional reproduction entry).

## Verifier smoke test (local, not clean Linux+CUDA)
- bash -n install.sh and all run.sh/run_repro.sh: exit 0
- claim1 run.sh: PASS (exit 0)
- claim2 run.sh: PASS (exit 0), all six metrics MATCH at 1e-4
- claim3 run.sh: PASS (exit 0), 7/7 files

## install.sh
- Documents intended install for the verified server environment.
- NOT yet validated on a clean Linux+CUDA host (explicit warning added).

## Public infrastructure
- NOT FEASIBLE as-is: requires Python 3.8 + torch 1.13.1+cu117 + CUDA 11.7,
  which conflicts with modern Colab runtimes (Python 3.10+, CUDA 12.x).
  No public-infra adaptation layer is provided.

## FEMNIST
- Preprocessed FEMNIST data is NOT bundled.
- Placeholder for official access instructions present; no fabricated URL.

## metadata.toml
- PLACEHOLDER; must be generated with the official ACSAC artmeta tool.