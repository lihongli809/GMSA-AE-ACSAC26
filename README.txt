GMSA - ACSAC 2026 Artifact Evaluation (Paper #330)
===================================================

1. Artifact Overview
--------------------
Title: GMSA: A Geometric Momentum and Subspace Analysis Framework against
Model Shuffle Attacks in Differentially Private Federated Learning.

This artifact packages the code and canonical result assets for three
artifact claims associated with the accepted ACSAC 2026 paper and its
Author Response:

  1. Main MSA Defense       (CIFAR-10 + FeMnist, FedAvg vs GMSA under MSA)
  2. Adaptive-MSA           (CIFAR-10 + FeMnist, LSA Gain / TPR / FPR)
  3. Hyperparameter Sensitivity (CIFAR-10, alpha_mom / kappa / MAD multiplier)

Paper Versioning Note
---------------------
The paper PDF submitted with this artifact is the accepted ACSAC paper
snapshot. The artifact also includes extended evaluations conducted during
the review phase and described in the Author Response, including Adaptive-MSA
and hyperparameter sensitivity experiments. These additional results are
provided as canonical artifact results and are intended to support the
corresponding analysis in the camera-ready version of the paper.

The artifact contains:
  - a frozen source snapshot of the three-claim code closure (artifact/src/);
  - canonical result assets copied byte-for-byte from the authors' server
    (artifact/results/);
  - claim-level canonical-result verifiers (claims/*/run.py + run.sh) and
  optional reproduction entries (claims/*/run_repro.py + run_repro.sh);
  - environment and installation information (install.sh, AE_requirements.txt).

The Early-stage MOM Ablation (Table 10) is NOT part of this artifact.

2. Current Evaluation Scope
---------------------------
The artifact is prepared to support evaluation for the Available and
Functional badges. This statement describes preparation intent only; it does
not mean a badge has already been granted, and this artifact does not claim
Reproduced.

3. Verification Philosophy
--------------------------
Three distinct activities are separated:

  - Canonical Result Verification: read the bundled canonical results and
    independently recompute statistics from the saved CSV/NPY files.
  - Scaled Reproduction: re-execute the code with a shortened configuration,
    as engineering-level reproduction / smoke support only.
  - Full Reproduction: re-run the complete original 100/50-round training
    protocol.

The current artifact primarily uses canonical-result verification.
Canonical-result verification should not be interpreted as full retraining.

4. Environment
--------------
An NVIDIA GPU is required by the current implementation: dp/noise_add.py
calls .cuda() unconditionally in the DP noise path. Do not assume CPU-only
execution.

Validated reference environment (authors' server):
  - Linux
  - Python 3.8.20
  - PyTorch 1.13.1+cu117
  - torchvision 0.14.1+cu117
  - 2 x NVIDIA RTX 3080 (10GB VRAM each)

"10GB is the strict minimum" is NOT claimed; 10GB VRAM is only the validated
reference configuration.

5. Non-invasive Source Snapshot
-------------------------------
artifact/src/ holds a frozen, byte-for-byte snapshot of the original research
source needed by the three claims. The original research project files were
not modified.

Claim canonical-result verifiers live in claims/*/run.py. Optional
reproduction launchers live in claims/*/run_repro.py. The shared compatibility
layer is claims/_ae_common.py. It provides import-level compatibility only: a small,
controlled set of stubs for modules that the three claims never use, so that
unrelated out-of-scope branches do not block startup due to missing imports.

The stub layer:
  - covers only isolated modules outside the three claim scopes;
  - fails explicitly if a stubbed name is actually invoked;
  - does not silently fabricate results;
  - does not change GMSA algorithm logic.

It does not remove all runtime dependencies and does not reproduce all
original execution paths.

6. FEMNIST Data
---------------
The bundled artifact does not include the preprocessed FEMNIST dataset.
The FEMNIST experiments use the LEAF FEMNIST benchmark.

For the original experiments, the LEAF preprocessing pipeline used
non-IID sampling with a 5% sampling fraction, followed by sample-level
train/test splitting. The corresponding preprocessing command was:

  ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample \
    --smplseed 1776501746 \
    --spltseed 1776501820

The resulting data are expected at:

  leaf/data/femnist/data/train/
  leaf/data/femnist/data/test/

relative to the project root.

The preprocessing metadata and checksums are maintained with the original
LEAF data on the reference system. The artifact does not redistribute the
preprocessed FEMNIST dataset. Preparing the FEMNIST data is required only
for optional end-to-end retraining; the claim verification scripts operate
on the bundled canonical results.

7. Table 10
-----------
Table 10 / Early-stage MOM Ablation is outside the current artifact claim
scope. Status: DO NOT CLAIM.

8. Layout
---------
  README.txt             this file
  LICENSE                upstream MIT license
  use.txt                usage notes
  install.sh             dependency install (stage-2 draft)
  AE_requirements.txt    minimal runtime dependency set
  metadata.toml          metadata generated by the official ACSAC artifact metadata tool
  CLAIM_MAPPING.md       claim -> paper/code/results/verification mapping
  artifact/src/          frozen three-claim source snapshot
  artifact/results/      canonical results (verbatim)
  artifact/data/         reserved for dataset-related notes/materials
  claims/                per-claim claim.txt / run.py / run.sh / run_repro.* / expected/
  infrastructure/        environment + metadata guidance

9. License / Attribution
------------------------
LICENSE is the upstream MIT license (Copyright (c) 2023 Ming Yang), carried
forward unchanged. The original MSA attack code is based on
shaoxiongji/federated-learning and Yang et al. (2023, Information Sciences);
see the upstream repository documentation for full attribution.