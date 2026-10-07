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

3. Verification Philosophy, Expected Runtime & Success Criteria
---------------------------------------------------------------
Three distinct activities are separated:

  - Canonical Result Verification (Minimal Viable Path): reads the bundled
    canonical results and independently recomputes the documented integrity
    or statistical checks from CSV/NPY files.
    * GPU: Not required.
    * Dataset preparation: Not required.
    * Expected runtime: lightweight; in a reference validation run, the three
      claim verifiers completed in 0.064s, 0.014s, and 0.020s respectively
      (under 0.1 seconds total). Actual runtime depends on the execution environment.
    * Success criteria: the verifier exits with code 0 and prints the final
      PASS message. The reported checks satisfy the documented tolerances
      and consistency criteria in the corresponding expected output.

  - Scaled Reproduction (Optional): re-executes the frozen research code with
    a shortened configuration as engineering-level reproduction / smoke support.
    * GPU: Required.
    * Dataset preparation: Required.
    * Expected runtime: hardware- and configuration-dependent; no formal
      runtime benchmark was performed as part of this artifact package.
    * Success criteria: the selected configuration completes without an
      execution error and produces the documented output files.

  - Full Reproduction (Optional): re-runs the complete original 100/50-round
    training protocol.
    * GPU: Required.
    * Dataset preparation: Required.
    * Expected runtime: not benchmarked as part of this artifact package;
      substantially longer than canonical-result verification and dependent
      on GPU hardware and dataset preparation.

The current artifact primarily supports canonical-result verification for the
Available and Functional badges. Full end-to-end retraining is not required
for the current evaluation scope.

4. Environment & Resource Requirements
--------------------------------------
The artifact distinguishes the minimal canonical-result verification path
from optional scaled/full reproduction.

  - OS: Linux is the supported execution environment.
  - GPU: Not required for canonical-result verification. An NVIDIA GPU with
    CUDA-enabled PyTorch is required for optional scaled/full reproduction
    because the frozen research code calls .cuda() unconditionally in the
    DP noise path.
  - CPU/RAM: No strict minimum has been established for canonical verification.
    The validated reference environment used an Intel Core i7-12700K CPU and
    64 GB RAM.
  - Disk: No fixed minimum has been established for canonical verification.
    Full FEMNIST reproduction requires substantial local storage; the
    preprocessed FEMNIST dataset occupies approximately 20 GB on the
    authors' reference system.
  - Software: Python 3.8.20, PyTorch 1.13.1+cu117, and torchvision
    0.14.1+cu117 were used in the validated reference environment.
  - GUI: Not required.
  - Network: The canonical-result verification path can run offline after
    the repository is obtained. Network access is needed only when obtaining
    the repository, installing dependencies, or preparing external datasets.
  - APIs: No API keys or paid online services are required.
  - License: MIT.

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

10. Quick Start (Smallest Viable Evaluation Path)
-------------------------------------------------
From a Linux environment, the minimal canonical-result verification does not
require an NVIDIA GPU, CUDA, or dataset preparation.

1. Clone the repository
git clone https://github.com/lihongli809/GMSA-AE-ACSAC26.git
cd GMSA-AE-ACSAC26

2. Run the three canonical-result verifiers
bash claims/claim1_main_msa/run.sh
bash claims/claim2_adaptive_msa/run.sh
bash claims/claim3_sensitivity/run.sh

Each verifier is read-only with respect to the bundled canonical results.
A successful run exits with code 0 and prints a final PASS message. The
corresponding expected output is provided in each claim's expected/
directory.

Optional scaled/full reproduction requires the CUDA-enabled environment
described above:
bash install.sh

Dataset preparation is additionally required for end-to-end reproduction;
see the FEMNIST Data section and the claim-specific reproduction entries.
