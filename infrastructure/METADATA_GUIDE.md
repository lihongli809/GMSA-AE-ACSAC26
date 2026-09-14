GMSA AE - metadata.toml Generation Guide

metadata.toml at the artifact root is currently a PLACEHOLDER and must NOT be
submitted as-is.

Final metadata must be generated and/or validated by the authors using the
official ACSAC artifact metadata tool (artmeta). Do not hand-invent the
schema and do not fabricate field values.

Known facts to record when generating metadata:
  - Artifact type: Code / Dataset (per the current official options).
  - Current badge scope: Available + Functional evaluation support
    (Reproduced is NOT claimed at this stage).
  - Hardware: NVIDIA GPU required; validated reference = RTX 3080 10GB.
  - Software: Linux, Python 3.8.20, PyTorch 1.13.1+cu117, torchvision
    0.14.1+cu117.
  - Expected resources: GPU with CUDA 11.7-compatible PyTorch wheels.
  - Constraints: full 100/50-round runs are expensive; canonical-result
    verification is the primary evaluation path.
  - Public infrastructure: repository/artifact URL to be finalized before
    submission (Zenodo or equivalent permanent archive).

This guide does NOT specify artmeta command-line arguments, schema fields, or
download URLs, because those must be taken from the official tool
instructions.