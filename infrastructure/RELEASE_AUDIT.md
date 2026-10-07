# Release Audit Status (Current)

## 1. Frozen Code Closure
- Snapshot present in `artifact/src/`.
- The AE-side import-compatibility stub layer is limited to modules outside
  the three artifact claim scopes.
- Stubbed names fail explicitly if invoked; the layer does not fabricate
  results or alter GMSA algorithm logic.

## 2. Canonical Results
- Canonical result assets are present in `artifact/results/`.
- Result files match the claim-specific verification paths.
- The canonical-result verifiers have been executed successfully on the
  authors' Linux reference environment.

## 3. Claim Mapping
- `CLAIM_MAPPING.md` defines the three artifact claims and their
  claim-to-script/result mappings.
- Table 10 / Early-stage MOM Ablation is explicitly out of scope.

## 4. Metadata
- `metadata.toml` is present and contains the current artifact metadata.

## 5. FEMNIST Data
- Preprocessed FEMNIST data is not bundled.
- The top-level `README.txt` documents the original LEAF preprocessing
  configuration and expected data paths.
- End-to-end FEMNIST reproduction therefore requires local dataset
  preparation.

## 6. Reproduction Environment
- The validated reference environment uses Linux, Python 3.8.20,
  PyTorch 1.13.1+cu117, torchvision 0.14.1+cu117, and NVIDIA GPUs.
- `install.sh` documents the intended CUDA-enabled installation but has not
  been validated on a clean Linux+CUDA host.
- Canonical-result verification does not require the GPU environment.

## 7. Public Infrastructure
- The artifact is publicly available at:
  https://github.com/lihongli809/GMSA-AE-ACSAC26
- No public-cloud-specific adaptation layer is provided; optional reproduction
  therefore depends on the documented local CUDA environment.
