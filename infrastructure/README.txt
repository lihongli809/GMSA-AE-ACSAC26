GMSA AE - Validated Reference Environment

Validated reference environment (NOT a strict minimum requirement):

  OS:           Linux
  Python:       3.8.20
  PyTorch:      1.13.1+cu117
  torchvision:  0.14.1+cu117
  GPU:          2 x NVIDIA RTX 3080 (10GB VRAM each)
  Driver:       NVIDIA 575.57.08 (authors' server)

An NVIDIA GPU is required by the current implementation because
dp/noise_add.py calls .cuda() unconditionally in the DP noise path. The
validated configuration uses 10GB-VRAM GPUs; no claim is made that 10GB is a
strict minimum.

PyTorch / torchvision 1.13.1+cu117 are NOT plain PyPI packages; they must be
installed from the PyTorch CUDA 11.7 wheel index. See install.sh for the
exact install command (stage-2 draft).