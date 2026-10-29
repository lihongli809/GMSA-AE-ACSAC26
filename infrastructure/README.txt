GMSA AE - Validated Reference Environment

Validated reference environment (NOT a strict minimum requirement):

  OS:           Linux
  Python:       3.8.20
  PyTorch:      1.13.1+cu117
  torchvision:  0.14.1+cu117
  GPU:          2 x NVIDIA RTX 3080 (10GB VRAM each)
  Driver:       NVIDIA 575.57.08 (authors' server)

Canonical-result verification uses the claims/*/run.py Python verifiers. These
verifiers use only the Python 3 standard library and are platform-independent:
they do not require Linux, a GPU, CUDA, or prepared datasets. The provided
run.sh wrappers use Bash and assume a Linux shell; running the underlying
Python verifier directly has also been confirmed on Windows Python.

An NVIDIA GPU is required for optional scaled/full research-code reproduction
because the frozen research-code DP noise path in dp/noise_add.py calls .cuda()
unconditionally. Full FEMNIST end-to-end reproduction additionally requires
locally prepared LEAF FEMNIST data, which is not bundled. The validated
configuration uses 10GB-VRAM GPUs; no claim is made that 10GB is a strict
minimum. Minimum RAM/disk for canonical-result verification have not been
measured.

PyTorch / torchvision 1.13.1+cu117 are NOT plain PyPI packages; they must be
installed from the PyTorch CUDA 11.7 wheel index. See install.sh for the
exact installation command.
