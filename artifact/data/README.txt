Dataset Availability
--------------------
The preprocessed FEMNIST dataset is not bundled with this artifact.
Canonical-result verification (the minimal viable evaluation path) does
not require the dataset.

For the original FEMNIST experiment configuration, LEAF preprocessing
commands, and expected data paths, see the FEMNIST Data section in the
top-level README.txt.

End-to-end FEMNIST reproduction additionally requires obtaining and
preprocessing the LEAF FEMNIST dataset locally. The artifact does not
redistribute that dataset.

CIFAR-10 data are also not bundled. The frozen reproduction entry uses
download=False and expects the dataset under `./data/cifar/`. Optional
CIFAR-10 reproduction therefore requires local dataset preparation.
