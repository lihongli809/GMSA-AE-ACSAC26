"""GMSA AE - shared launcher machinery (AE-side only).

Runs the FROZEN entry snapshots under artifact/src/ without editing them.
The frozen entries top-level import modules that are outside the three AE
claims (fltrust, flame, poisoning_hismsa, dp_poison, resnet, load_loan).
Those modules are intentionally not part of the artifact closure.

Safety design:
  * Stubs raise RuntimeError when actually invoked, so they can never
    silently hide a real code path.
  * Each run.py pre-validates --dataset/--defense/--custom_attack/--model to
    the claim's supported configurations, making those stub branches
    unreachable.
"""
import importlib.util
import os
import sys
import types

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifact", "src")
)

STUB_SPECS = {
    "aggreagation_method.fltrust": ["FLTrust"],
    "aggreagation_method.flame": ["FLAME"],
    "attacks.poisoning_hismsa": ["hismsa_attack"],
    "attacks.dp_poison": ["LocalUpdateDPPoison_PGD", "test_backdoor"],
    "models.resnet": ["ResNet18"],
    "models.load_loan": ["get_loan_dataset"],
}


def _stub_name(module_name, attr_name):
    def _out_of_scope(*args, **kwargs):
        raise RuntimeError(
            "AE launcher: %s.%s is outside the three AE claims "
            "(Main MSA Defense / Adaptive-MSA / Hyperparameter Sensitivity). "
            "This configuration is refused by the launcher."
            % (module_name, attr_name)
        )
    _out_of_scope.__name__ = attr_name
    return _out_of_scope


def install_compat_stubs():
    installed = []
    for mod_name, attrs in STUB_SPECS.items():
        if mod_name in sys.modules:
            continue
        mod = types.ModuleType(mod_name)
        for attr in attrs:
            setattr(mod, attr, _stub_name(mod_name, attr))
        sys.modules[mod_name] = mod
        installed.append(mod_name)
    return installed


def load_entry(entry_filename, module_name):
    path = os.path.join(SRC_DIR, entry_filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module