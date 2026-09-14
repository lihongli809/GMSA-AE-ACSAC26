# GMSA AE - Claim Mapping (stage 6)

Status labels are evidence-tiered. "VERIFIED" refers only to the stated
verification type, never to full paper reproduction.

---

## Claim 1 - Main MSA Defense
- Paper Figure/Table: Fig. 4 (FedAvg under MSA), Fig. 5 (GMSA defends MSA),
  and the r0.20 row of Table 4. Table 3 (FLTrust/FLAME) and Table 5
  (ablation) are NOT claimed; their files are not bundled. (CODE-DERIVED)
- Source Code: artifact/src/main.py; aggreagation_method/{gm_raf,fed_aggregation}.py;
  attacks/poisoning.py; dp/{noise_add,privacy}.py; models/{Nets,Update,test}.py;
  utils/{options_cifar,sampling}.py
- Canonical Result Files:
    artifact/results/main_msa/cifar_msa_def_r0.20.csv (+ _scores.npy)
    artifact/results/main_msa/cifar_msa_fedavg_r0.20.csv
    artifact/results/main_msa/femnist_msa_def_r0.20.csv (+ _scores.npy)
    artifact/results/main_msa/femnist_msa_fedavg_r0.20.csv
- Verification Script: claims/claim1_main_msa/run.py (verifier);
  claims/claim1_main_msa/run_repro.py (reproduction entry)
- Verification Type: Canonical Result Verification (integrity)
- Status: VERIFIED - file presence + round counts + CSV columns + NPY presence.
  Paper-number reproduction: UNVERIFIED.

## Claim 2 - Adaptive-MSA
- Paper Figure/Table: Rebuttal Table 9
- Source Code: artifact/src/main_adaptive_msa.py; attacks/{poisoning,poisoning_adaptive_msa}.py;
  aggreagation_method/gm_raf.py; dp/noise_add.py
- Canonical Result Files:
    artifact/results/adaptive_msa/cifar_adaptive_msa_gmsa_r0.20_final.csv
      (+ _final_scores.npy, _final_adaptive_scores.npy)
    artifact/results/adaptive_msa/cifar_msa_gmsa_r0.20_final.csv (baseline)
    artifact/results/adaptive_msa/femnist_adaptive_msa_gmsa_r0.20_final.csv
      (+ _final_scores.npy, _final_adaptive_scores.npy)
    artifact/results/adaptive_msa/femnist_msa_gmsa_r0.20_final.csv (baseline)
- Verification Script: claims/claim2_adaptive_msa/run.py (verifier);
  claims/claim2_adaptive_msa/run_repro.py (reproduction entry)
- Verification Type: Canonical Result Verification (statistical)
- Status: VERIFIED - canonical-result statistical consistency.
    CIFAR:   Gain 0.0413 / TPR 1.0000 / FPR 0.0111  (MATCH, tol 1e-4)
    FeMnist: Gain 0.0751 / TPR 0.9617 / FPR 0.0192  (MATCH, tol 1e-4)

## Claim 3 - Hyperparameter Sensitivity
- Paper Figure/Table: Rebuttal Table 8
- Source Code: artifact/src/main_hyperparam_sensitivity.py;
  aggreagation_method/gm_raf_sensitivity.py
- Canonical Result Files:
    artifact/results/sensitivity/cifar_msa_sens_center_r0.2.csv
    artifact/results/sensitivity/cifar_msa_sens_am0.1_r0.2.csv
    artifact/results/sensitivity/cifar_msa_sens_am0.5_r0.2.csv
    artifact/results/sensitivity/cifar_msa_sens_k1.0_r0.2.csv
    artifact/results/sensitivity/cifar_msa_sens_k2.0_r0.2.csv
    artifact/results/sensitivity/cifar_msa_sens_m2.0_r0.2.csv
    artifact/results/sensitivity/cifar_msa_sens_m4.0_r0.2.csv
- Verification Script: claims/claim3_sensitivity/run.py (verifier);
  claims/claim3_sensitivity/run_repro.py (reproduction entry)
- Verification Type: Canonical Result Verification (file integrity)
- Status: VERIFIED - file integrity (7 files, header + 20 contiguous rounds,
  no empty rows, no NaN). Table 8 numeric reproduction: UNVERIFIED.

## Table 10 - Early-stage MOM Ablation
- Status: DO NOT CLAIM (outside current artifact scope).

## Out of scope
- HisMSA, DP-Poison, FedSurgeon, PAVF: not part of this artifact.