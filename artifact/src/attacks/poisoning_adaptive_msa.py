# -*- coding: utf-8 -*-
# ====================================================================
# 文件: attacks/poisoning_adaptive_msa.py   (NEW FILE - rebuttal only)
# Defense-aware Adaptive-MSA (Reviewer rebuttal 实验专用攻击模块)
#
# 冻结设计 (frozen spec):
#   - 不修改 attacks/poisoning.py / aggreagation_method/gm_raf.py / dp/* / main.py。
#   - candidate 0 直接调用原始 blind_model_shuffle(w, round_idx)。
#   - candidate 1..N-1 使用本模块复制的 MSA 变换 + 局部 random.Random(seed)。
#   - clipping 复用 dp.noise_add.clipping (禁止重新实现裁剪数学)。
#   - surrogate LSA 严格镜像 gm_raf.py:40 与 gm_raf.py:91-104:
#       layer_keys = [k for k in keys if 'weight' in k]
#       zero-norm 保护 1e-8, cosine eps=1e-8
#       score = mean_sim - 1.5 * np.std(layer_sims)   (np.std 默认 ddof=0)
#   - reference: anchor_ref = anchor_{t-1}; Round 0 fallback = w_glob_prev。
#   - 返回值: raw pre-clip candidate; 主流程随后执行真实的
#       Delta W -> clipping -> reconstruction -> Gaussian noise -> GMSA。
# ====================================================================

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from attacks.poisoning import blind_model_shuffle
from dp.noise_add import clipping


def _candidate_seed(round_idx, i):
    """候选 i 的独立 seed。

    与原始 MSA 的 seed (round_idx + 2026) 永不冲突:
    (round_idx + 1) * 10000 + i == round_idx + 2026 在 round_idx >= 0,
    i in [1, 9999] 时无解 (左边 >= 10000, 右边 < 10000 + 2026 恒不成立)。
    """
    return (round_idx + 1) * 10000 + i


def _shuffle_with_rng(w, round_idx, shuffle_ratio, rng):
    """忠实复制 attacks/poisoning.py::blind_model_shuffle (L11-L88)。

    唯一差异: 全局 random.* 替换为局部 rng.*。
    同一 seed 下与原始实现位级一致 (random.Random 与全局 random
    均为 MT19937, 相同 int seed 产生相同序列)。
    仅用于 rebuttal 候选生成, 原始函数不被修改。
    """
    w_att = copy.deepcopy(w)

    groups = []
    if 'layer_input.weight' in w_att:
        # 1. 结构化数据 MLP_Loan
        groups.append(('layer_input.weight', 'layer_input.bias', 'layer_hidden.weight'))
        groups.append(('layer_hidden.weight', 'layer_hidden.bias', 'layer_out.weight'))
    elif 'conv1.weight' in w_att and 'fc1.weight' in w_att and 'layer1.0.conv1.weight' not in w_att:
        # 2. 图像通用 CNN (CNNCifar / CNNEmnist)
        groups.append(('conv1.weight', 'conv1.bias', 'conv2.weight'))
        groups.append(('fc1.weight', 'fc1.bias', 'fc2.weight'))
    elif 'features.0.weight' in w_att:
        # 3. AlexNet
        groups.append(('features.0.weight', 'features.0.bias', 'features.3.weight'))
        groups.append(('classifier.1.weight', 'classifier.1.bias', 'classifier.4.weight'))
    elif 'conv1.weight' in w_att and 'layer1.0.conv1.weight' in w_att:
        # 4. ResNet18
        groups.append(('layer4.1.conv2.weight', None, 'fc.weight'))

    for (w1_name, b1_name, w2_name) in groups:
        if w1_name not in w_att or w2_name not in w_att:
            continue

        dim = w_att[w1_name].shape[0]

        len_shuffle = int(shuffle_ratio * dim)
        idx = rng.sample(range(0, len_shuffle), len_shuffle) + list(range(len_shuffle, dim))
        scale = [rng.uniform(1, 2) if (i + round_idx) % 2 == 0 else rng.uniform(0.5, 1) for i in range(dim)]

        for i in range(dim):
            w_att[w1_name][i] = w_att[w1_name][i] * scale[i]
            if b1_name is not None and b1_name in w_att and w_att[b1_name] is not None:
                w_att[b1_name][i] = w_att[b1_name][i] * scale[i]

        for k in range(w_att[w2_name].shape[0]):
            for i in range(dim):
                w_att[w2_name][k][i] = w_att[w2_name][k][i] / scale[i]

        w_att[w1_name] = w_att[w1_name][idx]
        if b1_name is not None and b1_name in w_att and w_att[b1_name] is not None:
            w_att[b1_name] = w_att[b1_name][idx]

        for i in range(w_att[w2_name].shape[0]):
            w_att[w2_name][i] = w_att[w2_name][i][idx]

    for key in w_att.keys():
        if w_att[key].is_floating_point():
            w_att[key] = torch.clamp(w_att[key], -0.99, 0.99)

    return w_att


def _mirror_lsa(w_scored, anchor_ref):
    """surrogate LSA: 严格镜像 aggreagation_method/gm_raf.py 的评分逻辑。

    - 逐层 (所有含 'weight' 的键) 分别 flatten 后计算 cosine;
    - zero-norm 保护: client 与 anchor 任一方 norm <= 1e-8 时 sim = 0.0;
    - cosine eps = 1e-8;
    - score = mean_sim - 1.5 * np.std(layer_sims), ddof=0 (总体标准差);
    - 不包含 bias, 不拼接所有层, 不使用 threshold。
    """
    layer_keys = [k for k in w_scored.keys() if 'weight' in k]
    layer_sims = []
    for k in layer_keys:
        vec_client = w_scored[k].float().view(-1)
        vec_anchor = anchor_ref[k].view(-1)
        if torch.norm(vec_client) > 1e-8 and torch.norm(vec_anchor) > 1e-8:
            sim = F.cosine_similarity(vec_client.unsqueeze(0), vec_anchor.unsqueeze(0), eps=1e-8).item()
        else:
            sim = 0.0
        layer_sims.append(sim)
    mean_sim = np.mean(layer_sims)
    std_sim = np.std(layer_sims)  # ddof=0, 与服务器 gm_raf.py 一致
    return mean_sim - 1.5 * std_sim


def adaptive_model_shuffle(w, w_glob_prev, round_idx, anchor_ref, args,
                           num_candidates=20):
    """Defense-aware Adaptive-MSA (rebuttal only)。

    每轮生成 num_candidates 个 MSA candidate:
      - candidate 0: 直接调用原始 blind_model_shuffle(w, round_idx);
      - candidate 1..N-1: _shuffle_with_rng 复制版 + 互异 seed;
      - 所有 candidate 保持 shuffle_ratio=1.0 / scale in [0.5, 2] /
        相同 attacked layer pairs / clamp [-0.99, 0.99] / 相同变换族。

    对每个 candidate 计算 post-clip / pre-noise 的 surrogate LSA:
      delta_w = cand - w_glob_prev
      delta_w, _ = clipping(args, delta_w)          # 复用原函数
      w_scored = w_glob_prev + delta_w
      score = _mirror_lsa(w_scored, anchor_ref)

    返回:
      (selected_candidate, scores, best_idx)
      - selected_candidate: argmax(score) 对应的 raw pre-clip candidate;
      - scores: 每个 candidate 的 surrogate LSA score (scores[0] 即原始 MSA);
      - best_idx: argmax 下标。
    """
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")

    candidates = []
    candidates.append(blind_model_shuffle(w, round_idx))  # candidate 0 = 原始 MSA
    for i in range(1, num_candidates):
        candidates.append(
            _shuffle_with_rng(w, round_idx, 1.0,
                              random.Random(_candidate_seed(round_idx, i))))

    if anchor_ref is None:
        anchor_ref = w_glob_prev  # Round 0: 服务器 anchor 尚未生成, fallback 到本轮全局模型

    scores = []
    best_idx = 0
    best_score = -float('inf')
    for i, cand in enumerate(candidates):
        delta_w = {k: cand[k] - w_glob_prev[k] for k in cand.keys()}
        delta_w, _ = clipping(args, delta_w)
        w_scored = {}
        for k in cand.keys():
            w_scored[k] = w_glob_prev[k] + delta_w[k]
        score = _mirror_lsa(w_scored, anchor_ref)
        scores.append(float(score))
        if score > best_score:
            best_score = score
            best_idx = i

    return candidates[best_idx], scores, best_idx
