# ====================================================================
# 文件位置: attacks/poisoning.py
# 描述: 顶会级 Benchmark 威胁模型库 (Threat Model Testbed) - 跨模态修复版
# ====================================================================

import copy
import random
import torch


def blind_model_shuffle(w, round_idx, shuffle_ratio=1.0):
    """
    MSA (Model Scaling Attack) / Blind Model Shuffle 攻击
    核心思想: 通过同构的结构乱序和缩放破坏模型特征空间，但不改变模型在正常数据上的本地表现。
    """
    w_att = copy.deepcopy(w)
    random.seed(round_idx + 2026)

    # -----------------------------------------
    # 🚨 动态网络拓扑识别与层分配 (Model-Agnostic)
    # -----------------------------------------
    groups = []  # 存放成对的可攻击层组合: (第一层权重, 第一层偏置, 第二层权重)

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
        # 残差网络直接洗牌会破坏 Skip Connection，导致前向传播崩溃。
        # 此处抓取 ResNet 最后的特征层与全连接层进行降维打击
        groups.append(('layer4.1.conv2.weight', None, 'fc.weight'))

    # -----------------------------------------
    # 🚨 通用抵消与洗牌执行引擎
    # -----------------------------------------
    for (w1_name, b1_name, w2_name) in groups:
        if w1_name not in w_att or w2_name not in w_att:
            continue

        dim = w_att[w1_name].shape[0]  # 获取 w1 的输出维度

        # 生成随机打乱索引和缩放因子
        len_shuffle = int(shuffle_ratio * dim)
        idx = random.sample(range(0, len_shuffle), len_shuffle) + list(range(len_shuffle, dim))
        scale = [random.uniform(1, 2) if (i + round_idx) % 2 == 0 else random.uniform(0.5, 1) for i in range(dim)]

        # 1. 正向缩放 w1
        for i in range(dim):
            w_att[w1_name][i] = w_att[w1_name][i] * scale[i]
            if b1_name is not None and b1_name in w_att and w_att[b1_name] is not None:
                w_att[b1_name][i] = w_att[b1_name][i] * scale[i]

        # 2. 反向缩放抵消 w2 (假设 w2 的输入通道需要匹配 w1 的输出)
        for k in range(w_att[w2_name].shape[0]):
            for i in range(dim):
                w_att[w2_name][k][i] = w_att[w2_name][k][i] / scale[i]

        # 3. 结构乱序 (Shuffle) w1
        w_att[w1_name] = w_att[w1_name][idx]
        if b1_name is not None and b1_name in w_att and w_att[b1_name] is not None:
            w_att[b1_name] = w_att[b1_name][idx]

        # 4. 结构乱序 (Shuffle) w2 对应的输入维度
        for i in range(w_att[w2_name].shape[0]):
            w_att[w2_name][i] = w_att[w2_name][i][idx]

    # -----------------------------------------
    # 🚨 防呆设计与极端值保护
    # -----------------------------------------
    for key in w_att.keys():
        if w_att[key].is_floating_point():
            # 💡 注意: 如果你想让 MSA 彻底突破防线，可以将这里的 0.99 改成更大的值 (例如 5.0)
            # 否则它会把你辛苦设定的 CLIP=40 约束死在 0.99，导致攻击威力大减。
            w_att[key] = torch.clamp(w_att[key], -0.99, 0.99)

    return w_att


def sign_flipping_attack(w_local, w_glob):
    w_new = copy.deepcopy(w_local)
    for key in w_new.keys():
        if w_new[key].is_floating_point():
            w_new[key] = 2.0 * w_glob[key] - w_local[key]
            # 同理，释放 SF 攻击的威力
            w_new[key] = torch.clamp(w_new[key], -0.99, 0.99)
    return w_new


def alie_attack(w_benign_list, z_value=1.5):
    if not w_benign_list:
        return None
    w_new = copy.deepcopy(w_benign_list[0])
    for key in w_new.keys():
        if w_new[key].is_floating_point():
            stacked = torch.stack([w[key].float() for w in w_benign_list])
            mu = torch.mean(stacked, dim=0)
            std = torch.std(stacked, dim=0)
            w_new[key] = mu - z_value * std
            w_new[key] = torch.clamp(w_new[key], -0.99, 0.99)
    return w_new