# -*- coding: utf-8 -*-
# ====================================================================
# 文件: main_adaptive_msa.py   (NEW FILE - rebuttal only)
# Adaptive-MSA rebuttal 专用实验入口。
#
# 来源: 完整复制自 main.py (2026-08-23, 无 git, 以 gmsa_baseline_backup 备份为准)。
# 新增内容 (其余与 main.py 保持一致):
#   1) 自带 args_parser (复制自 utils/options_cifar.py) + --msa_num_candidates
#   2) 攻击分支 custom_attack == 'adaptive_msa'
#   3) 启动期 candidate-0 等价性守卫 (_assert_adaptive_equivalence)
#   4) 记录 Original_LSA / Best_LSA / LSA_Gain
#
# 注意: 本文件刻意不修改 main.py。若 main.py 后续演进, 本副本需人工同步。
# ====================================================================

import copy
import logging
import random
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# ================= 1. 严格引入原生模块 =================
from aggreagation_method.fed_aggregation import aggregation
from dp.noise_add import clipping, noise_add
from dp.privacy import privacy_account
from models.Update import LocalUpdate
from models.test import test_img
from utils.sampling import cifar, cifar_dirichlet
from models.Nets import CNNCifar, CNNEmnist, MLP_Loan
from models.resnet import ResNet18
from models.load_loan import get_loan_dataset
# ================= 2. 引入防御与攻击模块 =================
from aggreagation_method.gm_raf import GM_RAF
from aggreagation_method.fltrust import FLTrust
from aggreagation_method.flame import FLAME
from attacks.poisoning import blind_model_shuffle, sign_flipping_attack, alie_attack
from attacks.poisoning_hismsa import hismsa_attack
# ================= 2b. Adaptive-MSA (rebuttal only) =================
from attacks.poisoning_adaptive_msa import adaptive_model_shuffle, _shuffle_with_rng

# 引入 DP-Poison 专属模块 (请确保你已经创建了 attacks/dp_poison.py)
from attacks.dp_poison import LocalUpdateDPPoison_PGD, test_backdoor


# ================= 3. 参数解析 (复制自 utils/options_cifar.py, 不修改原文件) =================
def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=7, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--client_optimizer', type=str, default='adam', help='SGD with momentum; adam')
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--aggregation_methods', type=str, default="fedavg",
                        help='fedavg/krum/trimmed_mean')
    parser.add_argument('--lr_decay', type=float, default=0.99)

    # attack
    parser.add_argument('--comprised_rate', type=float, default=0.2)
    parser.add_argument('--attack_type', type=str, default="MSA",
                        help='MSA,gaussian_attack,no_attack')
    parser.add_argument('--shuffle_ratio', type=float, default=1,
                        help='0.25,0.5,0.75,1')

    # dp
    parser.add_argument('--use_dp', type=bool, default=True,
                        help='True/False')
    parser.add_argument('--clipthr', type=int, default=40,
                        help='threshold for parameter pruning')
    parser.add_argument('--privacy_budget', type=float, default=40)
    parser.add_argument('--attackers_privacy_budget', type=float, default=100)
    parser.add_argument('--delta', type=float, default=0.01)

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='None', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    # GM-RAF 专属控制参数
    parser.add_argument('--custom_attack', type=str, default='none', help="攻击策略: none, msa, lf, alie, hismsa, adaptive_msa")
    parser.add_argument('--defense', type=str, default='fedavg', help="防御策略: fedavg, gm_raf")
    parser.add_argument('--rate', type=float, default=0.0, help="恶意节点比例 (0.0 - 1.0)")
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet Non-IID 分布浓度参数')
    parser.add_argument('--var', type=int, default=0, help='消融实验变体编号: 0(Full), 1(w/o LSA), 2(w/o MOM), 3(w/o PCD)')
    parser.add_argument('--no_mom', action='store_true', help='彻底关闭全局动量 (专供消融实验)')
    parser.add_argument('--exp_name', type=str, default='test_exp', help='本次实验的名称标识(用于保存log和csv文件)')

    # Adaptive-MSA (rebuttal only)
    parser.add_argument('--msa_num_candidates', type=int, default=20,
                        help='Adaptive-MSA: 每轮生成的 MSA candidate 数量 (默认 20)')

    args = parser.parse_args()
    return args


def _assert_adaptive_equivalence(state, round_idx):
    """启动期守卫: 复制版在 seed=round_idx+2026 下必须与原始 blind_model_shuffle
    位级一致 (candidate 0 等价性保护)。不一致则立即 STOP, 拒绝开始训练。"""
    a = blind_model_shuffle(state, round_idx)
    b = _shuffle_with_rng(state, round_idx, 1.0, random.Random(round_idx + 2026))
    if set(a.keys()) != set(b.keys()):
        raise RuntimeError(
            "Adaptive-MSA equivalence guard FAILED (key sets differ). Refusing to run.")
    for k in a.keys():
        if not torch.equal(a[k], b[k]):
            raise RuntimeError(
                f"Adaptive-MSA equivalence guard FAILED at key={k}. Refusing to run.")


# ================= 4. 原版 LEAF 数据读取 =================
def read_leaf_dir(data_dir):
    data_dict = {}
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"🚨 未找到目录: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    for f in files:
        with open(os.path.join(data_dir, f), 'r') as inf:
            cdata = json.load(inf)
        data_dict.update(cdata['user_data'])
    return data_dict


class FEMNISTDataset(Dataset):
    def __init__(self, data_dict):
        self.features = []
        self.labels = []
        for u, d in data_dict.items():
            self.features.extend(d['x'])
            self.labels.extend(d['y'])
        self.features = torch.tensor(self.features, dtype=torch.float32).view(-1, 1, 28, 28)

        # ✅ 绝杀修复 2：加上原版 FEMNIST/MNIST 的标准化，拯救收敛率
        self.features = (self.features - 0.1307) / 0.3081

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.features)

    def __getitem__(self, idx): return self.features[idx], self.labels[idx]


# ================= 5. 主程序 =================
def main():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 5.1 数据加载与精准划分
    if args.dataset == 'cifar':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=False, transform=trans)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=False, transform=trans)
        args.num_classes = 10
        if args.split == 'user':
            dict_users = cifar_dirichlet(dataset_train, args.num_users, alpha=args.alpha)
        else:
            dict_users = cifar(dataset_train, args.num_users)

    elif args.dataset == 'femnist':
        train_data = read_leaf_dir('./leaf/data/femnist/data/train')
        test_data = read_leaf_dir('./leaf/data/femnist/data/test')
        dataset_train = FEMNISTDataset(train_data)
        dataset_test = FEMNISTDataset(test_data)
        args.num_classes = 62

        dict_users = {}
        current_idx = 0
        for i, (u, d) in enumerate(train_data.items()):
            if i >= args.num_users: break
            l = len(d['x'])
            dict_users[i] = set(range(current_idx, current_idx + l))
            current_idx += l
    elif args.dataset == 'loan':
        # 1. 加载清洗好的 TensorDataset 和特征维度
        dataset_train_full, feature_dim, args.class_weights = get_loan_dataset(
            './data/loan/accepted_2007_to_2018Q4.csv')
        args.feature_dim = feature_dim  # 🚨 存入 args，一会给 MLP 用

        # 2. 划分 80% 训练集, 20% 测试集
        train_size = int(0.8 * len(dataset_train_full))
        test_size = len(dataset_train_full) - train_size
        dataset_train, dataset_test = torch.utils.data.random_split(dataset_train_full, [train_size, test_size])

        args.num_classes = 2  # 坏账(1) vs 已结清(0)

        # 3. IID 均匀分发给客户端
        dict_users = cifar(dataset_train, args.num_users)

    else:
        raise ValueError("Dataset unknown")

    # ==========================================
    # 兼容 Label Flipping (LF 攻击) 的标签反转逻辑
    # ==========================================
    root_dataset = data.Subset(dataset_train, list(range(100)))

    poisoned_train_set = copy.deepcopy(dataset_train)
    actual_ds = poisoned_train_set.dataset if hasattr(poisoned_train_set, 'dataset') else poisoned_train_set

    if hasattr(actual_ds, 'targets'):
        if isinstance(actual_ds.targets, torch.Tensor):
            actual_ds.targets = (args.num_classes - 1) - actual_ds.targets
        else:
            actual_ds.targets = [(args.num_classes - 1) - t for t in actual_ds.targets]
    elif hasattr(actual_ds, 'labels'):
        actual_ds.labels = (args.num_classes - 1) - actual_ds.labels
    elif hasattr(actual_ds, 'tensors'):
        # 🚨 专门针对 Loan (TensorDataset) 的 LF 攻击兼容
        new_labels = (args.num_classes - 1) - actual_ds.tensors[1]
        actual_ds.tensors = (actual_ds.tensors[0], new_labels)

    # ==========================================
    # 挂载对应的数据集物理引擎 (CNN / ResNet / MLP)
    # ==========================================
    if args.dataset == 'cifar':
        if getattr(args, 'model', 'cnn') == 'resnet18':
            net_glob = ResNet18(num_classes=args.num_classes).to(args.device)
        else:
            net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'femnist':
        net_glob = CNNEmnist(args=args).to(args.device)
    elif args.dataset == 'loan':
        # 🚨 挂载专属的 MLP 引擎
        net_glob = MLP_Loan(dim_in=args.feature_dim, dim_out=args.num_classes).to(args.device)

    net_glob.train()

    # ==========================================
    # Adaptive-MSA 启动期等价性守卫 (rebuttal only)
    # ==========================================
    if args.custom_attack == 'adaptive_msa':
        _assert_adaptive_equivalence(net_glob.state_dict(), 0)
        print(f"[Adaptive-MSA] candidate-0 equivalence guard PASSED | num_candidates={args.msa_num_candidates}")

    defender = None
    m_clients = max(int(args.frac * args.num_users), 1)
    if args.defense == 'gm_raf':
        defender = GM_RAF(num_clients=m_clients, device=args.device, var=args.var, no_mom=args.no_mom)
    elif args.defense == 'fltrust':
        defender = FLTrust(root_dataset=root_dataset, args=args)
    elif args.defense == 'flame':
        defender = FLAME(num_clients=m_clients)

    global_attacker_pool = set(range(int(args.num_users * args.rate)))

    rec_acc, rec_loss, rec_tpr, rec_fpr, rec_rej = [], [], [], [], []
    rec_scores = []
    rec_asr = []
    # Adaptive-MSA 记录 (rebuttal only)
    rec_adaptive_orig, rec_adaptive_best, rec_adaptive_gain = [], [], []

    # 5.3 联邦主循环
    hismsa_history_distances = []
    hismsa_prev_global = None
    for round_idx in range(args.epochs):
        w_glob_prev = copy.deepcopy(net_glob.state_dict())
        w_locals, loss_locals, round_attacker_idxs = [], [], []
        idxs_users = np.random.choice(range(args.num_users), m_clients, replace=False)
        round_adaptive_orig, round_adaptive_best = [], []

        noise_scale_benign, noise_scale_malic = 0.0, 0.0

        if getattr(args, 'use_dp', True):
            noise_scale_benign = privacy_account(args, is_attack=False, num_items_train=len(dataset_train))
            noise_scale_malic = privacy_account(args, is_attack=True, num_items_train=len(dataset_train))

        for i, idx in enumerate(idxs_users):
            is_dppoison = (args.custom_attack == 'dppoison' and idx in global_attacker_pool)

            if is_dppoison:
                local_dpp = LocalUpdateDPPoison_PGD(
                    args=args, dataset=dataset_train, idxs=dict_users[idx],
                    true_noise_std=noise_scale_malic, target_label=0, alpha=0.5
                )
                w_glob_prev = copy.deepcopy(net_glob.state_dict())
                w, loss = local_dpp.train(
                    net_glob=copy.deepcopy(net_glob).to(args.device),
                    w_glob_prev=w_glob_prev
                )
            elif args.custom_attack == 'lf' and idx in global_attacker_pool:
                local = LocalUpdate(args=args, dataset=poisoned_train_set, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), id=idx)
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), id=idx)

            if idx in global_attacker_pool:
                round_attacker_idxs.append(i)
                if args.custom_attack == 'adaptive_msa':
                    # reference = anchor_{t-1} (gm_raf 时为 defender.anchor_momentum;
                    # Round 0 或其他防御时为 None -> 模块内 fallback w_glob_prev)
                    anchor_ref = getattr(defender, 'anchor_momentum', None)
                    w, cand_scores, best_idx = adaptive_model_shuffle(
                        w, w_glob_prev, round_idx, anchor_ref, args,
                        num_candidates=args.msa_num_candidates)
                    round_adaptive_orig.append(cand_scores[0])
                    round_adaptive_best.append(float(max(cand_scores)))
                elif args.custom_attack == 'hismsa':
                    w = hismsa_attack(w, w_glob_prev, hismsa_history_distances)
                elif args.custom_attack == 'msa':
                    w = blind_model_shuffle(w, round_idx)
                elif args.custom_attack == 'sf':
                    w = sign_flipping_attack(w, w_glob_prev)

            # ==========================================================
            # ✅ 修复隔离点 A：所有数据集统一使用安全的 Delta W 裁剪
            # ==========================================================
            if getattr(args, 'use_dp', True) and not is_dppoison:
                # 1. 计算更新量 Delta W
                delta_w = {k: w[k] - w_glob_prev[k] for k in w.keys()}
                # 2. 对 Delta W 施加 DP 裁剪边界
                delta_w, _ = clipping(args, delta_w)
                # 3. 将合法的更新量加回全局模型，重构本地权重
                for k in w.keys():
                    w[k] = w_glob_prev[k] + delta_w[k]

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(loss)

        if args.custom_attack == 'alie' and round_attacker_idxs:
            benign_w = [w_locals[i] for i in range(len(w_locals)) if i not in round_attacker_idxs]
            if benign_w:
                w_alie = alie_attack(benign_w)
                for i in round_attacker_idxs: w_locals[i] = copy.deepcopy(w_alie)

        # Adaptive-MSA 逐轮统计 (rebuttal only): 本 round 全部攻击者的均值
        if args.custom_attack == 'adaptive_msa' and round_adaptive_orig:
            rec_adaptive_orig.append(float(np.mean(round_adaptive_orig)))
            rec_adaptive_best.append(float(np.mean(round_adaptive_best)))
            rec_adaptive_gain.append(rec_adaptive_best[-1] - rec_adaptive_orig[-1])
        else:
            rec_adaptive_orig.append(np.nan)
            rec_adaptive_best.append(np.nan)
            rec_adaptive_gain.append(np.nan)

        # ==========================================================
        # ✅ 修复隔离点 B：恢复优雅的动态加噪逻辑
        # ==========================================================
        if getattr(args, 'use_dp', True):
            w_locals_noised = []
            for i, w_local in enumerate(w_locals):
                # DPPoison 攻击者不加噪声
                if args.custom_attack == 'dppoison' and i in round_attacker_idxs:
                    w_locals_noised.append(w_local)
                else:
                    # 根据身份匹配正确的 Sigma（由 privacy_account 科学计算得出）
                    current_noise = noise_scale_malic if i in round_attacker_idxs else noise_scale_benign
                    # 调用原生 noise_add 对重构后的权重直接加噪
                    w_noised = noise_add(args, current_noise, [w_local])[0]
                    w_locals_noised.append(w_noised)

            w_locals = w_locals_noised

        # ==========================================================
        # 下面的聚合与测试逻辑完全不变
        # ==========================================================
        if args.defense in ['gm_raf', 'fltrust', 'flame']:
            w_glob, scores, mask = defender.run(w_locals, round_idx, w_glob_prev)
            if args.defense == 'gm_raf':
                rec_scores.append(scores)

            tp = sum([1 for i in round_attacker_idxs if mask[i] == 0])
            fp = sum([1 for i in range(m_clients) if i not in round_attacker_idxs and mask[i] == 0])
            tpr = tp / len(round_attacker_idxs) if round_attacker_idxs else 1.0
            fpr = fp / (m_clients - len(round_attacker_idxs)) if len(round_attacker_idxs) < m_clients else 0.0
            rej = m_clients - sum(mask)
            print(f"R{round_idx:02d} | {args.defense.upper()} | TPR: {tpr:.2f} | FPR: {fpr:.2f} | Rej: {rej}")
        else:
            w_glob = aggregation(args, w_locals)
            tpr, fpr, rej = 0.0, 0.0, 0
            print(f"R{round_idx:02d} | FedAvg")

        net_glob.load_state_dict(w_glob)

        # HisMSA 历史位移记录
        if hismsa_prev_global is not None:
            sq_dist = 0.0
            for k in w_glob.keys():
                if w_glob[k].is_floating_point():
                    diff_t = w_glob[k] - hismsa_prev_global[k]
                    sq_dist += torch.sum(diff_t ** 2).item()
            hismsa_history_distances.append(sq_dist ** 0.5)
        hismsa_prev_global = copy.deepcopy(w_glob)

        rec_tpr.append(tpr)
        rec_fpr.append(fpr)
        rec_rej.append(rej)

        # 测试评估
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_val = (acc_test.item() if torch.is_tensor(acc_test) else acc_test) * 100
        rec_acc.append(acc_val)
        rec_loss.append(loss_test)

        asr_val = 0.0
        if args.custom_attack == 'dppoison':
            asr_val, _ = test_backdoor(net_glob, dataset_test, args, target_label=0)
        rec_asr.append(asr_val)

        if args.custom_attack == 'dppoison':
            print(f"    | Main Acc: {acc_val:.2f}% | ASR: {asr_val:.2f}%")
        else:
            print(f"    | Acc: {acc_val:.2f}% | Loss: {loss_test:.4f}")

    # 5.4 保存结果
    os.makedirs('./results', exist_ok=True)
    df_dict = {
        'Round': range(args.epochs),
        'Acc': rec_acc,
        'Loss': rec_loss,
        'Rej': rec_rej,
        'TPR': rec_tpr,
        'FPR': rec_fpr,
        # Adaptive-MSA 专属列 (rebuttal only; 非 adaptive 运行时为 NaN)
        'Original_LSA': rec_adaptive_orig,
        'Best_LSA': rec_adaptive_best,
        'LSA_Gain': rec_adaptive_gain,
    }
    if args.custom_attack == 'dppoison':
        df_dict['ASR'] = rec_asr

    pd.DataFrame(df_dict).to_csv(f"./results/{args.exp_name}.csv", index=False)
    print(f"💾 Saved to ./results/{args.exp_name}.csv")

    if args.defense == 'gm_raf' and len(rec_scores) > 0:
        np.save(f"./results/{args.exp_name}_scores.npy", np.array(rec_scores))
        print(f"💾 Saved LSA Scores to ./results/{args.exp_name}_scores.npy")

    # Adaptive-MSA 专属记录 (rebuttal only)
    if args.custom_attack == 'adaptive_msa':
        np.save(f"./results/{args.exp_name}_adaptive_scores.npy",
                np.array([rec_adaptive_orig, rec_adaptive_best, rec_adaptive_gain]))
        print(f"💾 Saved Adaptive Scores to ./results/{args.exp_name}_adaptive_scores.npy")


if __name__ == '__main__':
    main()
