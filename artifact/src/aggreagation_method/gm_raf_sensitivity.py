# ====================================================================
# 文件位置: aggreagation_method/gm_raf_sensitivity.py
# 描述: GM-RAF 防御引擎 (Reviewer A 超参敏感性实验副本)
#       与 aggreagation_method/gm_raf.py 行为等价; 唯一差异:
#       __init__ 新增 alpha_mom=0.2 / kappa=1.5 / mad_multiplier=3.0,
#       run() 内三处字面量 0.2 / 1.5 / 3.0 改为对应实例参数。
#       默认参数下与原版逐位一致 (见 check_gm_raf_sensitivity_equivalence.py)。
# ====================================================================

import copy
import numpy as np
import torch
import torch.nn.functional as F


class GM_RAF(object):
    """
    GM-RAF 防御引擎：包含层级子空间对齐(LSA)、动态MAD阈值、主成分去噪(PCD)及全局动量(MOM)。
    """

    def __init__(self, num_clients, device, no_mom=False, var=0,
                 alpha_mom=0.2, kappa=1.5, mad_multiplier=3.0):
        """
        :param num_clients: 参与本轮聚合的客户端数量
        :param device: 运行设备 (如 'cuda:0' 或 'cpu')
        :param no_mom: 是否关闭全局动量 (消融实验 Var 2)
        :param var: 消融变体控制参数 (0: Full GM-RAF, 1: w/o LSA, 2: Var2保留但机制调整, 3: w/o PCD)
        :param alpha_mom: 动量系数 (默认 0.2, 与原版字面量一致)
        :param kappa: LSA 方差惩罚系数 (默认 1.5, 与原版字面量一致)
        :param mad_multiplier: MAD 阈值乘数 (默认 3.0, 与原版字面量一致)
        """
        self.num_clients = num_clients
        self.device = device
        self.anchor_momentum = None
        self.no_mom = no_mom
        self.var = var
        self.alpha_mom = alpha_mom
        self.kappa = kappa
        self.mad_multiplier = mad_multiplier

    def run(self, w_locals, round_idx, w_glob_prev=None):
        """
        执行防御过滤与鲁棒聚合
        :param w_locals: 客户端上传的本地模型权重列表
        :param round_idx: 当前联邦学习的轮次
        :return: w_avg (聚合后的全局模型), instant_scores (信任分数), mask (0/1接受掩码)
        """
        if not w_locals:
            return None, [], []

        layer_keys = [k for k in w_locals[0].keys() if 'weight' in k]

        # ---------------------------------------------------------
        # 1. 锚点计算 (Anchor Calculation) 与 全局动量 (Momentum)
        # ---------------------------------------------------------
        if self.var == 2:
            w_mean_curr = {}
            for k in layer_keys:
                stack = torch.stack([w[k].float() for w in w_locals])
                w_mean_curr[k] = torch.mean(stack, dim=0)
            self.anchor_momentum = w_mean_curr
        else:
            w_median_curr = {}
            # 🚨 显存优化手术 1：抛弃全局 residuals 列表，改为逐层计算即时释放
            for k in layer_keys:
                if self.anchor_momentum is None:
                    stack = torch.stack([w[k].float() for w in w_locals])
                else:
                    stack = torch.stack([(w[k] - self.anchor_momentum[k]).float() for w in w_locals])

                median_residual = torch.median(stack, dim=0)[0]

                if self.anchor_momentum is None:
                    w_median_curr[k] = median_residual
                else:
                    w_median_curr[k] = self.anchor_momentum[k] + median_residual

                # 强制回收本层的堆叠张量，防止显存雪崩
                del stack
                torch.cuda.empty_cache()

            alpha_mom = 1.0 if self.no_mom else self.alpha_mom
            if self.anchor_momentum is None:
                self.anchor_momentum = w_median_curr
            else:
                for k in layer_keys:
                    self.anchor_momentum[k] = (1 - alpha_mom) * self.anchor_momentum[k] + alpha_mom * w_median_curr[k]

        # ---------------------------------------------------------
        # 2. 层级子空间对齐 (LSA) 与 异常检测
        # ---------------------------------------------------------
        instant_scores = []
        if self.var == 1:
            # 消融：不使用 LSA，使用全局欧式距离
            for i in range(self.num_clients):
                vec_client_all = torch.cat([w_locals[i][k].float().view(-1) for k in layer_keys])
                vec_anchor_all = torch.cat([self.anchor_momentum[k].view(-1) for k in layer_keys])
                dist = torch.norm(vec_client_all - vec_anchor_all).item()
                instant_scores.append(-dist)
        else:
            # 标准 GM-RAF：层级余弦相似度
            for i in range(self.num_clients):
                layer_sims = []
                for k in layer_keys:
                    vec_client = w_locals[i][k].float().view(-1)
                    vec_anchor = self.anchor_momentum[k].view(-1)
                    if torch.norm(vec_client) > 1e-8 and torch.norm(vec_anchor) > 1e-8:
                        sim = F.cosine_similarity(vec_client.unsqueeze(0), vec_anchor.unsqueeze(0), eps=1e-8).item()
                    else:
                        sim = 0.0
                    layer_sims.append(sim)
                mean_sim = np.mean(layer_sims)
                std_sim = np.std(layer_sims)
                lsa_score = mean_sim - self.kappa * std_sim
                instant_scores.append(lsa_score)

        instant_scores = np.array(instant_scores)

        # ---------------------------------------------------------
        # 3. 动态自适应阈值 (MAD Thresholding)
        # ---------------------------------------------------------
        med_score = np.median(instant_scores)
        mad_score = np.median(np.abs(instant_scores - med_score)) + 1e-6
        threshold = med_score - self.mad_multiplier * mad_score

        if self.var != 1:
            threshold = max(threshold, 0.05)

        selected_indices = np.where(instant_scores >= threshold)[0]
        mask = [1 if i in selected_indices else 0 for i in range(self.num_clients)]

        # 如果所有客户端都被拒绝，则返回上一轮的锚点以保持模型稳定
        # 🚨 绝杀修复：如果所有人都被拒绝，本轮作废，直接返回上一轮的全局模型
        if len(selected_indices) == 0:
            if w_glob_prev is not None:
                return copy.deepcopy(w_glob_prev), instant_scores, mask
            else:
                return self.anchor_momentum, instant_scores, mask

        # ---------------------------------------------------------
        # 4. 主成分去噪 (PCD) 与 安全聚合
        # ---------------------------------------------------------
        w_avg = copy.deepcopy(w_locals[0])

        if self.var == 3:
            # 消融：w/o PCD (直接硬平均)
            for k in w_avg.keys():
                stack = torch.stack([w_locals[idx][k].float() for idx in selected_indices])
                w_avg[k] = torch.mean(stack, dim=0)
            return w_avg, instant_scores, mask

        # 标准 GM-RAF: 执行 PCD
        for k in layer_keys:
            layer_params = torch.stack([w_locals[idx][k].float() for idx in selected_indices])
            N_acc = layer_params.shape[0]
            original_shape = layer_params.shape[1:]

            X = layer_params.view(N_acc, -1)
            if N_acc > 1:
                X_mean = torch.mean(X, dim=0, keepdim=True)
                X_centered = X - X_mean

                # 🚨 显存优化手术 2：超高维矩阵的 SVD 必须在 CPU 上进行
                X_centered_cpu = X_centered.cpu()
                U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(X_centered_cpu, full_matrices=False)

                K_components = 1
                # 只把我们需要的前 K 个主成分拷回 GPU
                U = U_cpu[:, :K_components].to(X.device)
                S = S_cpu[:K_components].to(X.device)
                Vh = Vh_cpu[:K_components, :].to(X.device)

                X_denoised = X_mean + torch.matmul(U * S.unsqueeze(0), Vh)
                avg_layer_flat = torch.mean(X_denoised, dim=0)

                # 深度清理临时变量
                del X_centered, X_centered_cpu, U_cpu, S_cpu, Vh_cpu
                torch.cuda.empty_cache()
            else:
                # 只有一个客户端被选中时，无需去噪
                avg_layer_flat = X.squeeze(0)

            w_avg[k] = avg_layer_flat.view(original_shape)

        # 非权重层 (如 BatchNorm 的 running_mean 等) 直接求平均
        non_weight_keys = [k for k in w_locals[0].keys() if k not in layer_keys]
        for k in non_weight_keys:
            stack_non_weight = torch.stack([w_locals[idx][k].float() for idx in selected_indices])
            w_avg[k] = torch.mean(stack_non_weight, dim=0)

        return w_avg, instant_scores, mask
