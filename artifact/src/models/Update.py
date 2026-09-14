import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, id):
        net.train()

        # ==========================================
        # 🚨 隔离点 D：针对不同数据集的优化器隔离
        # ==========================================
        if self.args.dataset == 'loan':
            # Loan 专属：在 LDP 强加噪环境下，必须使用最纯净的优化器！
            # 剥离所有 momentum 和 weight_decay，防止模型在噪声中反复横跳和萎缩。
            if self.args.client_optimizer == "sgd":
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                            lr=self.args.lr, momentum=0.0, weight_decay=0.0)
            else:
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                             lr=self.args.lr, weight_decay=0.0)
        else:
            # CIFAR / FEMNIST 原逻辑：它们需要动量和权重衰减来保证 CNN 的收敛
            if self.args.client_optimizer == "sgd":
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                            lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
            else:
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                             lr=self.args.lr, weight_decay=1e-4)
        # ==========================================

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()

                # 这里的 log_probs 在 CIFAR 里是 log对数概率，在 Loan 里是原始 Logits
                log_probs = net(images)

                # ==========================================
                # 🚨 隔离点 C：针对不同数据集的 Loss 隔离
                # ==========================================
                if self.args.dataset == 'loan':
                    # Loan 专属：因为 MLP_Loan 输出的是原始 Logits，必须用 cross_entropy (内部自带 softmax)
                    # 同时挂载我们刚刚计算出的 80/20 类别权重，狠狠惩罚模型对违约类的忽视
                    if hasattr(self.args, 'class_weights'):
                        loss = F.cross_entropy(log_probs, labels, weight=self.args.class_weights.to(self.args.device))
                    else:
                        loss = F.cross_entropy(log_probs, labels)
                else:
                    # CIFAR / FEMNIST 原逻辑：它们原版的网络末端带有 log_softmax，必须继续用 nll_loss
                    loss = F.nll_loss(log_probs, labels)
                # ==========================================

                loss.backward()

                # 🚨 100% 对齐原版：找回丢失的“本地梯度防爆阀”！(防止在 Clip=40 下爆炸)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # 为了减少刷屏，这里保持你原来的 logging，或者可以根据需要去掉
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                id, iter, sum(epoch_loss) / len(epoch_loss)))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
