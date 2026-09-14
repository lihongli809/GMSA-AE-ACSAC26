import numpy as np

def cifar(dataset, num_users):
    """
    原生的均匀数据划分 (IID)
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# ================= 追加的 Dirichlet 划分逻辑 =================

def cifar_dirichlet(dataset, num_users, alpha=0.5):
    """
    基于 Dirichlet 分布的 Non-IID 数据划分 (兼容 CIFAR 和 FEMNIST)
    :param dataset: torchvision 或自定义的 dataset 对象
    :param num_users: 客户端总数
    :param alpha: Dirichlet 浓度参数 (越小越 Non-IID)
    :return: dict_users: 字典 {client_id: set(图片索引)}
    """
    min_size = 0
    min_require_size = 10  # 确保每个客户端至少有 10 张图片，防止极小样本崩溃
    N = len(dataset)
    dict_users = {}

    # 提取所有标签
    if hasattr(dataset, 'targets'):
        y_train = np.array(dataset.targets)
    else:
        # 兼容其他形式的数据集 (如 LEAF)
        y_train = np.array([y for _, y in dataset])

    # 🚨 核心修复：动态识别类别数 (CIFAR会得出10，FEMNIST会得出62)
    K = int(y_train.max() + 1)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            # 按照 Dirichlet 分布生成每个客户端分到类别 k 的比例
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))

            # 平衡策略：如果某个客户端分到的数据太多，就降低它的比例
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # 将索引按比例切分并分配给各个客户端
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        min_size = min([len(idx_j) for idx_j in idx_batch])

    # 将 list 转换为 set，与原生 cifar 函数的返回格式保持 100% 一致
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = set(idx_batch[j])

    return dict_users