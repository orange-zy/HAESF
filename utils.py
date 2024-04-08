import datetime
import os
import random
import numpy as np
import torch
from scipy import io as sio
import dgl
import scipy.sparse as sp


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# set the result storage path
def setup_log_dir(args):
    dt = datetime.datetime.now()
    date_postfix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )
    log_dir = os.path.join(
        args["log_dir"], "{}_{}".format(args["dataset"], date_postfix)
    )

    os.makedirs(log_dir)
    print("Created directory {}".format(log_dir))

    return log_dir


def setup(args):
    set_random_seed(args["seed"])
    args["dataset"] = "DBLP"
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    args["log_dir"] = setup_log_dir(args)
    if args["dataset"] == "ACM":
        args["patience"] = 10
        args["α"] = 5
    elif args["dataset"] == "IMDB":
        args["patience"] = 100
        args["α"] = 40

    return args


# load data.
def load_data(dataset):
    if dataset == "ACM":
        return load_acm()
    elif dataset == "DBLP":
        return load_dblp()
    elif dataset == "IMDB":
        return load_imdb()
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))


def load_dblp():
    data_path = 'data/DBLP.mat'
    data = sio.loadmat(data_path)
    labels, features = (
        # numpy矩阵 -> tensor -> tensor数据取整/浮点
        torch.from_numpy(data["label"]).long(),
        torch.from_numpy(data["features"]).float(),
    )
    num_classes = labels.shape[1]
    # torch.nonzero，取tensor矩阵中的每行非零元素的位置索引，得到节点的标签.
    labels = labels.nonzero()[:, 1]
    # 按一定比例构建邻接矩阵，目前最好的比例
    adj = 0.8 * data['net_APA'] + 0.35 * data['net_APTPA'] + 0.25 * data['net_APCPA']
    adj = torch.Tensor(adj)
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))

    apa_g = dgl.from_scipy(sp.csr_matrix(data['net_APA']))
    apcpa_g = dgl.from_scipy(sp.csr_matrix(data['net_APCPA']))
    aptpa_g = dgl.from_scipy(sp.csr_matrix(data['net_APTPA']))
    gs = [apa_g, apcpa_g, aptpa_g]

    return (
        gs,
        features,
        labels,
        num_classes,
        adj,
    )


def load_acm():
    data_path = 'data/ACM.mat'
    data = sio.loadmat(data_path)

    labels, features = (
        # numpy矩阵 -> tensor -> tensor数据取整/浮点
        torch.from_numpy(data["label"]).long(),
        torch.from_numpy(data["feature"]).float(),
    )

    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    adj = data['PAP'] + data['PLP']
    # net_PTP 起到 negative 的作用
    # adj = 5 * data['PAP'] + 4 * data['PLP'] + 1 * data['PTP']
    adj = torch.Tensor(adj)
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))

    # Adjacency matrices for meta path based neighbors
    author_g = dgl.from_scipy(sp.csr_matrix(data["PAP"]))
    subject_g = dgl.from_scipy(sp.csr_matrix(data["PLP"]))
    # other_g = dgl.from_scipy(sp.csr_matrix(data['PTP']))
    gs = [author_g, subject_g]

    return (
        gs,
        features,
        labels,
        num_classes,
        adj,
    )


def load_imdb():
    data_path = 'data/IMDBNew.mat'
    data = sio.loadmat(data_path)

    labels, features = (
        # numpy矩阵 -> tensor -> tensor数据取整/浮点
        torch.from_numpy(data["labelCopy"]).long(),
        torch.from_numpy(data["featureCopy"]).float(),
    )

    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    adj = 5 * data['MAMCopy'] + 2 * data['MDMCopy'] + 2 * data['MYMCopy']
    adj = torch.Tensor(adj)
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))

    # Adjacency matrices for meta path based neighbors
    mam_g = dgl.from_scipy(sp.csr_matrix(data["MAMCopy"]))
    mdm_g = dgl.from_scipy(sp.csr_matrix(data["MDMCopy"]))
    mym_g = dgl.from_scipy(sp.csr_matrix(data['MYMCopy']))
    gs = [mam_g, mdm_g, mym_g]

    return (
        gs,
        features,
        labels,
        num_classes,
        adj,
    )


# Early stop,防止过拟合,在NMI最好时停止！
class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = patience
        self.counter = 0
        self.best_nmi = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, nmi, model, epoch):
        if self.best_loss is None:
            self.best_nmi = nmi
            self.best_loss = loss
            self.save_checkpoint(model)

        elif (nmi < self.best_nmi) and (epoch > 200):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (nmi >= self.best_nmi):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_nmi = np.max((nmi, self.best_nmi))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
