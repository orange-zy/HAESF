import torch
from utils import EarlyStopping, load_data
from evaluate import metric_sets
import numpy as np
from sklearn import metrics
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 消除FutureWarning
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def main(args):
    # result storage
    path = Path(args["log_dir"])
    writer = SummaryWriter(path)

    para1 = args["α"]
    para2 = args["β"]

    (
        gs,
        features,
        labels,
        num_classes,
        adj,
    ) = load_data(args["dataset"])

    features = features.to(args["device"])
    adj = adj.to(args["device"])
    # labels = labels.to(args["device"])

    from model import HAESF
    model = HAESF(
        num_meta_paths=len(gs),
        node_size=features.shape[0],
        in_size=features.shape[1],
        hidden_size=args["hidden_units"],
        out_size=num_classes,
        num_heads=args["num_heads"],
        dropout=args["dropout"],
    ).to(args["device"])

    g = [graph.to(args["device"]) for graph in gs]

    # 根据情况灵活选择NMF变体
    # single-layer NMF
    # U = torch.rand(features.shape[0], num_classes)
    # V = torch.Tensor(np.random.random([num_classes, features.shape[1]]))

    # double-layer NMF
    W1 = torch.rand(features.shape[0], args["hidden_units"]*8)
    W2 = torch.rand(args["hidden_units"]*8, num_classes)
    S = torch.rand(num_classes, features.shape[1])

    # # # NMF layer
    # model.cluster_layer1.data = U.to(args['device'])
    # model.cluster_layer2.data = V.to(args['device'])
    model.cluster_layer1.data = W1.to(args['device'])
    model.cluster_layer2.data = W2.to(args['device'])
    model.cluster_layer3.data = S.to(args['device'])

    # best_loss
    stopper = EarlyStopping(patience=args["patience"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    print("Start training")
    best_nmf_loss = 188.0
    for epoch in tqdm(range(args["num_epochs"])):
        model.train()
        hid, embedding, adj_pred, V = model(g, features)

        nmf_loss = torch.sqrt(torch.norm(features - model.cluster_layer1 @  model.cluster_layer2 @ V))
        unity_loss = torch.sqrt(torch.norm(features - embedding @ V))
        re_loss = torch.nn.functional.binary_cross_entropy(adj_pred, adj)
        print(nmf_loss, unity_loss, re_loss)
        # loss = 10 * re_loss
        if nmf_loss < best_nmf_loss:
            best_nmf_loss = nmf_loss
            loss = para1 * re_loss + para2 * unity_loss
        else:
            loss = para1 * re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, indices = torch.max(embedding, dim=1)
        prediction = indices.long().cpu().numpy()
        NMI = metrics.normalized_mutual_info_score(prediction, labels)

        early_stop = stopper.step(loss.data.item(), NMI, model, epoch)

        # print("Epoch {:d} | Train Loss {:.4f} | NMI {}".format(epoch + 1, loss.item(), NMI,))
        with open(path / 'result.txt', 'a') as f:
            f.write("Epoch {:d} | Train Loss {:.4f} | NMI {}".format(epoch + 1, loss.item(), NMI,))
            f.write(f'\n')
        if early_stop:
            break

    result = metric_sets(model, g, num_classes, features, labels.tolist())

    print("Final result_Kmeans :NMI {} | Ari {:.4f}| Acc {:.4f} | F1_macro {:.4f}".format(
        result[0], result[1], result[2], result[3]))

    with open(path / 'result.txt', 'a') as f:
        f.write("Final result_Kmeans :NMI {} | Ari {:.4f}|Acc {:.4f}| F1_macro {:.4f}".format(
        result[0], result[1], result[2], result[3]))
        f.write(f'\n')

    stopper.load_checkpoint(model)


if __name__ == "__main__":
    import argparse
    from utils import setup

    parser = argparse.ArgumentParser("HAE-SF")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )

    # The default configuration
    default_configure = {
        "lr": 0.005,
        "num_heads": [8],
        "hidden_units": 8,
        "dropout": 0.6,
        "weight_decay": 0.001,
        "num_epochs": 400,
        "patience": 100,
        "seed": 6,
        "α": 50,
        "β": 0.25
    }

    args = parser.parse_args().__dict__
    args.update(default_configure)
    args = setup(args)

    main(args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()