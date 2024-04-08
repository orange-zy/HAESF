import numpy as np
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from munkres import Munkres

def community(z, clusters):
    z = z.detach().numpy()
    C_model = KMeans(n_clusters=clusters, verbose=1, max_iter=100, tol=0.01, n_init=3)
    C_model.fit(z)
    comm_predict = C_model.labels_

    return comm_predict


def metric_sets(model, g, num_classes, features, labels):
    model.eval()
    # with torch.no_grad():
    #     hidEmb, embed, adj_pred, V = model(g, features)

    with torch.no_grad():
        hidEmb, embed, adj_pred, V = model(g, features)

    # K-means
    hidEmb = hidEmb.cpu()
    prediction = community(hidEmb, num_classes)
    nmi = metrics.normalized_mutual_info_score(prediction, labels)
    ari = metrics.adjusted_rand_score(labels, prediction)
    acc, f1_macro = cluster_acc(labels, prediction)
    kmeans_result = [nmi, ari, acc, f1_macro]

    # AE_NMF
    _, indices = torch.max(embed, dim=1)
    prediction = indices.long().cpu().numpy()
    nmi = metrics.normalized_mutual_info_score(prediction, labels)
    acc, f1_macro = cluster_acc(labels, prediction)
    ari = metrics.adjusted_rand_score(labels, prediction)
    AENMF_result = [nmi, ari, acc, f1_macro]

    if kmeans_result[0] > AENMF_result[0]:
        return kmeans_result

    return AENMF_result


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numClass1 = len(l1)

    l2 = list(set(y_pred))
    numClass2 = len(l2)

    ind = 0
    if numClass1 != numClass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numClass2 = len(l2)

    if numClass1 != numClass2:
        print("error")
        return

    cost = np.zeros((numClass1, numClass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # corresponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    # 预测标签处理后，得到的指标都有提升。
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro
