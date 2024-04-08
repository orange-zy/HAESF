import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch.nn.parameter import Parameter


# 求不同元路径的语义注意力
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        # nn.Sequential以一种简单的方式顺序组合多个神经网络层，从而形成一个神经网络模型
        self.project = nn.Sequential(
            # nn.Linear创建一个线性变换层（全连接层）
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


# Node_Attention --> Semantic_Attention --> 两层注意力
class HANLayer(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix，几条元路径就有几个gat_layer
        self.gat_layers = nn.ModuleList()

        for i in range(num_meta_paths):
            self.gat_layers.append(
                # 节点注意力,基于注意力的图卷积网络
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))

        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAESF(nn.Module):
    def __init__(
        self, num_meta_paths, node_size, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAESF, self).__init__()

        self.out_size = out_size
        self.in_size = in_size
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )

        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

        self.cluster_layer1 = Parameter(torch.Tensor(node_size, hidden_size*8))
        torch.nn.init.xavier_normal_(self.cluster_layer1.data)
        self.cluster_layer2 = Parameter(torch.Tensor(hidden_size*8, out_size))
        torch.nn.init.xavier_normal_(self.cluster_layer2.data)
        self.cluster_layer3 = Parameter(torch.Tensor(out_size, in_size))
        torch.nn.init.xavier_normal_(self.cluster_layer3.data)

    def get_semantic(self, feature):
        W1 = self.cluster_layer1
        W2 = self.cluster_layer2
        S = self.cluster_layer3

        W1 = torch.multiply(W1, torch.div(feature @ S.T @ W2.T, W1 @ W2 @ S @ S.T @ W2.T))
        W2 = torch.multiply(W2, torch.div(W1.T @ feature @ S.T, W1.T @ W1 @ W2 @ S @ S.T))
        S = torch.multiply(S, torch.div(W2.T @ W1.T @ feature, W2.T @ W1.T @ W1 @ W2 @ S))
        return S

    def forward(self, g, h):
        V = self.get_semantic(h)

        # self.layers 有一个HANLayer,HANLayer有3个gat_layer(元路径数)
        for gnn in self.layers:
            h = gnn(g, h)
        emb = self.predict(h)
        adj_pred = self.decoder(emb)

        return h, emb, adj_pred, V

    def decoder(self, Z):
        adj_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return adj_pred
