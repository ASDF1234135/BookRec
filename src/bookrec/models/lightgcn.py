"""LightGCN. n_layers=0 is plain MF (same code path)."""

import numpy as np
import torch
from torch import nn


class _SymmetricSparseMM(torch.autograd.Function):
    """adj @ x for a SYMMETRIC sparse adj.

    Autograd's default backward has to transpose the sparse matrix every step, which was
    ~25x slower here. Since adj == adj.T, the gradient w.r.t. x is simply adj @ grad.
    """

    @staticmethod
    def forward(ctx, adj, x):
        ctx.adj = adj
        return torch.sparse.mm(adj, x)

    @staticmethod
    def backward(ctx, grad):
        return None, torch.sparse.mm(ctx.adj, grad)


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def build_adj(self, train_pairs) -> torch.Tensor:
        """Symmetric-normalised sparse adjacency D^-1/2 A D^-1/2 (torch.sparse).

        train_pairs: (N, 2) array of (user_idx, item_idx). Nodes 0..U-1 are users and
        U..U+I-1 are items, so A is the bipartite graph [[0, R], [R^T, 0]].
        """
        pairs = torch.as_tensor(np.asarray(train_pairs), dtype=torch.long)
        users = pairs[:, 0]
        items = pairs[:, 1] + self.n_users  # shift items after the user block
        n = self.n_users + self.n_items

        # Both directions, so the matrix is symmetric.
        rows = torch.cat([users, items])
        cols = torch.cat([items, users])

        deg = torch.zeros(n).scatter_add_(0, rows, torch.ones(rows.numel()))
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # isolated nodes

        vals = d_inv_sqrt[rows] * d_inv_sqrt[cols]
        coo = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()
        # CSR multiplies much faster than COO on CPU.
        return coo.to_sparse_csr()

    def propagate(self, adj: torch.Tensor):
        """Average embeddings across layers 0..n_layers; return (user_emb, item_emb).

        With n_layers=0 the loop is skipped and this is plain MF.
        """
        x = self.emb.weight
        layers = [x]
        for _ in range(self.n_layers):
            x = _SymmetricSparseMM.apply(adj, x)
            layers.append(x)
        out = torch.stack(layers).mean(dim=0)
        return out[: self.n_users], out[self.n_users :]

    def bpr_loss(self, adj, users, pos, neg, l2: float) -> torch.Tensor:
        """BPR loss + L2 on the layer-0 embeddings of the batch.

        users, pos, neg are index tensors; pos/neg are item indices (0..I-1).
        """
        user_all, item_all = self.propagate(adj)
        u, p, n = user_all[users], item_all[pos], item_all[neg]

        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)
        loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

        # Regularise the raw (layer-0) embeddings of just the rows used in this batch.
        w = self.emb.weight
        u0, p0, n0 = w[users], w[pos + self.n_users], w[neg + self.n_users]
        reg = (u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()) / 2 / users.numel()
        return loss + l2 * reg
