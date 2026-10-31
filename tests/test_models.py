import numpy as np
import pytest
import torch

from bookrec.models.lightgcn import LightGCN
from bookrec.models.popularity import Popularity
from bookrec.train import sample_negatives


def test_popularity_counts_and_score_shape():
    pop = Popularity().fit({0: [0, 1], 1: [1], 2: [1, 2]}, n_items=4)
    assert pop.counts.tolist() == [1, 3, 1, 0]
    scores = pop.score(np.array([0, 1, 2]))
    assert scores.shape == (3, 4)
    assert (scores == pop.counts).all()


def test_adjacency_is_symmetric_and_normalised():
    m = LightGCN(n_users=2, n_items=2, dim=4, n_layers=1)
    adj = m.build_adj(np.array([[0, 0], [0, 1], [1, 0]])).to_dense()
    assert torch.allclose(adj, adj.T)
    # user0 has degree 2, item0 has degree 2 -> 1/sqrt(2*2)
    assert adj[0, 2 + 0].item() == pytest.approx(0.5)
    # user1 degree 1, item0 degree 2 -> 1/sqrt(2)
    assert adj[1, 2 + 0].item() == pytest.approx(1 / np.sqrt(2))
    # users never connect to users
    assert adj[:2, :2].abs().sum().item() == 0


def test_zero_layers_is_plain_embeddings_mf():
    m = LightGCN(3, 4, dim=5, n_layers=0)
    adj = m.build_adj(np.array([[0, 0], [1, 1], [2, 2]]))
    u, i = m.propagate(adj)
    assert torch.allclose(u, m.emb.weight[:3])
    assert torch.allclose(i, m.emb.weight[3:])


def test_one_layer_is_mean_of_layer0_and_layer1():
    m = LightGCN(2, 2, dim=3, n_layers=1)
    adj = m.build_adj(np.array([[0, 0], [1, 1]]))
    u, i = m.propagate(adj)
    x = m.emb.weight
    expected = (x + torch.sparse.mm(adj, x)) / 2
    assert torch.allclose(torch.cat([u, i]), expected)


@pytest.mark.parametrize("layers", [0, 2])
def test_bpr_training_reduces_loss(layers):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_users, n_items = 30, 20
    pairs = np.array([(u, (u // 3 * 2 + j) % n_items) for u in range(n_users) for j in range(3)])
    m = LightGCN(n_users, n_items, dim=16, n_layers=layers)
    adj = m.build_adj(pairs)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    losses = []
    for _ in range(80):
        idx = rng.integers(0, len(pairs), 64)
        loss = m.bpr_loss(
            adj,
            torch.tensor(pairs[idx, 0]),
            torch.tensor(pairs[idx, 1]),
            torch.tensor(rng.integers(0, n_items, 64)),
            1e-4,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_sample_negatives_avoids_positives():
    rng = np.random.default_rng(0)
    n_users, n_items = 50, 60
    pairs = np.array([(u, (u + j) % n_items) for u in range(n_users) for j in range(3)])
    keys = np.sort(pairs[:, 0] * n_items + pairs[:, 1])
    users = np.repeat(np.arange(n_users), 40)
    neg = sample_negatives(users, n_items, keys, rng)
    assert ((neg >= 0) & (neg < n_items)).all()
    assert not np.isin(users * n_items + neg, keys).any()
