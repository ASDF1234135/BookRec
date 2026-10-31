"""Train a model, evaluate, log to MLflow."""

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from bookrec.eval.metrics import evaluate
from bookrec.models.lightgcn import LightGCN
from bookrec.models.popularity import Popularity

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "bookrec"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def to_pos(df: pd.DataFrame) -> dict:
    """user_idx -> list of book_idx."""
    return df.groupby("user_idx")["book_idx"].agg(list).to_dict()


def load_data(cfg: dict) -> dict:
    """Read processed parquet splits.

    Normal run:  fit on train, evaluate on val (hyper-parameter search).
    cfg["final"]: fit on train+val, evaluate on t1 (the official test set).
    In both cases the exclusion set for ranking is exactly the fit data.
    """
    d = Path(cfg.get("data_dir", "data/processed"))
    train = pd.read_parquet(d / "train.parquet")
    val = pd.read_parquet(d / "val.parquet")

    if cfg.get("final", False):
        fit = pd.concat([train, val], ignore_index=True)
        target = pd.read_parquet(d / "t1.parquet")
        split_name = "t1"
    else:
        fit, target, split_name = train, val, "val"

    card_path = d / "data_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
        n_users, n_items, data_hash = card["n_users"], card["n_books"], card["data_hash"]
    else:  # e.g. the CI fixtures have no data card
        n_users = int(train["user_idx"].max()) + 1
        n_items = int(train["book_idx"].max()) + 1
        data_hash = hashlib.sha256((d / "train.parquet").read_bytes()).hexdigest()[:16]

    return {
        "fit": fit,
        "target": target,
        "split_name": split_name,
        "n_users": n_users,
        "n_items": n_items,
        "data_hash": data_hash,
    }


def sample_negatives(users: np.ndarray, n_items: int, pos_keys: np.ndarray, rng) -> np.ndarray:
    """One uniform negative per user, re-drawn (a few times) if it is a known positive.

    pos_keys is the sorted array of user * n_items + item for all training positives.
    """
    neg = rng.integers(0, n_items, len(users))
    for _ in range(5):
        q = users.astype(np.int64) * n_items + neg
        at = np.minimum(np.searchsorted(pos_keys, q), len(pos_keys) - 1)
        bad = pos_keys[at] == q
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, int(bad.sum()))
    return neg


def make_score_fn(model: LightGCN, adj):
    """score_fn for evaluate(): propagate once, then user_emb @ item_emb.T per batch."""
    with torch.no_grad():
        user_emb, item_emb = model.propagate(adj)

    def score_fn(user_ids):
        with torch.no_grad():
            return user_emb[torch.as_tensor(user_ids)] @ item_emb.T

    return score_fn


def train_lightgcn(cfg: dict, data: dict, seed: int) -> tuple[LightGCN, torch.Tensor, dict]:
    rng = np.random.default_rng(seed)
    n_users, n_items, k = data["n_users"], data["n_items"], cfg.get("k", 20)

    pairs = data["fit"][["user_idx", "book_idx"]].to_numpy(dtype=np.int64)
    pos_keys = np.sort(pairs[:, 0] * n_items + pairs[:, 1])
    fit_pos, target_pos = to_pos(data["fit"]), to_pos(data["target"])

    # Validation during training can use a fixed subset of users to save time
    # (plan: ~5,000). The final report below always uses everyone.
    val_pos = target_pos
    max_users = cfg.get("val_max_users")
    if max_users and len(target_pos) > max_users:
        chosen = np.random.default_rng(0).choice(sorted(target_pos), max_users, replace=False)
        val_pos = {int(u): target_pos[u] for u in chosen}

    model = LightGCN(n_users, n_items, cfg["dim"], cfg["n_layers"])
    adj = model.build_adj(pairs)  # fixed graph: build once
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    final = cfg.get("final", False)
    max_epochs = cfg.get("final_epochs", cfg["max_epochs"]) if final else cfg["max_epochs"]
    best, best_epoch, best_state, bad_evals = -1.0, 0, None, 0
    batch_size = cfg["batch_size"]

    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(len(pairs))
        total = 0.0
        for s in range(0, len(perm), batch_size):
            b = pairs[perm[s : s + batch_size]]
            neg = sample_negatives(b[:, 0], n_items, pos_keys, rng)
            loss = model.bpr_loss(
                adj,
                torch.from_numpy(b[:, 0]),
                torch.from_numpy(b[:, 1]),
                torch.from_numpy(neg),
                cfg["l2"],
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        mlflow.log_metric("train_loss", total / len(pairs), step=epoch)

        # Early stopping only in the validation setting; the final run trains a fixed
        # number of epochs (set final_epochs to the best_epoch of the validation run).
        if not final and epoch % cfg["eval_every"] == 0:
            res = evaluate(make_score_fn(model, adj), fit_pos, val_pos, n_items, k)
            score = res[f"ndcg@{k}"]
            mlflow.log_metric(f"val_ndcg_at_{k}", score, step=epoch)
            print(f"epoch {epoch}: loss {total / len(pairs):.4f} val ndcg@{k} {score:.4f}")
            if score > best:
                best, best_epoch, bad_evals = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_evals += 1
                if bad_evals >= cfg["patience"]:
                    print("early stopping")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, adj, {"best_epoch": best_epoch if not final else max_epochs}


def run_one(cfg: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_data(cfg)
    k = cfg.get("k", 20)
    fit_pos, target_pos = to_pos(data["fit"]), to_pos(data["target"])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    name = f"{cfg['model']}-L{cfg.get('n_layers', 0)}-seed{seed}"

    with mlflow.start_run(run_name=name) as active:
        mlflow.log_params({**cfg, "seed": seed, "seeds": str(cfg.get("seeds"))})
        mlflow.set_tags(
            {
                "git_hash": git_hash(),
                "data_hash": data["data_hash"],
                "eval_split": data["split_name"],
            }
        )

        start = time.time()
        extra, user_emb, item_emb, model = {}, None, None, None
        if cfg["model"] == "popularity":
            pop = Popularity().fit(fit_pos, data["n_items"])
            score_fn = pop.score
        else:
            model, adj, extra = train_lightgcn(cfg, data, seed)
            score_fn = make_score_fn(model, adj)
            with torch.no_grad():
                u, i = model.propagate(adj)
            user_emb, item_emb = u.numpy(), i.numpy()
        train_time = time.time() - start

        # Final report: always the full target set.
        res = evaluate(score_fn, fit_pos, target_pos, data["n_items"], k)
        per_user = res.pop("per_user_ndcg")
        user_ids = res.pop("user_ids")

        # MLflow metric names cannot contain "@".
        metrics = {key.replace("@", "_at_"): v for key, v in res.items()}
        metrics["train_time_s"] = train_time
        if "best_epoch" in extra:
            metrics["best_epoch"] = extra["best_epoch"]
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.savez(tmp / "per_user_ndcg.npz", user_ids=user_ids, ndcg=per_user)
            mlflow.log_artifact(str(tmp / "per_user_ndcg.npz"))
            if model is not None:
                # Embeddings are what export.py needs to build the FAISS index.
                np.save(tmp / "user_emb.npy", user_emb)
                np.save(tmp / "item_emb.npy", item_emb)
                torch.save(model.state_dict(), tmp / "model.pt")
                for f in ("user_emb.npy", "item_emb.npy", "model.pt"):
                    mlflow.log_artifact(str(tmp / f))

        print(f"[{name}] run_id={active.info.run_id}")
        print({key: round(v, 4) for key, v in metrics.items()})
        return {"run_id": active.info.run_id, **metrics}


def run(cfg: dict, all_seeds: bool = False) -> list[dict]:
    """Train with cfg['seed']; with all_seeds, once per entry in cfg['seeds']."""
    seeds = cfg.get("seeds", [cfg.get("seed", 0)]) if all_seeds else [cfg.get("seed", 0)]
    return [run_one(cfg, s) for s in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--all-seeds", action="store_true", help="run every seed in the config")
    ap.add_argument("--final", action="store_true", help="fit on train+val, evaluate on T1")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.final:
        cfg["final"] = True
    run(cfg, all_seeds=args.all_seeds)


if __name__ == "__main__":
    main()
