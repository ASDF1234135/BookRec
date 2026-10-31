"""Small grid search: run train.run() for every combination and rank by validation NDCG.

Example:
    python -m bookrec.grid --config configs/lightgcn.yaml \
        --grid n_layers=1,2,3 lr=1e-3,5e-3 l2=1e-4,1e-3
"""

import argparse
import itertools

import yaml

from bookrec.train import load_config, run


def parse_grid(items: list[str]) -> dict[str, list]:
    """['lr=1e-3,5e-3', 'n_layers=1,2'] -> {'lr': [0.001, 0.005], 'n_layers': [1, 2]}."""
    grid = {}
    for item in items:
        key, values = item.split("=", 1)
        # yaml.safe_load turns "1e-3" into a float only if written as 1.0e-3, so go via float/int.
        grid[key] = [_to_number(v) for v in values.split(",")]
    return grid


def _to_number(text: str):
    value = yaml.safe_load(text)
    if isinstance(value, str):  # e.g. "1e-3" is read as a string by YAML
        try:
            return float(text)
        except ValueError:
            return value
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--grid", nargs="+", required=True, help="key=v1,v2 ...")
    args = ap.parse_args()

    base = load_config(args.config)
    grid = parse_grid(args.grid)
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"{len(combos)} runs: {grid}")

    k = base.get("k", 20)
    results = []
    for i, values in enumerate(combos, 1):
        overrides = dict(zip(keys, values, strict=True))
        print(f"\n=== run {i}/{len(combos)}: {overrides}")
        out = run({**base, **overrides})[0]
        results.append((out[f"ndcg_at_{k}"], overrides, out["run_id"], out.get("best_epoch")))

    results.sort(key=lambda r: r[0], reverse=True)
    print("\n=== results (best first)")
    for score, overrides, run_id, best_epoch in results:
        print(f"ndcg@{k}={score:.4f}  best_epoch={best_epoch}  {overrides}  run_id={run_id}")


if __name__ == "__main__":
    main()
