"""Write a random-init checkpoint in the train.py format so src/export.py can
produce a model.bin + golden.txt without a training run.

The default config matches the shipped 28.9M-parameter deploy model; --small
emits a tiny config suitable for a committed CI fixture. TinyLM zeroes the
ple_norm gains at init so training starts as a no-op; here they are
re-randomized so the PLE branch numerics are exercised during verification.
"""

import argparse
import os

import torch

from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="checkpoint name; written to runs/<tag>.pt")
    ap.add_argument("--small", action="store_true", help="tiny config for CI fixtures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.small:
        cfg = Config(arm="ple", vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                     ffn_hidden=96, seq_len=128, ple_dim=32)
    else:
        cfg = Config(arm="ple", vocab_size=32768, d_model=96, n_layers=6, n_heads=4,
                     ffn_hidden=66, seq_len=512, ple_dim=128)
    torch.manual_seed(args.seed)
    model = TinyLM(cfg)
    for block in model.blocks:
        torch.nn.init.normal_(block.ple_norm.weight, std=0.5)

    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, f"{args.tag}.pt")
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, path)
    b = model.param_budget()
    print(f"wrote {path}  core={b['core']:,} stream={b['stream']:,} table={b['table']:,}")


if __name__ == "__main__":
    main()
