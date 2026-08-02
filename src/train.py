"""Train one ablation arm and report val loss at matched core-parameter budget."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import Config, TinyLM, make_model

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RUNS = os.path.join(HERE, "..", "runs")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Batcher:
    def __init__(self, split, batch_size, seq_len, device, suffix=""):
        self.data = np.memmap(os.path.join(DATA, f"{split}{suffix}.bin"), dtype=np.uint16, mode="r")
        self.bs, self.sl, self.device = batch_size, seq_len, device
        self.rng = np.random.default_rng(1234 if split == "val" else None)

    def __call__(self):
        ix = self.rng.integers(0, len(self.data) - self.sl - 1, self.bs)
        x = np.stack([self.data[i : i + self.sl] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.sl] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)


@torch.no_grad()
def evaluate(model, batcher, iters):
    model.eval()
    batcher.rng = np.random.default_rng(1234)  # same val batches for every arm
    losses = [model(*batcher())[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, total, peak, warmup, stable_frac=0.6):
    if step < warmup:
        return peak * (step + 1) / warmup
    stable_end = warmup + int((total - warmup) * stable_frac)
    if step < stable_end:
        return peak
    p = (step - stable_end) / max(1, total - stable_end)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["baseline", "ple", "ple_notable", "fatembed", "bigcore"],
    )
    ap.add_argument("--target-core", type=int, default=1_500_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                    help="muon orthogonalizes momentum updates on the 2D core matrices (Newton-Schulz), AdamW handles embeddings/tables/norms")
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--stable-frac", type=float, default=0.6,
                    help="WSD: fraction of post-warmup steps held at peak lr before decay")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--ple-dim", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--fixed-ffn", type=int, default=None,
                    help="pin ffn_hidden and skip the core solver (table-scaling sweep)")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--data-suffix", default=None,
                    help="override the train/val bin suffix (e.g. _npc for train_npc.bin)")
    ap.add_argument("--init-from", default=None,
                    help="path to a runs/*.pt checkpoint to fine-tune from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(RUNS, exist_ok=True)

    # vocab 4096 uses the original train.bin/val.bin; other vocabs use suffixed bins.
    suffix = args.data_suffix if args.data_suffix is not None else (
        "" if args.vocab == 4096 else f"_v{args.vocab}"
    )

    base = Config(seq_len=args.seq_len, ple_dim=args.ple_dim, vocab_size=args.vocab,
                  d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
    model = make_model(args.arm, args.target_core, base, fixed_ffn=args.fixed_ffn).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["state"])
        print(f"initialized from {args.init_from}")
    budget = model.param_budget()
    cfg = model.cfg

    # No weight decay on 1-D params (norms) or on lookup tables.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "table" in n or "tok_emb" in n else decay).append(p)
    adam_decay = [] if args.optimizer == "muon" else decay
    opt = torch.optim.AdamW(
        [{"params": adam_decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    muon_bufs = [torch.zeros_like(p) for p in decay] if args.optimizer == "muon" else []

    train_b = Batcher("train", args.batch_size, args.seq_len, device, suffix)
    val_b = Batcher("val", args.batch_size, args.seq_len, device, suffix)

    name = f"{args.arm}{'-' + args.tag if args.tag else ''}-s{args.seed}"
    history, best = [], float("inf")
    t0 = time.time()

    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup, args.stable_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_b()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.optimizer == "muon":
            muon_lr = lr_at(step, args.steps, args.muon_lr, args.warmup, args.stable_frac)
            with torch.no_grad():
                for p, buf in zip(decay, muon_bufs):
                    buf.lerp_(p.grad, 1 - args.muon_momentum)
                    u = p.grad.lerp(buf, args.muon_momentum)
                    p.add_(zeropower_via_newtonschulz5(u),
                           alpha=-muon_lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)

        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_b, args.eval_iters)
            best = min(best, vl)
            tok = (step + 1) * args.batch_size * args.seq_len
            history.append({"step": step, "tokens": tok, "train": loss.item(), "val": vl})
            print(
                f"{name} step {step:5d} | tok {tok / 1e6:6.1f}M | train {loss.item():.4f} "
                f"| val {vl:.4f} | ppl {math.exp(vl):7.2f} | {time.time() - t0:5.0f}s",
                flush=True,
            )
            torch.save({"cfg": cfg.__dict__, "state": model.state_dict()},
                       os.path.join(RUNS, f"{name}-latest.pt"))

    result = {
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "params": budget,
        "final_val": history[-1]["val"],
        "best_val": best,
        "final_ppl": math.exp(history[-1]["val"]),
        "tokens_seen": args.steps * args.batch_size * args.seq_len,
        "steps": args.steps,
        "wall_seconds": time.time() - t0,
        "history": history,
    }
    with open(os.path.join(RUNS, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict()},
               os.path.join(RUNS, f"{name}.pt"))
    print(f"{name} DONE core={budget['core']:,} table={budget['table']:,} "
          f"val={result['final_val']:.4f} ppl={result['final_ppl']:.2f}")


if __name__ == "__main__":
    main()
