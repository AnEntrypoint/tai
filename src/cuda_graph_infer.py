"""CUDA-graph decode engine for the PLE TinyLM.

One full single-token decode step is captured as a torch.cuda.CUDAGraph over
static buffers (KV cache at max length, masked attention driven by a GPU
position scalar), then replayed once per generated token. Greedy argmax
stays on the GPU; only one token id crosses PCIe per token.

This is the honest test of "CUDA graphs bypass the CPU": for a 28.9M model
the CPU int8 runtime reads ~2.5MB of weights per token while an fp16 GPU
step reads ~58MB, so the question is bandwidth vs launch overhead, and only
measurement answers it.
"""

import argparse
import os
import time

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "data", "bpe32768.json")


class GraphEngine:
    def __init__(self, ckpt, device="cuda", dtype=torch.float16):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = Config(**ck["cfg"])
        model = TinyLM(self.cfg)
        model.load_state_dict(ck["state"])
        model.eval()
        self.m = model.to(device=device, dtype=dtype)
        self.device = device
        c = self.cfg
        self.nh, self.dh = c.n_heads, c.head_dim
        self.kc = torch.zeros(c.n_layers, c.seq_len, c.n_heads * self.dh, device=device, dtype=dtype)
        self.vc = torch.zeros(c.n_layers, c.seq_len, c.n_heads * self.dh, device=device, dtype=dtype)
        self.tok = torch.zeros(1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.rope_c = self.m.cos.to(device=device, dtype=dtype)
        self.rope_s = self.m.sin.to(device=device, dtype=dtype)
        self.arange = torch.arange(c.seq_len, device=device)

    def step(self):
        m, c = self.m, self.cfg
        D, L, P, F = c.d_model, c.n_layers, c.ple_dim, c.ffn_hidden
        H, Dh = self.nh, self.dh
        pos = self.pos
        x = m.tok_emb(self.tok).unsqueeze(0)
        ple = m.ple_model_proj(x) * (D ** -0.5)
        ple = m.ple_proj_norm(ple.view(1, 1, L, P))
        table = m.ple_table(self.tok).view(1, 1, L, P)
        ple = (ple + table * (P ** 0.5)) * (2 ** -0.5)
        cos = self.rope_c[pos].view(1, 1, 1, Dh // 2)
        sin = self.rope_s[pos].view(1, 1, 1, Dh // 2)
        causal = (self.arange <= pos).view(1, 1, 1, c.seq_len)
        for i, block in enumerate(m.blocks):
            h = block.attn_norm(x)
            qkv = block.attn.qkv(h)
            q, k, v = qkv.split(D, dim=2)
            q = q.view(1, 1, H, Dh).transpose(1, 2)
            k = k.view(1, 1, H, Dh).transpose(1, 2)
            v = v.view(1, 1, H, Dh).transpose(1, 2)
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
            self.kc[i].index_copy_(0, pos, k.reshape(1, H * Dh))
            self.vc[i].index_copy_(0, pos, v.reshape(1, H * Dh))
            ks = self.kc[i].view(1, c.seq_len, H, Dh).transpose(1, 2)
            vs = self.vc[i].view(1, c.seq_len, H, Dh).transpose(1, 2)
            scores = (q @ ks.transpose(-1, -2)) * (Dh ** -0.5)
            scores = scores.masked_fill(~causal, float("-inf"))
            o = torch.softmax(scores, dim=-1) @ vs
            x = x + block.attn.proj(o.transpose(1, 2).reshape(1, 1, D))
            h = block.ffn_norm(x)
            x = x + block.ffn.down(torch.nn.functional.silu(block.ffn.gate(h)) * block.ffn.up(h))
            g = torch.nn.functional.gelu(block.ple_gate(x))
            x = x + block.ple_norm(block.ple_proj(g * ple[:, :, i]))
        x = m.out_norm(x)
        return m.head(x)[0, -1].float()

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = self.step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self.step()

    def forward(self, token, pos):
        self.tok.fill_(token)
        self.pos.fill_(pos)
        self.graph.replay()
        return self.static_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--prompt", default="Once upon a time")
    args = ap.parse_args()

    eng = GraphEngine(args.ckpt)
    tok = Tokenizer.from_file(TOK)
    ids = tok.encode(args.prompt).ids
    cfg = eng.cfg

    t0 = time.perf_counter()
    eng.capture()
    t_capture = time.perf_counter() - t0

    for i, t in enumerate(ids):
        logits = eng.forward(t, i)
    next_id = int(logits.argmax().item())

    t0 = time.perf_counter()
    n = 0
    pos = len(ids)
    for _ in range(args.tokens):
        if pos >= cfg.seq_len:
            break
        logits = eng.forward(next_id, pos)
        next_id = int(logits.argmax().item())
        pos += 1
        n += 1
    dt = time.perf_counter() - t0

    print(f"prompt: {args.prompt!r}")
    print(f"next token after prompt: {tok.decode([next_id])!r}")
    print(f"(decode loop ran {n} tokens; text print elided, use npc_eval.py for text)")
    print(f"capture: {t_capture:.2f}s   decode: {n} tokens in {dt:.2f}s = {n / dt:.1f} tok/s ({dt * 1000 / max(n, 1):.2f} ms/token)")


if __name__ == "__main__":
    main()
