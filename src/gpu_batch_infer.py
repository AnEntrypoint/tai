"""Batched multi-stream CUDA-graph decode engine for the PLE TinyLM.

One captured graph serves B NPC streams per replay: tokens and positions are
GPU vectors, each stream gets its own causal mask from its own position
scalar, and the KV cache is [B, layers, seq, dim]. Weights are read once per
replay for all B streams, which is the entire point of batching tiny models
on a GPU: the single-stream graph is kernel-serialization-bound, so B
streams ride nearly free until compute saturates.
"""

import argparse
import os
import time

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "data", "bpe32768.json")


class BatchEngine:
    def __init__(self, ckpt, batch, device="cuda", dtype=torch.float16):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = Config(**ck["cfg"])
        model = TinyLM(self.cfg)
        model.load_state_dict(ck["state"])
        model.eval()
        self.m = model.to(device=device, dtype=dtype)
        self.device = device
        c = self.cfg
        b = batch
        self.batch = b
        self.nh, self.dh = c.n_heads, c.head_dim
        self.kc = torch.zeros(c.n_layers, b, c.seq_len, c.n_heads * self.dh, device=device, dtype=dtype)
        self.vc = torch.zeros(c.n_layers, b, c.seq_len, c.n_heads * self.dh, device=device, dtype=dtype)
        self.tok = torch.zeros(b, dtype=torch.long, device=device)
        self.pos = torch.zeros(b, dtype=torch.long, device=device)
        self.rope_c = self.m.cos.to(device=device, dtype=dtype)
        self.rope_s = self.m.sin.to(device=device, dtype=dtype)
        self.arange = torch.arange(c.seq_len, device=device)

    def step(self):
        m, c = self.m, self.cfg
        b = self.batch
        D, L, P = c.d_model, c.n_layers, c.ple_dim
        H, Dh = self.nh, self.dh
        x = m.tok_emb(self.tok).unsqueeze(1)
        ple = m.ple_model_proj(x) * (D ** -0.5)
        ple = m.ple_proj_norm(ple.view(b, 1, L, P))
        table = m.ple_table(self.tok).view(b, 1, L, P)
        ple = (ple + table * (P ** 0.5)) * (2 ** -0.5)
        cos = self.rope_c[self.pos].view(b, 1, 1, Dh // 2)
        sin = self.rope_s[self.pos].view(b, 1, 1, Dh // 2)
        causal = self.arange.view(1, 1, 1, c.seq_len) <= self.pos.view(b, 1, 1, 1)
        for i, block in enumerate(m.blocks):
            h = block.attn_norm(x)
            qkv = block.attn.qkv(h)
            q, k, v = qkv.split(D, dim=2)
            q = q.view(b, 1, H, Dh).transpose(1, 2)
            k = k.view(b, 1, H, Dh).transpose(1, 2)
            v = v.view(b, 1, H, Dh).transpose(1, 2)
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
            idx = self.pos.view(b, 1, 1).expand(b, 1, H * Dh)
            self.kc[i].scatter_(1, idx, k.reshape(b, 1, H * Dh))
            self.vc[i].scatter_(1, idx, v.reshape(b, 1, H * Dh))
            ks = self.kc[i].view(b, c.seq_len, H, Dh).transpose(1, 2)
            vs = self.vc[i].view(b, c.seq_len, H, Dh).transpose(1, 2)
            scores = (q @ ks.transpose(-1, -2)) * (Dh ** -0.5)
            scores = scores.masked_fill(~causal, float("-inf"))
            o = torch.softmax(scores, dim=-1) @ vs
            x = x + block.attn.proj(o.transpose(1, 2).reshape(b, 1, D))
            h = block.ffn_norm(x)
            x = x + block.ffn.down(torch.nn.functional.silu(block.ffn.gate(h)) * block.ffn.up(h))
            g = torch.nn.functional.gelu(block.ple_gate(x))
            x = x + block.ple_norm(block.ple_proj(g * ple[:, :, i]))
        x = m.out_norm(x)
        return m.head(x)[:, -1].float()

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self.step()

    def forward(self, tokens, positions):
        self.tok.copy_(tokens)
        self.pos.copy_(positions)
        self.graph.replay()
        return self.static_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOK)
    base = tok.encode("Once upon a time").ids

    sizes = [1, 4, 8, 16, 32, 64] if args.sweep else [args.batch]
    for b in sizes:
        eng = BatchEngine(args.ckpt, b)
        eng.capture()
        prompts = [base[: 2 + (i % 3)] for i in range(b)]
        cur = torch.zeros(b, dtype=torch.long, device="cuda")
        pos = torch.zeros(b, dtype=torch.long, device="cuda")
        for j in range(max(len(p) for p in prompts)):
            for i, p in enumerate(prompts):
                if j < len(p):
                    cur[i] = p[j]
                    pos[i] = j
            out = eng.forward(cur, pos)
        cur = out.argmax(-1)
        t0 = time.perf_counter()
        for _ in range(args.tokens):
            pos = pos + 1
            out = eng.forward(cur, pos)
            cur = out.argmax(-1)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total = args.tokens * b
        print(f"B={b:3d}: {total} tokens in {dt:.2f}s = {total / dt:.0f} tok/s aggregate ({dt * 1000 / args.tokens:.2f} ms/step)")


if __name__ == "__main__":
    main()
