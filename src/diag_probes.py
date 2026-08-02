"""Diagnostic probes: decoding grid + teacher-forced coherence ranking.

Separates decoding levers from training levers on a fixed set of ST card
prompts: for each decoding config, generate and measure a non-word rate
(words absent from a corpus-derived dictionary); then compare the model's
own teacher-forced logprob on gold continuations vs its own samples.
"""

import argparse
import json
import os
import re
from collections import Counter

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")

QUESTIONS = ["Tell me about yourself.", "What do you have for sale?",
             "Any stories from these parts?", "Where are we?"]


def build_lexicon():
    counts = Counter()
    sources = [os.path.join(DATA, "tinystories_slice.txt"),
               os.path.join(NPC, "st_conversations.jsonl")]
    for path in sources:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if path.endswith(".jsonl"):
                    try:
                        line = json.loads(line)["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue
                counts.update(w for w in re.findall(r"[a-z]{4,}", line.lower()))
    return {w for w, c in counts.items() if c >= 20}


def nonword_rate(text, lex):
    words = re.findall(r"[a-z]{4,}", text.lower())
    if not words:
        return 0.0, []
    bad = [w for w in words if w not in lex]
    return len(bad) / len(words), bad


def gen(model, ids, tokens, temperature, top_k):
    outs = []
    with torch.no_grad():
        for _ in range(tokens):
            logits, _ = model(ids)
            z = logits[:, -1, :]
            if temperature <= 0:
                nxt = z.argmax(dim=-1, keepdim=True)
            else:
                z = z / temperature
                th = z.topk(top_k, dim=-1).values[:, -1:]
                z = z.masked_fill(z < th, float("-inf"))
                nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            outs.append(nxt)
    return torch.cat(outs, dim=1)


def mean_lp(model, ids, plen):
    with torch.no_grad():
        logits, _ = model(ids[:, :-1])
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_lp[0, plen - 1:].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=77)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))
    lex = build_lexicon()
    print(f"lexicon: {len(lex)} words")

    import random
    rows = [json.loads(l)["text"] for l in open(os.path.join(NPC, "st_conversations.jsonl"), encoding="utf-8") if l.strip()]
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: args.rows]

    configs = [("greedy", 0.0, 0), ("t0.3", 0.3, 40), ("t0.5", 0.5, 40), ("t0.7", 0.7, 40), ("t0.9", 0.9, 40)]
    NL = chr(10)
    grid_rates = {name: [] for name, _, _ in configs}
    for ridx, row in enumerate(rows):
        header, rest = row.split("<START>" + NL, 1)
        name = rest.split(":", 1)[0]
        lines = rest.split(NL)
        first_line = lines[0]
        gold = None
        q = None
        other_gold = None
        for i, ln in enumerate(lines):
            if ln.startswith("Player:") and i + 1 < len(lines) and not lines[i + 1].startswith("Player:"):
                if q is None:
                    q = ln.split(":", 1)[1].strip()
                    gold = lines[i + 1].split(":", 1)[-1].strip()
                elif other_gold is None:
                    other_gold = lines[i + 1].split(":", 1)[-1].strip()
                    break
        if q is None:
            q = QUESTIONS[ridx % len(QUESTIONS)]
        prompt = header + NL + "<START>" + NL + first_line + NL + "Player: " + q + NL + name + ":"
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        plen = ids.shape[1]
        print(f"== card {ridx} ({name}) q={q!r}")
        for cname, temp, tk in configs:
            out = gen(model, ids.clone(), args.tokens, temp, tk)
            text = tok.decode(out[0].tolist())
            rate, bad = nonword_rate(text, lex)
            grid_rates[cname].append(rate)
            print(f"  [{cname}] nonword={rate:.2f} :: {text.strip()[:110]!r}" + (f" bad={bad[:4]}" if bad else ""))
        if gold:
            gold_ids = torch.tensor([tok.encode(prompt + " " + gold).ids], device=device)
            gl = mean_lp(model, gold_ids, plen)
            own = gen(model, ids.clone(), args.tokens, 0.7, 40)
            own_ids = torch.cat([ids, own], dim=1)
            ol = mean_lp(model, own_ids, plen)
            ml = float("nan")
            if other_gold:
                mis_ids = torch.tensor([tok.encode(prompt + " " + other_gold).ids], device=device)
                ml = mean_lp(model, mis_ids, plen)
            print(f"  teacher-forced mean lp: gold={gl:.3f} mismatched={ml:.3f} own_sample={ol:.3f} delta={gl - ol:+.3f}")
            print(f"  gold: {gold[:110]!r}")
    print("== decode grid mean non-word rates:")
    for cname, _, _ in configs:
        rates = grid_rates[cname]
        print(f"  {cname}: {sum(rates) / len(rates):.3f}")


if __name__ == "__main__":
    main()
