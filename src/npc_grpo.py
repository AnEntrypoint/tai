"""GRPO-lite adherence RL for the NPC model (critic-free, DeepSeekMath-style).

Per prompt: sample K responses from the current policy, score each with the
rule-based adherence reward (no invented names, no template echo, intent
engagement, clean stop, non-trivial length, no n-gram looping), advantage =
(reward - group mean) / group std, and take a policy-gradient step on the
sampled tokens' per-token mean logprob (length-normalized, so the objective
scale does not grow with the generation window). No critic; a KL anchor to
the frozen starting policy at tiny lr.

Prompts are chosen by adaptive difficulty: each prompt's running pass rate
(mean reward >= 3.0) is tracked and the sampler draws from prompts in the
15-85% learning zone plus fresh prompts (RL-scaling curriculum), instead of
round-robin over already-saturated prompts. Response dedup is global across
the whole run, so a template that farms the reward once is penalized on
every later repeat.

The reward is the npc_score rule set: this is exactly the RLAIF-for-dialogue-
impression setup (arXiv:2501.12698) with a programmatic reward model, and the
Echoverse verifier-reads-rollouts loop with the scorer as the grounded grader.
"""

import argparse
import os
import random
import re

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, TinyLM
from npc_eval import PERSONAS
from npc_score import COMMON, INTENT_KEYS, TEMPLATE_ECHO, drift_names, ngram_repeat

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "data", "bpe32768.json")

QUESTIONS = [
    "Hello there.",
    "What do you have for sale?",
    "Tell me about yourself.",
    "Do you have a quest for me?",
]

UNSEEN_PERSONAS = [
    "Ysolde is a beekeeper on the cliff farms of Merrow Point. She speaks slowly, measures her words, and trusts patients more than promises.",
    "Capo is a one-eyed ratcatcher in the sewers of Brannock. Cheerful about horrible work, proud of his terrier Nails, and paid by the tail.",
    "Abbess Rime leads a snow-locked monastery library. Formal, dry-witted, and quietly protective of the forbidden shelf.",
    "Pell is a ten-year-old orphan guide in the Grand Bazaar. Fast, cheeky, knows every shortcut, charges two coppers and a secret.",
    "Magistra Vool runs a failing alchemical college. Defensive about the budget, brilliant about reagents, allergic to amateurs.",
    "Old Tam is a lighthouse keeper who talks to the fog. Sparse speech, long pauses, claims the light answers him back.",
]


def build_st_prompts(tok, n):
    import json
    import random as rnd
    rows = []
    path = os.path.join(HERE, "..", "data", "npc", "st_conversations.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line)["text"])
    rnd.seed(13)
    rnd.shuffle(rows)
    qs = ["Tell me about yourself.", "What do you have for sale?", "Do you have a quest for me?",
          "Any stories from these parts?", "Where are we?", "Hello."]
    out = []
    NL = chr(10)
    for row in rows[:n]:
        header, first = row.split("<START>" + NL, 1)
        name = first.split(":", 1)[0]
        first_line = first.split(NL, 1)[0]
        for q in rnd.sample(qs, 2):
            prompt = header + NL + "<START>" + NL + first_line + NL + "Player: " + q + NL + name + ":"
            out.append((prompt, q, header + NL + name))
    return out


def build_prompts(tok, wide=0):
    prompts = []
    for bio in PERSONAS:
        for q in QUESTIONS:
            prompts.append((bio, q))
    for bio in UNSEEN_PERSONAS:
        for q in QUESTIONS:
            prompts.append((f"You are an NPC. {bio}", q))
    if wide:
        import json
        path = os.path.join(HERE, "..", "data", "npc", "synthetic_names.jsonl")
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line)["text"])
        import random as rnd
        rnd.seed(11)
        rnd.shuffle(rows)
        for row in rows[:wide]:
            sys_part, rest = row.split("### Player:", 1)
            q_first = rest.split("### NPC:", 1)[0].strip()
            prompts.append((sys_part + "### System:\n" if not sys_part.endswith("\n") else sys_part, q_first))
    return prompts


def reward_of(text, stopped, q, bio):
    r = 0.0
    body = text.strip()
    swapped = False
    for m in re.finditer(r"(?:I am|call me|I'm) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", body):
        if m.group(1) not in bio:
            swapped = True
            break
    if not swapped:
        for m in re.finditer(r"([A-Z][a-z]+) is the name", body):
            if m.group(1) not in bio:
                swapped = True
                break
    if swapped:
        return -1.0
    r += 1.0
    if "Description:" in body or "<START>" in body:
        return -0.5
    if any(t in body.lower() for t in TEMPLATE_ECHO):
        return -0.5
    keys = INTENT_KEYS.get(q)
    if keys is None or any(k in body.lower() for k in keys):
        r += 1.0
    blob_words = set(w for w in re.findall(r"[a-z]{5,}", bio.lower())
                     if w not in {"about", "their", "would", "could", "should", "there", "which",
                                  "these", "those", "where", "every", "player", "start"})
    body_words = set(re.findall(r"[a-z]{5,}", body.lower()))
    if blob_words & body_words:
        r += 1.0
    if stopped or (chr(10) + "Player:") in text or chr(10) in body:
        r += 0.5
    if len(body) >= 24:
        r += 0.5
    if ngram_repeat(body):
        r -= 1.0
    return r


def generate_k(model, ids, k, tokens, temperature, top_k):
    idx = ids.repeat(k, 1)
    outs = []
    with torch.no_grad():
        for _ in range(tokens):
            logits, _ = model(idx)
            z = logits[:, -1, :] / temperature
            thresh = z.topk(top_k, dim=-1).values[:, -1:]
            z = z.masked_fill(z < thresh, float("-inf"))
            nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            outs.append(nxt)
    return torch.cat(outs, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts", type=int, default=0, help="prompts per cycle; 0 = all")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--kl-beta", type=float, default=0.04)
    ap.add_argument("--wide", type=int, default=0, help="extra random-name personas from synthetic_names.jsonl")
    ap.add_argument("--st", type=int, default=0, help="use ST-format prompts with N cards")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(ck["state"])
    ref = TinyLM(cfg).to(device)
    ref.load_state_dict(ck["state"])
    for prm in ref.parameters():
        prm.requires_grad = False
    tok = Tokenizer.from_file(TOK)
    eot = tok.token_to_id("<|endoftext|>")

    failures = open(os.path.join(HERE, "..", "data", "npc", "grpo-failures.jsonl"), "a", encoding="utf-8")
    if args.st:
        prompts = [(p, q, blob) for p, q, blob in build_st_prompts(tok, args.st)]
    else:
        prompts = [(b, q, b) for b, q in build_prompts(tok, args.wide)]
    if args.prompts:
        prompts = prompts[: args.prompts]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    stats = [[0, 0] for _ in prompts]
    seen_global = {}
    step_rng = random.Random(7)
    for step in range(args.steps):
        in_zone = [i for i, (a, p) in enumerate(stats) if a >= 2 and 0.15 <= p / a <= 0.85]
        fresh = [i for i, (a, p) in enumerate(stats) if a < 2]
        pool = (in_zone + fresh) or list(range(len(prompts)))
        pidx = step_rng.choice(pool)
        prompt, q, bio = prompts[pidx]
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        plen = ids.shape[1]

        model.eval()
        gen = generate_k(model, ids, args.k, args.tokens, args.temperature, 40)
        full = torch.cat([ids.repeat(args.k, 1), gen], dim=1)

        texts, rewards, seqs = [], [], []
        for row in full:
            resp = row[plen:].tolist()
            text = tok.decode(resp)
            stops = [c for c in (text.find("### Player:"), text.find("### System:")) if c >= 0]
            stopped = bool(stops)
            texts.append(text)
            rewards.append(reward_of(text, stopped, q, bio))
            seqs.append(row)

        for i, text in enumerate(texts):
            key = " ".join(text.split())[:120]
            if key in seen_global:
                rewards[i] -= 0.5
            seen_global[key] = True

        for text, r in zip(texts, rewards):
            if r < 1.5:
                import json
                failures.write(json.dumps({"bio": bio, "q": q, "response": text, "reward": r}) + "\n")
        mean_r = sum(rewards) / len(rewards)
        stats[pidx][0] += 1
        stats[pidx][1] += mean_r >= 3.0
        adv = torch.tensor([r - mean_r for r in rewards], device=device, dtype=torch.float32)
        if adv.abs().max() < 1e-6:
            continue
        adv = adv / (adv.std() + 1e-4)

        model.train()
        seq = torch.stack(seqs)
        inp, tgt = seq[:, :-1], seq[:, 1:]
        logits, _ = model(inp)
        lp = F.log_softmax(logits.float(), dim=-1)
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        resp_lp = tok_lp[:, plen - 1:].mean(-1)
        with torch.no_grad():
            ref_logits, _ = ref(inp)
            ref_lp_full = F.log_softmax(ref_logits.float(), dim=-1)
            ref_tok_lp = ref_lp_full.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        kl = (tok_lp[:, plen - 1:] - ref_tok_lp[:, plen - 1:]).mean()
        loss = -(adv * resp_lp).mean() + args.kl_beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0 or step == args.steps - 1:
            best = texts[rewards.index(max(rewards))]
            print(f"step {step:4d} | reward {mean_r:+.2f} (max {max(rewards):+.2f} min {min(rewards):+.2f}) "
                  f"| loss {loss.item():+.4f} | q={q[:28]!r}", flush=True)

    failures.close()
    out = args.out or args.ckpt.replace(".pt", "-grpo.pt")
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
