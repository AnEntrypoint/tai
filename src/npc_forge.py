"""The forge: adversarial generate-grade-inject co-evolution cycles.

Each cycle:
  1. GENERATE: the current model answers a spread of ST-format card prompts
     (identity, sale, quest, lore, object interaction, place) from both card
     pools and unseen random names.
  2. GRADE: rule-based pre-filters per response -- prefix compliance, name
     drift, intent engagement, clean stop, repetition, object grounding --
     into a flaw histogram (the dashboard a judge reads).
  3. INJECT: passing rollouts become rejection-sampled SFT data
     (st_forge_data.jsonl); failures are logged with their flaw classes for
     the judge to repair (st_forge_failures.jsonl).

The judge (an LLM, not these rules) reads the dashboard and failures each
cycle and authors targeted training data; the forge feeds it back in.
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from npc_score import COMMON, INTENT_KEYS, TEMPLATE_ECHO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")

QUESTIONS = {
    "identity": "Tell me about yourself.",
    "sale": "What do you have for sale?",
    "quest": "Do you have a quest for me?",
    "lore": "Any stories from these parts?",
    "object": "Is that thing on your table for sale?",
    "place": "Where are we?",
}

INTENT_MAP = {
    "identity": ["i am", "i'm", "my name", "they call me", "call me", "the name is", "my trade", "my living"],
    "sale": INTENT_KEYS["What do you have for sale?"],
    "quest": ["quest", "task", "help", "need", "work", "do that", "want", "bring", "find", "carried", "clear"],
    "lore": ["story", "stories", "heard", "they say", "tale", "legend", "history",
             "old road", "folk", "mill", "bell", "chapel", "comet", "keep", "winter", "mother swore"],
    "object": INTENT_KEYS["What do you have for sale?"] + ["it is", "this", "that"],
    "place": ["here", "this place", "you stand", "village", "town", "city", "crossing", "friend"],
}


def drift_names(text, blob):
    names = set()
    for sentence in re.split(r"[.!?\n]", text):
        for w in sentence.strip().split():
            w = w.strip('",*#()\'')
            if re.fullmatch(r"[A-Z][a-z]{2,}", w) and w not in COMMON and w not in blob:
                names.add(w)
    return names


def content_words(text):
    return set(w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in {"about", "their", "would", "could", "should", "there", "which", "these", "those", "where", "every"})


def grade(name, card_blob, qname, q, resp):
    flaws = []
    if resp.lstrip().startswith(("Player:", "###")):
        flaws.append("prefix_leak")
    body = resp.strip()
    for m in re.finditer(r"(?:I am|call me|I'm) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", body):
        if m.group(1) not in card_blob:
            flaws.append("persona_swap")
            break
    for m in re.finditer(r"([A-Z][a-z]+) is the name", body):
        if m.group(1) not in card_blob:
            flaws.append("persona_swap")
            break
    if "Description:" in body or "<START>" in body:
        flaws.append("card_continuation")
    low = body.lower()
    if any(t in low for t in TEMPLATE_ECHO):
        flaws.append("template_echo")
    elif name.split()[0] not in body and qname == "identity":
        flaws.append("identity_missed")
    keys = INTENT_MAP[qname]
    if not any(k in body.lower() for k in keys):
        flaws.append("intent_missed")
    if "### Player" in resp or "### System" in resp:
        flaws.append("marker_leak")
    if len(body) < 16:
        flaws.append("too_short")
    words = body.split()
    if len(words) > 8:
        grams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
        if grams.most_common(1)[0][1] > 2:
            flaws.append("repetition")
    if qname == "object":
        card_objs = content_words(card_blob)
        resp_objs = content_words(body)
        if not (card_objs & resp_objs):
            flaws.append("object_ungrounded")
    return flaws


def generate_batch(model, ids, k, tokens, temperature, top_k):
    idx = ids.repeat(k, 1)
    with torch.no_grad():
        for _ in range(tokens):
            logits, _ = model(idx)
            z = logits[:, -1, :] / temperature
            thresh = z.topk(top_k, dim=-1).values[:, -1:]
            z = z.masked_fill(z < thresh, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(torch.softmax(z, dim=-1), 1)], dim=1)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--cards", type=int, default=40)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    rows = [json.loads(l)["text"] for l in open(os.path.join(NPC, "st_conversations.jsonl"), encoding="utf-8") if l.strip()]
    random.shuffle(rows)
    cards = rows[: args.cards]

    histogram = Counter()
    n_resp = 0
    n_fail = 0
    f_data = open(os.path.join(NPC, "st_forge_data.jsonl"), "a", encoding="utf-8")
    f_fail = open(os.path.join(NPC, "st_forge_failures.jsonl"), "a", encoding="utf-8")

    for card in cards:
        header, first = card.split("<START>\n", 1)
        name = first.split(":", 1)[0]
        first_line = first.split("\n", 1)[0]
        blob = header + name
        for qname, q in random.sample(list(QUESTIONS.items()), 2):
            prompt = f"{header}\n<START>\n{first_line}\nPlayer: {q}\n{name}:"
            ids = torch.tensor([tok.encode(prompt).ids], device=device)
            plen = ids.shape[1]
            out = generate_batch(model, ids, args.k, args.tokens, args.temperature, 40)
            for row in out:
                resp = tok.decode(row[plen:].tolist())
                stops = [c for c in (resp.find("\nPlayer:"), resp.find("Player:"), resp.find("### ")) if c >= 0]
                stopped = bool(stops)
                text = resp[: min(stops)] if stops else resp
                flaws = grade(name, blob, qname, q, text)
                if not stopped:
                    flaws.append("no_stop")
                n_resp += 1
                n_fail += bool(flaws)
                for fl in flaws:
                    histogram[fl] += 1
                if flaws:
                    f_fail.write(json.dumps({"name": name, "q": q, "resp": text.strip(), "flaws": flaws}) + "\n")
                else:
                    convo = f"{prompt} {text.strip()}\n"
                    f_data.write(json.dumps({"text": convo}) + "\n")
    f_data.close()
    f_fail.close()

    print(f"\n=== forge dashboard ({n_resp} responses) ===")
    print(f"pass rate: {n_resp - n_fail}/{n_resp} = {(n_resp - n_fail) / max(1, n_resp):.0%}")
    for fl, c in histogram.most_common():
        print(f"  {fl:20s} {c:4d}  {c / n_resp:.0%}")


if __name__ == "__main__":
    main()
