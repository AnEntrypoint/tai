"""Build the single-purpose ST training bins.

Mixture (interleaved, anti-overfit by construction):
  - chimbiwide real roleplay conversations, re-rendered to name-prefix ST
  - authored exemplars (st_authored*.jsonl, all ST features covered)
  - a capped template subset from st_data.py output (binding grammar, ~15%)
  - a TinyStories token slice (~20%) so the model keeps general coherence

Output: data/train_npc.bin + data/val_npc.bin (uint16 + eot).
"""

import json
import os
import re

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")
TOK = os.path.join(DATA, "bpe32768.json")
TEMPLATE_CAP = 6000
TINYSTORIES_TOKENS = 4_000_000


def name_of(system_text):
    m = re.search(r"You are ([A-Z][A-Za-z' -]{1,40}?)[.,]", system_text)
    if m:
        return m.group(1).strip()
    m = re.search(r"Background: ([A-Z][A-Za-z' -]{1,40}?) (?:is|was)", system_text)
    return m.group(1).strip() if m else "NPC"


def rerender_real(row):
    msgs = row["messages"]
    system = ""
    turns = []
    for m in msgs:
        role, content = m.get("role"), m.get("content", "")
        if role == "system" or (not system and role == "user" and "roleplay mode" in content.lower()):
            system = content
            continue
        turns.append((role, content))
    if not system or not turns:
        return None
    name = name_of(system)
    desc = system
    for marker in ("Roleplaying Instructions:", "Roleplay Instructions:"):
        i = desc.find(marker)
        if i > 0:
            desc = desc[:i].rstrip(" .\n")
    lines = [f"Description: {desc}", "<START>"]
    for role, content in turns:
        speaker = name if role != "user" else "Player"
        lines.append(f"{speaker}: {content.strip()}")
    return "\n".join(lines) + "\n"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)




def _bulk_encoder(tok):
    try:
        import gigatoken as gt
        g = gt.Tokenizer(tok)
        return lambda texts: [list(r) for r in g.encode_batch(texts)]
    except Exception:
        return lambda texts: [e.ids for e in tok.encode_batch(texts)]

def main():
    tok = Tokenizer.from_file(TOK)
    encode = _bulk_encoder(tok)
    eot = tok.token_to_id("<|endoftext|>")
    texts = []

    for path in ("npc_dialogue.jsonl", "rpg-quests-dialogue.jsonl"):
        for row in read_jsonl(os.path.join(NPC, path)):
            r = rerender_real(row)
            if r:
                texts.append(r)
    n_real = len(texts)

    for path in ("st_authored.jsonl", "st_authored2.jsonl"):
        for row in read_jsonl(os.path.join(NPC, path)):
            texts.append(row["text"])
    n_auth = len(texts) - n_real

    tmpl = [row["text"] for row in read_jsonl(os.path.join(NPC, "st_conversations.jsonl"))]
    texts.extend(tmpl[:TEMPLATE_CAP])
    print(f"real {n_real} | authored {n_auth} | template {TEMPLATE_CAP} | total {len(texts)}")

    ids = []
    for i, enc in enumerate(encode(texts)):
        ids.extend(enc)
        ids.append(eot)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(texts)}, {len(ids) / 1e6:.1f}M tokens", flush=True)

    ts = np.memmap(os.path.join(DATA, "train_v32768.bin"), dtype=np.uint16, mode="r")
    ids.extend(ts[:TINYSTORIES_TOKENS].tolist())
    print(f"added {TINYSTORIES_TOKENS / 1e6:.1f}M TinyStories tokens; total {len(ids) / 1e6:.1f}M")

    arr = np.array(ids, dtype=np.uint16)
    n_val = max(1, int(len(arr) * 0.005))
    arr[:-n_val].tofile(os.path.join(DATA, "train_npc.bin"))
    arr[-n_val:].tofile(os.path.join(DATA, "val_npc.bin"))
    print(f"train {len(arr) - n_val:,} / val {n_val:,}")


if __name__ == "__main__":
    main()
