"""Convert the downloaded NPC dialog datasets into uint16 train/val bins.

Sources (all chatml or derivable to turns):
  data/npc/npc_dialogue.jsonl        persona roleplay conversations
  data/npc/rpg-quests-dialogue.jsonl quest-giver conversations
  data/npc/dprashar-output.json      quest monologues (Title/Objective/Text)
  data/npc/amaydle-train.parquet     single-turn Q/A with bios

Output format per conversation:
  ### System:\n<persona/location/instructions>\n### Player:\n<user>\n### NPC:\n<assistant>\n...
with the tokenizer's eot id appended after each conversation, matching the
random-window batching in train.py.
"""

import json
import os

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
TOK = os.path.join(HERE, "..", "data", "bpe32768.json")
VAL_FRACTION = 0.005
MAX_CHARS = 32000


def chatml_turns(messages):
    system = None
    turns = []
    for m in messages:
        role, content = m.get("role"), m.get("content", "")
        if role == "system" or (system is None and role == "user" and "roleplay mode" in content.lower()):
            system = content
            continue
        turns.append((role, content))
    return system, turns


def render(system, turns):
    parts = []
    if system:
        parts.append("### System:\n" + system.strip())
    for role, content in turns:
        who = "### Player:" if role == "user" else "### NPC:"
        parts.append(who + "\n" + content.strip())
    return "\n".join(parts) + "\n"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def conversations():
    for path in ("npc_dialogue.jsonl", "rpg-quests-dialogue.jsonl"):
        for row in read_jsonl(os.path.join(NPC, path)):
            system, turns = chatml_turns(row["messages"])
            if system and turns:
                yield render(system, turns)

    for synth in ("synthetic.jsonl", "synthetic_names.jsonl"):
        path = os.path.join(NPC, synth)
        if os.path.exists(path):
            for row in read_jsonl(path):
                yield row["text"]

    rows = json.load(open(os.path.join(NPC, "dprashar-output.json"), encoding="utf-8"))
    for row in rows:
        system = (f"You are a quest-giving NPC in a fantasy world. "
                  f"Quest: {row['Title']}. {row['Objective']}")
        turns = [("user", "Greetings."), ("assistant", row["Text"])]
        yield render(system, turns)

    t = pq.read_table(os.path.join(NPC, "amaydle-train.parquet")).to_pylist()
    for row in t:
        system = f"You are {row['Name']}. {row['Biography']}"
        turns = [("user", row["Query"]), ("assistant", row["Response"])]
        yield render(system, turns)


def main():
    tok = Tokenizer.from_file(TOK)
    eot = tok.token_to_id("<|endoftext|>")
    convos = [c for c in conversations() if 0 < len(c) <= MAX_CHARS]
    print(f"{len(convos)} conversations")

    ids = []
    for i, c in enumerate(convos):
        ids.extend(tok.encode(c).ids)
        ids.append(eot)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(convos)} convos, {len(ids) / 1e6:.1f}M tokens", flush=True)

    arr = np.array(ids, dtype=np.uint16)
    n_val = max(1, int(len(arr) * VAL_FRACTION))
    data = os.path.join(HERE, "..", "data")
    arr[:-n_val].tofile(os.path.join(data, "train_npc.bin"))
    arr[-n_val:].tofile(os.path.join(data, "val_npc.bin"))
    print(f"train {len(arr) - n_val:,} tokens / val {n_val:,} tokens")
    print(f"compression: {sum(len(c) for c in convos) / len(arr):.2f} bytes/token")


if __name__ == "__main__":
    main()
