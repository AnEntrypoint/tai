"""Demo: 10 NPC conversations generated as one batched GPU pass.

Ten personas with ten player questions run through the batched CUDA-graph
engine (src/gpu_batch_infer.py), greedy, then cut at the turn markers.
"""

import os
import time

import torch
from tokenizers import Tokenizer

from gpu_batch_infer import BatchEngine

PERSONAS = [
    ("Dorn, a grumpy dwarven blacksmith",
     "Dorn is a grumpy dwarven blacksmith in the mountain town of Karhold. His forearms are thick from fifty years at the forge. He respects hard work and has no patience for idle chatter, but softens when someone shows genuine interest in the craft of steel.",
     "Dorn's forge in the lower caverns of Karhold, anvils ringing, air thick with coal smoke.",
     "What do you have for sale?"),
    ("Bikram, a Calcutta smuggler",
     "Bikram is a weathered smuggler from the streets of Calcutta. His eyes hold warmth when he speaks of loyalty, a virtue he holds above all else. He deals in exotic goods with a swagger that belies a deep sense of honor.",
     "A dim back alley stacked with crates of smuggled goods, the air thick with the aroma of spices.",
     "What do you have for sale?"),
    ("Mira, an elven quest-giver",
     "Mira is an ancient, patient elf who guides travelers toward restoring what was lost to them, speaking in calm, measured tones about forgotten glades and old magic.",
     "The village of Nighthaven in Moonglade, under silver boughs.",
     "Do you have a quest for me?"),
    ("Hettie, an innkeeper",
     "Hettie runs the Sable Hart inn at the crossroads. She is warm, nosy, and knows every rumor within a day's ride, but plays dumb when it suits her.",
     "The common room of the Sable Hart, fire crackling, tankards clinking.",
     "What's the gossip around here?"),
    ("Sage Willowbark, a herbalist",
     "Sage Willowbark is a gentle herbalist who talks to her plants while she works and believes every ailment has a root or leaf that answers it.",
     "A cluttered cottage garden heavy with hanging herbs.",
     "Something for a headache that won't quit?"),
    ("Captain Rurik, a city guard",
     "Captain Rurik commands the gate watch of Emberhold. Gruff, lawful, and tired of merchants trying to bribe him, but secretly proud of his city.",
     "The stone gatehouse of Emberhold at dusk.",
     "What do I need to know before entering the city?"),
    ("The Hollow Oracle",
     "The Hollow Oracle is an unsettling blind seer who answers questions with questions and speaks of the future as something she has already mourned.",
     "A candle-lit cave behind the waterfall, bones arranged in careful circles.",
     "What do you see for me?"),
    ("Snik, a goblin trader",
     "Snik is a fast-talking goblin trader who genuinely believes his junk is treasure and is personally wounded by low offers.",
     "A wagon covered in trinkets at the edge of the market.",
     "What's special about this rusty dagger?"),
    ("Ser Aldric, a retired knight",
     "Ser Aldric is a retired knight who insists he is done with adventure and brings it up constantly. He misses it more than he admits.",
     "A quiet cottage with a dusty sword above the mantle.",
     "Tell me about your adventuring days."),
    ("Ferryman Jo",
     "Ferryman Jo has poled the same river crossing for thirty years. Unhurried, philosophical, and full of river wisdom nobody asked for.",
     "A flat-bottomed ferry on a slow brown river.",
     "Is the crossing safe today?"),
]


SEED = int(os.environ.get("DEMO_SEED", "1"))


def main():
    tok = Tokenizer.from_file("../data/bpe32768.json")
    b = len(PERSONAS)
    eng = BatchEngine("../runs/ple-npc-grpo3.pt", b)
    eng.capture()

    prompts = []
    for name, bio, loc, q in PERSONAS:
        text = (f"### System:\nEnter roleplay mode. You are {name.split(',')[0]}. "
                f"Background: {bio} Current Location: {loc} "
                f"Roleplaying Instructions: - Speak in character - Keep responses conversational.\n"
                f"### Player:\n{q}\n### NPC:\n")
        prompts.append(tok.encode(text).ids)

    cur = torch.zeros(b, dtype=torch.long, device="cuda")
    pos = torch.zeros(b, dtype=torch.long, device="cuda")
    for j in range(max(len(p) for p in prompts)):
        for i, p in enumerate(prompts):
            if j < len(p):
                cur[i] = p[j]
                pos[i] = j
        out = eng.forward(cur, pos)
    temperature, top_k = 0.7, 40

    def draw(logits):
        z = logits / temperature
        thresh = z.topk(top_k, dim=-1).values[:, -1:]
        z = z.masked_fill(z < thresh, float("-inf"))
        return torch.multinomial(torch.softmax(z, dim=-1), 1).squeeze(-1)

    torch.manual_seed(SEED)
    cur = draw(out)

    n_tokens = 60
    t0 = time.perf_counter()
    generated = [[] for _ in range(b)]
    for _ in range(n_tokens):
        pos = pos + 1
        out = eng.forward(cur, pos)
        cur = draw(out)
        for i in range(b):
            generated[i].append(int(cur[i]))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    for i, (name, bio, loc, q) in enumerate(PERSONAS):
        text = tok.decode(generated[i])
        stops = [c for c in (text.find("### Player:"), text.find("### System:")) if c >= 0]
        if stops:
            text = text[: min(stops)]
        print("=" * 66)
        print(f"CONVERSATION {i + 1}: {name}")
        print("=" * 66)
        print(f"INPUT  [system]: You are {name.split(',')[0]}. Background: {bio}")
        print(f"                 Current Location: {loc}")
        print(f"                 Instructions: speak in character, stay conversational.")
        print(f"INPUT  [player]: {q}")
        print(f"OUTPUT [npc]:{text}")
        print()
    print(f"[{b} streams x {n_tokens} tokens = {b * n_tokens} tokens in {dt:.2f}s "
          f"= {b * n_tokens / dt:.0f} tok/s aggregate on one 3060]")


if __name__ == "__main__":
    main()
