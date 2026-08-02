"""Sample NPC conversations from a checkpoint with fixed personas and questions.

Prints each turn so training rounds can be compared by eye: does the model
greet in character, answer questions, stay coherent across turns?
"""

import argparse
import os

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "data", "bpe32768.json")

PERSONAS = [
    (
        "Bikram is a rough and tough smuggler from the streets of Calcutta. "
        "His weathered face tells a story of hardship, but his eyes hold warmth "
        "when he speaks of loyalty, a virtue he holds above all else. He deals "
        "in exotic goods in a dim back alley thick with the aroma of spices."
    ),
    (
        "Mira is an elven quest-giver in the village of Nighthaven, Moonglade. "
        "Ancient and patient, she guides travelers toward restoring what was "
        "lost to them, speaking in calm, measured tones about forgotten glades "
        "and old magic."
    ),
    (
        "Dorn is a grumpy dwarven blacksmith in the mountain town of Karhold. "
        "He respects hard work and has no patience for idle chatter, but he "
        "softens when someone shows genuine interest in the craft of steel."
    ),
]

QUESTIONS = [
    "Hello there.",
    "What do you have for sale?",
    "Tell me about yourself.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    device = "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TinyLM(Config(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(TOK)

    for i, bio in enumerate(PERSONAS):
        print(f"\n===== persona {i + 1} =====")
        for q in QUESTIONS:
            prompt = f"### System:\nYou are an NPC. {bio}\n### Player:\n{q}\n### NPC:\n"
            ids = torch.tensor([tok.encode(prompt).ids], device=device)
            out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k)
            text = tok.decode(out[0].tolist()[len(ids[0]):])
            stops = [text.find(m) for m in ("### Player:", "### System:")]
            stops = [c for c in stops if c >= 0]
            if stops:
                text = text[: min(stops)]
            print(f"Player: {q}")
            print(f"NPC: {text.strip()}\n")


if __name__ == "__main__":
    main()
