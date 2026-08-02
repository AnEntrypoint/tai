"""Adherence metrics for the NPC persona probes.

For each fixed persona/question pair (same probes as npc_eval.py), generate a
turn and score:
  name_drift  -- a capitalized name appears that is nowhere in the system prompt
  intent      -- the response engages the question's category (keyword families)
  clean_stop  -- generation ended at a turn marker, not the token budget

Reported as rates over all probes, per checkpoint, so rounds can be compared
by numbers instead of vibes.
"""

import argparse
import os
import re

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from npc_eval import PERSONAS, QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "data", "bpe32768.json")

COMMON = {
    "I", "The", "A", "An", "He", "She", "It", "You", "We", "They", "But", "And",
    "Or", "If", "Ah", "Oh", "Yes", "No", "Well", "Good", "Safe", "Welcome",
    "Greetings", "Farewell", "In", "On", "At", "To", "For", "Of", "With",
    "From", "By", "As", "Is", "Are", "Was", "Were", "Be", "Been", "Do", "Does",
    "Did", "Have", "Has", "Had", "My", "Your", "His", "Her", "Their", "Our",
    "This", "That", "These", "Those", "There", "Here", "What", "When", "Where",
    "Who", "Why", "How", "Please", "Tell", "Let", "Now", "Then", "So", "Very",
    "Much", "Many", "More", "Most", "Some", "Any", "All", "One", "Two", "Not",
    "Just", "Only", "Even", "Also", "Still", "Yet", "Again", "Once", "Ever",
    "Never", "Always", "Sometimes", "Perhaps", "Maybe", "Indeed", "Truly",
}

INTENT_KEYS = {
    "Hello there.": ["welcome", "well met", "greetings", "hello", "friend", "traveler", "visitor"],
    "What do you have for sale?": ["sale", "sell", "wares", "buy", "coin", "gold", "price", "offer", "goods", "stock", "deal", "trade"],
    "Tell me about yourself.": ["i am", "i'm", "my name", "they call me", "i have", "i've", "my life", "my work", "my family"],
}

TEMPLATE_ECHO = ("i deal in what this place provides",
                 "say what you need and i will name a price",
                 "you are a new friend",
                 "they say you ",
                 "the user is seeking")


def drift_names(text, system_blob):
    names = set()
    for sentence in re.split(r"[.!?\n]", text):
        words = sentence.strip().split()
        for w in words[1:]:
            w = w.strip('",*#()\'')
            if re.fullmatch(r"[A-Z][a-z]{2,}", w) and w not in COMMON and w not in system_blob:
                names.add(w)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(TOK)

    n = drift = intent_ok = clean = 0
    for i, bio in enumerate(PERSONAS):
        system = f"### System:\nYou are an NPC. {bio}\n"
        for q in QUESTIONS:
            prompt = system + f"### Player:\n{q}\n### NPC:\n"
            ids = torch.tensor([tok.encode(prompt).ids])
            out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=40)[0].tolist()
            raw = tok.decode(out[len(ids[0]):])
            stops = [c for c in (raw.find("### Player:"), raw.find("### System:")) if c >= 0]
            stopped = bool(stops)
            text = raw[: min(stops)] if stops else raw
            n += 1
            clean += stopped
            d = drift_names(text, bio)
            drift += bool(d)
            keys = INTENT_KEYS.get(q)
            if keys is None or any(k in text.lower() for k in keys):
                intent_ok += 1
            if d:
                print(f"  drift p{i + 1} q={q!r}: {sorted(d)[:4]} :: {text.strip()[:80]!r}")
    print(f"\nprobes: {n}")
    print(f"name-drift rate : {drift}/{n} = {drift / n:.0%}")
    print(f"intent rate     : {intent_ok}/{n} = {intent_ok / n:.0%}")
    print(f"clean-stop rate : {clean}/{n} = {clean / n:.0%}")


if __name__ == "__main__":
    main()
