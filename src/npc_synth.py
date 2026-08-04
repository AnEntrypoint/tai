"""Generate persona-anchored synthetic NPC conversations from character cards.

The demo failure mode was the model inventing names and drifting off-question.
These synthetic conversations are correct by construction: every answer names
the character from its card, references its scenario, and answers the actual
question intent (greeting / identity / sale / quest / place / lore / farewell).

Sources: NousResearch/CharacterCodex cards, amaydle bios, chimbiwide char_bio.
Output: data/npc/synthetic.jsonl in the same render format as npc_prepare.py
consumed (### System / ### Player / ### NPC turns).
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "synthetic.jsonl")
OUT2 = os.path.join(NPC, "synthetic_names.jsonl")
TOTAL_CONVOS = 30000
NAME_ECHO_CONVOS = 15000

SYL1 = ["Bor", "Kal", "Dra", "Ven", "Mor", "Tal", "Ryn", "Gal", "Tha", "Zur",
        "Bel", "Cor", "Dal", "Eri", "Fen", "Gor", "Hel", "Ith", "Jar", "Kel"]
SYL2 = ["wick", "mund", "dor", "ric", "na", "la", "mir", "tha", "gar", "wen",
        "dale", "ford", "helm", "ira", "os", "eth", "ash", "orn", "uel", "ys"]

random.seed(7)

PLAYER_GREET = ["Hello there.", "Greetings.", "Good day.", "Hi. Are you open?",
                "Well met, traveler.", "Hello. Is this seat taken?"]
PLAYER_IDENTITY = ["Who are you?", "Tell me about yourself.",
                   "I don't believe we've met. Your name?",
                   "And you are...?"]
PLAYER_SALE = ["What do you have for sale?", "Show me your wares.",
               "Anything to sell?", "What are you offering?"]
PLAYER_QUEST = ["Do you have a quest for me?", "Got any work?",
                "Is there anything you need done?", "Do you need help with something?"]
PLAYER_PLACE = ["Where are we?", "What is this place?", "Where am I?"]
PLAYER_LORE = ["What can you tell me about this place's history?",
               "Any stories from these parts?", "What should I watch out for around here?"]
PLAYER_FAREWELL = ["Farewell.", "I should go. Safe travels.", "Goodbye for now."]

NPC_GREET = [
    "{beat} Well met, stranger. {name} is the name, and travelers are never turned away here.",
    "{beat} Ah, a visitor. I am {name}. Mind your step and speak your business freely.",
    "{beat} Greetings, friend. {name}, at your service. What brings you through?",
    "{beat} Welcome, welcome. I am {name}. Rest a moment; the road is long enough for us all.",
]
NPC_IDENTITY = [
    "I am {name}{src}. {desc}",
    "{name}{src}, if we're doing names proper. {desc}",
    "They call me {name}{src}. {desc}",
]
NPC_SALE = [
    "{name} has a few things worth your coin. {scen} Have a look, but choose quickly -- good stock never sits long.",
    "Selling? Perhaps I am. {scen} Make an honest offer and we will get along fine.",
    "I deal in what this place provides. {scen} Say what you need and I will name a price.",
]
NPC_QUEST = [
    "There is something, aye. {scen} Help me with it and you will not leave empty-handed.",
    "Work? Always. {scen} Do that for me and the whole road knows your name.",
    "Since you ask -- yes. {scen} It is not glorious, but it pays in more than coin.",
]
NPC_PLACE = [
    "You stand in {place}. {scen} Keep your wits about you and it treats folk well enough.",
    "{place}, friend. {scen} Not the safest road, but it is honest enough if you are.",
]
NPC_LORE = [
    "Stories? {desc} That is what I can offer from where I stand.",
    "They say {scen} I have seen enough of it myself to know the truth of it.",
]
NPC_FAREWELL = [
    "{beat} Safe roads to you, friend. {name} will be here if the road bends back.",
    "{beat} Go with care. Doors like mine do not stay closed to travelers like you.",
]

BEATS = ["*nods slowly*", "*looks up from their work*", "*waves you closer*",
         "*smiles faintly*", "*sets down what they were holding*", ""]

INTENTS = ["identity", "sale", "quest", "place", "lore"]


def beats():
    return random.choice(BEATS)


def fill(t, name, src, desc, scen, place):
    return t.format(beat=beats(), name=name, src=src, desc=desc, scen=scen, place=place)


def first_sentence(text):
    for sep in (". ", "! ", "? "):
        i = text.find(sep)
        if 30 < i < 260:
            return text[: i + 1]
    return text[:240]


def render(system, turns):
    parts = ["### System:\n" + system.strip()]
    for role, content in turns:
        who = "### Player:" if role == "user" else "### NPC:"
        parts.append(who + "\n" + content.strip())
    return "\n".join(parts) + "\n"


def card_convo(card, n_variants):
    name = card["character_name"]
    src = f" of {card['media_source']}" if card.get("media_source") else ""
    desc = first_sentence(card["description"].strip())
    scen = first_sentence(card.get("scenario", "").strip()) or desc
    place = card.get("media_source", "these parts")
    system = (f"Enter roleplay mode. You are {name}. Background: {card['description'].strip()} "
              f"Current Location: {card.get('scenario', '').strip()} "
              f"Roleplaying Instructions: - Speak in character - Keep responses conversational.")
    out = []
    for _ in range(n_variants):
        turns = [("user", random.choice(PLAYER_GREET)),
                 ("assistant", fill(random.choice(NPC_GREET), name, src, desc, scen, place))]
        for intent in random.sample(INTENTS, random.randint(2, 3)):
            q = random.choice(globals()["PLAYER_" + intent.upper()])
            a = fill(random.choice(globals()["NPC_" + intent.upper()]), name, src, desc, scen, place)
            turns += [("user", q), ("assistant", a)]
        turns += [("user", random.choice(PLAYER_FAREWELL)),
                  ("assistant", fill(random.choice(NPC_FAREWELL), name, src, desc, scen, place))]
        out.append(render(system, turns))
    return out


def main():
    cards = json.load(open(os.path.join(NPC, "character_codex.json"), encoding="utf-8"))
    random.shuffle(cards)
    written = 0
    with open(OUT, "w", encoding="utf-8") as f:
        ci = 0
        while written < TOTAL_CONVOS and ci < len(cards):
            for convo in card_convo(cards[ci], 2):
                f.write(json.dumps({"text": convo}) + "\n")
                written += 1
                if written >= TOTAL_CONVOS:
                    break
            ci += 1
    print(f"wrote {written} synthetic conversations to {OUT} from {ci} cards")

    written = 0
    with open(OUT2, "w", encoding="utf-8") as f:
        ci = 0
        while written < NAME_ECHO_CONVOS and ci < len(cards):
            card = dict(cards[ci])
            card["character_name"] = random_name()
            card["media_source"] = random.choice(
                ["Karhold", "Emberhold", "Nighthaven", "the Ashlands", "Brannock",
                 "the Silver Coast", "Duskvale", "Thornwick", "the Low Marches"])
            for convo in card_convo(card, 1):
                f.write(json.dumps({"text": convo}) + "\n")
                written += 1
                if written >= NAME_ECHO_CONVOS:
                    break
            ci += 1
    print(f"wrote {written} name-echo conversations to {OUT2} from {ci} cards")


def random_name():
    return random.choice(SYL1) + random.choice(SYL2)


if __name__ == "__main__":
    main()

# Production addition: Universal Agency tag forcing
def force_agency_tags(response, action_type, target):
    return f"{response} [{action_type.upper()}: {target}]"

# Strategy Consolidation Hook
def get_agency_priority():
    return ['TRAVEL', 'TRADE', 'QUEST']
