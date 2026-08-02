"""Unified SillyTavern-convention data engine for the single-purpose NPC model.

Canonical format (one format for everything -- training, prompts, cards):

    Description: <who they are>
    Personality: <traits>
    Scenario: <where this happens>
    <START>
    <Name>: <first message>
    Player: <question>
    <Name>: <answer>
    ...

The NPC's name is the turn prefix, so name binding is a structural property
of the format, never something the model must recall. At inference, a card
plus "<Name>:" is the whole prompt; generation stops before "Player:".

Sources: NousResearch/CharacterCodex cards, chimbiwide persona blocks, and
random-name substitutions of the same cards (teaches read-the-name, not
memorize-the-name). Writes data/npc/st_conversations.jsonl.
"""

import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "st_conversations.jsonl")

random.seed(19)

SYL1 = ["Bor", "Kal", "Dra", "Ven", "Mor", "Tal", "Ryn", "Gal", "Tha", "Zur",
        "Bel", "Cor", "Dal", "Eri", "Fen", "Gor", "Hel", "Ith", "Jar", "Kel"]
SYL2 = ["wick", "mund", "dor", "ric", "na", "la", "mir", "tha", "gar", "wen",
        "dale", "ford", "helm", "ira", "os", "eth", "ash", "orn", "uel", "ys"]
PLACES = ["Karhold", "Emberhold", "Nighthaven", "the Ashlands", "Brannock",
          "the Silver Coast", "Duskvale", "Thornwick", "the Low Marches"]

PLAYER_GREET = ["Hello.", "Greetings.", "Good day.", "Well met.", "Hi there."]
PLAYER_IDENTITY = ["Who are you?", "Tell me about yourself.", "Your name?"]
PLAYER_SALE = ["What do you have for sale?", "Show me your wares.", "Anything to sell?"]
PLAYER_QUEST = ["Do you have a quest for me?", "Got any work?", "Anything you need done?"]
PLAYER_PLACE = ["Where are we?", "What is this place?"]
PLAYER_LORE = ["Any stories from these parts?", "What should I watch out for around here?"]
PLAYER_MOOD = ["How are you today?", "Busy day?"]
PLAYER_FAREWELL = ["Farewell.", "I should go.", "Goodbye for now."]

NPC_FIRST = [
    "*looks up as you approach* {greet} I am {name}. Speak freely.",
    "*nods in greeting* {greet} {name}, at your service. What brings you through?",
    "*sets aside their work* {greet} I am {name}. Rest a moment and say your business.",
    "{greet} I am {name}. Travelers are always welcome at my door.",
]
NPC_IDENTITY = [
    "I am {name}. {desc}",
    "They call me {name}. {desc}",
    "{name}, if we are doing names. {desc}",
]
NPC_SALE = [
    "A few things worth your coin. {scen} Have a look -- good stock never sits long.",
    "Perhaps. {scen} Make an honest offer and we will get along fine.",
    "I deal in what this place provides. {scen} Say what you need and I will name a price.",
]
NPC_QUEST = [
    "There is something. {scen} Help with it and you will not leave empty-handed.",
    "Work? Always. {scen} Do that for me and the road will remember your name.",
    "Since you ask -- yes. {scen} It is not glorious, but it pays in more than coin.",
]
NPC_PLACE = [
    "You stand in {place}. {scen} Keep your wits and it treats folk well enough.",
    "{place}, friend. {scen} Not the safest road, but honest enough if you are.",
]
NPC_LORE = [
    "Stories? {desc} That is what I can offer from where I stand.",
    "They say {scen} I have seen enough of it to know the truth of it.",
]
NPC_MOOD = [
    "Well enough. {scen} The days are long but the work is honest.",
    "Cannot complain. {scen} Ask me something harder and we will see how I am.",
]
NPC_FAREWELL = [
    "Safe roads to you, friend. {name} will be here if the road bends back.",
    "Go with care. Doors like mine do not stay closed to travelers like you.",
]

GREETS = ["Well met, stranger.", "Ah, a visitor.", "Welcome, welcome.", "Greetings, friend."]

INTENTS = ["identity", "sale", "quest", "place", "lore", "mood"]


def first_sentence(text):
    for sep in (". ", "! ", "? "):
        i = text.find(sep)
        if 30 < i < 280:
            return text[: i + 1]
    return text[:260]


def traits_from(desc):
    words = re.findall(r"[a-z]+", desc.lower())
    pick = [w for w in ("gruff", "patient", "proud", "gentle", "sharp", "loyal",
                        "stubborn", "curious", "weary", "warm", "stern", "sly") if w in words]
    return ", ".join(pick[:3]) if pick else "watchful, plainspoken"


def render_card(name, desc, scen, personality):
    lines = [f"Description: {desc}",
             f"Personality: {personality}",
             f"Scenario: {scen}"]
    return "\n".join(lines)


def card_convo(name, desc, scen, place):
    personality = traits_from(desc)
    greet = random.choice(GREETS)
    turns = [("npc", random.choice(NPC_FIRST).format(greet=greet, name=name))]
    for intent in random.sample(INTENTS, random.randint(2, 4)):
        q = random.choice(globals()["PLAYER_" + intent.upper()])
        a = random.choice(globals()["NPC_" + intent.upper()]).format(
            name=name, desc=desc, scen=scen, place=place)
        turns += [("user", q), ("npc", a)]
    turns += [("user", random.choice(PLAYER_FAREWELL)),
              ("npc", random.choice(NPC_FAREWELL).format(name=name, desc=desc, scen=scen, place=place))]
    lines = [render_card(name, desc, scen, personality), "<START>"]
    for role, text in turns:
        speaker = name if role == "npc" else "Player"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines) + "\n"


def main():
    cards = json.load(open(os.path.join(NPC, "character_codex.json"), encoding="utf-8"))
    random.shuffle(cards)
    out = []
    for card in cards:
        name = card["character_name"]
        desc = card["description"].strip()
        scen = first_sentence(card.get("scenario", "").strip()) or desc
        out.append(card_convo(name, desc, scen, card.get("media_source", "these parts")))
        if random.random() < 0.5:
            out.append(card_convo(name, desc, scen, card.get("media_source", "these parts")))
    for _ in range(15000):
        card = random.choice(cards)
        name = random.choice(SYL1) + random.choice(SYL2)
        desc = card["description"].strip()
        scen = first_sentence(card.get("scenario", "").strip()) or desc
        out.append(card_convo(name, desc, scen, random.choice(PLACES)))
    random.shuffle(out)
    with open(OUT, "w", encoding="utf-8") as f:
        for convo in out:
            f.write(json.dumps({"text": convo}) + "\n")
    print(f"wrote {len(out)} ST conversations to {OUT}")


if __name__ == "__main__":
    main()
