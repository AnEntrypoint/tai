"""Build ST-format card conversations from the CharacterCodex pool.

Every NPC response is assembled combinatorially -- opener x grounding x
closer drawn from independent first-person banks, grounded with real goods
from the world DB ITEMS table, a trade detected from the card description,
and place names. No raw card text is spliced into a response: second-person
scenario prose in NPC mouths was the mode-collapse attractor ("I deal in
what this place provides. You are a new friend who...") that this rewrite
removes. Variety per intent is in the hundreds of unique skeletons instead
of two or three fixed templates.
"""

import json
import os
import random
import re

from st_world import ITEMS

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
    "*wipes their hands on a cloth* {greet}. {name}. Mind where you step.",
    "{greet} {name} here. You picked a fine hour to come in out of the road.",
    "*marks their place and looks up* {greet} I am {name}. Take your time.",
    "{greet} I am {name}, and this is my corner of the world. What do you need?",
]

IDENTITY_OPEN = [
    "I am {name}.",
    "They call me {name}.",
    "{name}, if we are doing names.",
    "The name is {name}.",
]
IDENTITY_MID = [
    "I make my living as {trade} here in {place}.",
    "{place} is home, and my work as {trade} keeps me busy.",
    "Most folk in {place} know me as {trade}, and that suits me.",
    "I have been {trade} in these parts longer than I care to count.",
    "My trade is {trade}, my town is {place}, and both suit me fine.",
    "Ask around {place} and they will point you to me for {trade} work.",
]
IDENTITY_END = [
    "Ask what you came to ask.",
    "That is the short of it.",
    "Now you know as much as anyone does.",
    "What else is worth knowing?",
]

SALE_OPEN = [
    "Depends on your coin.",
    "A few things worth your coin.",
    "Perhaps, if your purse is honest.",
    "I might. Stock changes with the road.",
    "Everything on the shelf is for sale; most of the floor is not.",
    "For the right coin, nearly anything.",
]
SALE_CLOSE = [
    "No credit, no exceptions.",
    "Have a look -- good stock never sits long.",
    "Make an honest offer and we will get along fine.",
    "Coin up front, and it is yours.",
    "Buy it or admire it, both are free to start.",
]

QUEST_OPEN = [
    "There is something.",
    "Work? Always.",
    "Since you ask -- yes.",
    "Now that you mention it.",
    "Maybe. Depends on your stomach.",
]
QUEST_MID = [
    "A shipment of mine went missing on the {place} road, and I want it back.",
    "Someone has been slipping into my stores at night, and I want a name.",
    "I need a letter carried to {place} and an answer brought back.",
    "An old partner owes me coin and an apology; I would settle for either.",
    "Rats the size of dogs have taken the cellar, and I want it cleared.",
    "A rival has been buying up my stock to starve me out; find who funds them.",
    "The well on the east side has gone foul, and someone should look into it.",
    "Bandits took a crate meant for the temple; bring it home.",
]
QUEST_CLOSE = [
    "Help with it and you will not leave empty-handed.",
    "Do that for me and the road will remember your name.",
    "It is not glorious, but it pays in more than coin.",
    "Say yes and I will tell you the rest.",
]

PLACE_OPEN = [
    "You stand in {place}.",
    "{place}, friend.",
    "This is {place}.",
    "{place} -- for better or worse.",
]
PLACE_MID = [
    "Keep your wits and it treats folk well enough.",
    "Not the safest road, but honest enough if you are.",
    "Mind the dark corners after sunset.",
    "The work is hard and the people harder.",
    "Quiet enough if you keep to yourself.",
]

LORE_OPEN = [
    "Stories? A few.",
    "They say a lot of things.",
    "I have heard my share.",
    "Sit long enough and everyone tells you everything.",
]
LORE_MID = [
    "The old road through {place} was a king's highway once; you can still find the stones.",
    "Folk still lock their doors here since the fire season, and they are right to.",
    "They found coins under the mill last spring, older than any crown that rules now.",
    "The well water turned sweet the year the comet passed; nobody explains it.",
    "A cart goes missing on the north road every winter, always the last one.",
    "The chapel bell rang by itself the night the old keeper died, or so my mother swore.",
    "Traders from the coast pay double for anything made here, and never say why.",
    "There is a room in the old keep that no key fits; every lord has tried.",
]
LORE_CLOSE = [
    "That is what I can offer from where I stand.",
    "Believe half of it and you will do fine.",
    "Ask someone older and you will get a different ending.",
    "Make of it what you will.",
]

MOOD_OPEN = [
    "Well enough.",
    "Cannot complain.",
    "Busy, and that is a blessing.",
    "Tired, if I am honest.",
]
MOOD_CLOSE = [
    "The days are long but the work is honest.",
    "Ask me something harder and we will see how I am.",
    "The road keeps sending folk, so I keep answering.",
    "Trade is slow, but slow is not stopped.",
]

NPC_FAREWELL = [
    "Safe roads to you, friend. {name} will be here if the road bends back.",
    "Go with care. Doors like mine do not stay closed to travelers like you.",
    "Off with you, then. The work will not do itself.",
    "Fair winds. Come back with coin or questions, both spend here.",
    "Mind the road. And mind yourself on it.",
]

GREETS = ["Well met, stranger.", "Ah, a visitor.", "Welcome, welcome.", "Greetings, friend."]

INTENTS = ["identity", "sale", "quest", "place", "lore", "mood"]

TRADE_KEYS = [
    ("blacksmith", {"smith", "forge", "anvil", "weapon", "armorer", "armorer", "metal", "sword"}),
    ("alchemist", {"alchemist", "alchemy", "potion", "elixir", "apothecary", "reagent", "chemist"}),
    ("herbalist", {"herbalist", "herb", "plant", "remedy", "healer", "midwife", "gardener"}),
    ("provisioner", {"innkeeper", "inn", "tavern", "cook", "baker", "brewer", "merchant", "shopkeeper", "grocer"}),
    ("trinket-dealer", {"trinket", "curio", "peddler", "trader", "collector", "antiquities", "jeweler"}),
]


def trade_of(desc):
    words = set(re.findall(r"[a-z]+", desc.lower()))
    for trade, keys in TRADE_KEYS:
        if words & keys:
            return trade
    return random.choice(list(ITEMS))


def price_str(p):
    return f"{p} gold"


def sale_ground(trade):
    items = random.sample(ITEMS[trade], 2)
    a, b = items
    forms = [
        f"{a[0]}, {price_str(a[1])} -- {a[2]}.",
        f"{a[0]} at {price_str(a[1])}, {a[2]}.",
        f"{a[0]}, {price_str(a[1])} -- {a[2]}. There is also {b[0]} for {price_str(b[1])} if your purse is light.",
        f"{a[0]} at {price_str(a[1])}, and {b[0]} at {price_str(b[1])}. {a[2].capitalize()}.",
    ]
    return random.choice(forms)


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


def first_sentence(text):
    for sep in (". ", "! ", "? "):
        i = text.find(sep)
        if 30 < i < 280:
            return text[: i + 1]
    return text[:260]


def respond(intent, name, desc, place, trade):
    if intent == "identity":
        return " ".join([random.choice(IDENTITY_OPEN).format(name=name),
                         random.choice(IDENTITY_MID).format(trade=trade, place=place),
                         random.choice(IDENTITY_END)])
    if intent == "sale":
        return " ".join([random.choice(SALE_OPEN),
                         sale_ground(trade),
                         random.choice(SALE_CLOSE)])
    if intent == "quest":
        return " ".join([random.choice(QUEST_OPEN),
                         random.choice(QUEST_MID).format(place=place),
                         random.choice(QUEST_CLOSE)])
    if intent == "place":
        return " ".join([random.choice(PLACE_OPEN).format(place=place),
                         random.choice(PLACE_MID)])
    if intent == "lore":
        return " ".join([random.choice(LORE_OPEN),
                         random.choice(LORE_MID).format(place=place),
                         random.choice(LORE_CLOSE)])
    if intent == "mood":
        return " ".join([random.choice(MOOD_OPEN), random.choice(MOOD_CLOSE)])
    raise ValueError(intent)


def card_convo(name, desc, scen, place):
    personality = traits_from(desc)
    trade = trade_of(desc)
    greet = random.choice(GREETS)
    turns = [("npc", random.choice(NPC_FIRST).format(greet=greet, name=name))]
    for intent in random.sample(INTENTS, random.randint(2, 4)):
        q = random.choice(globals()["PLAYER_" + intent.upper()])
        turns += [("user", q), ("npc", respond(intent, name, desc, place, trade))]
    turns += [("user", random.choice(PLAYER_FAREWELL)),
              ("npc", random.choice(NPC_FAREWELL).format(name=name))]
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
