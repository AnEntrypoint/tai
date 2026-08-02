"""Programmatic world database -> grounded NPC dialogues.

Every content answer in these conversations cites a real row of the DB:
actual items with prices and lore hooks, actual quests (from the dprashar
quest pool), actual places and their connections. The model learns that
"what do you have for sale" is answered with named stock at named prices,
that quests have objectives and destinations, and that places have exits.
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "st_world.jsonl")

random.seed(23)

ITEMS = {
    "blacksmith": [
        ("Frostbrand blade", 240, "holds an edge even when the whetstone freezes", "pulled from a duelist's grave in the Icemark, or so the last owner swore"),
        ("Gatehold hammer", 95, "one-piece ash and iron, will not split at the head", "the same pattern the gate wardens carry, minus the crest"),
        ("Ringmail vest", 180, "river-steel rings, quiet as mail gets", "made for a scout captain who paid in advance and never came back"),
        ("Horseshoe set, mountain grade", 22, "clipped for scree and ice", "the farrier at the pass orders them by the hundred"),
        ("Splitting axe", 48, "bit wider than my palm, sings when it's sharp", "good for firewood and bar doors, in that order"),
        ("Boar spear", 76, "cross-pinned so the boar can't run up the shaft", "the crossbar saved a hunter's leg last winter, ask me how I know"),
        ("Razor wire, ten yards", 60, "for fences, traps, and arguments", "I ask no questions when I sell it, and I remember no faces"),
        ("Anvil-made nails, box of fifty", 9, "square-shanked, they don't work loose", "roofers swear by them and at them"),
    ],
    "alchemist": [
        ("Firebelly tonic", 38, "warms you for an hour, tingles for two", "pepper-root and ember oil, same recipe that saved the north watch"),
        ("True-sight drops", 120, "one drop per eye, shows illusions as smoke", "customs men buy them in threes and never say why"),
        ("Dreamless vial", 55, "eight hours without a single dream", "the lighthouse keeper swears by it during storm weeks"),
        ("Mender's salve", 26, "closes cuts in a day, stings like honesty", "boiled willow and silverleaf, nothing exotic"),
        ("Quiet-water", 44, "a sip slows a racing heart", "the stage performers down the road use it before curtain"),
        ("Antidote, general", 90, "works on most common venoms and all uncommon hangovers", "keep one in your boot and pray you forget it's there"),
    ],
    "herbalist": [
        ("Willow bark", 4, "chew it slow for pain, it tastes like a grudge", "every mother in the valley knows this one"),
        ("Vervain twist", 6, "for shoulders that carry the week", "I pick it myself behind the cottage"),
        ("Feverfew bundle", 5, "steep it for the aches that come with rain", "the bees leave it alone and so do I"),
        ("Salt-thyme honey", 14, "pale, finishes with a little brine", "from the Merrow cliffs, the only place it grows"),
        ("Dream-mint", 9, "a leaf under the tongue for sleep", "the night watch asks for it by name"),
        ("Woundwort", 7, "press it on a cut, keep it there a day", "older than the village, by the old people's count"),
    ],
    "provisioner": [
        ("Hard cheese wheel", 18, "travels a month and only improves", "from the Brannock caves, aged in the dark"),
        ("Salt pork side", 26, "feeds four on the road for a week", "cured the way my father taught me"),
        ("Waybread, dozen", 8, "one keeps you walking till supper", "baked hard enough to break a tooth, soft enough to earn the name"),
        ("Dried plums, sack", 10, "winter in a bag", "the orchard behind the inn, same trees"),
        ("River trout, smoked", 15, "caught upstream this week", "Ferryman Jo's brother does the smoking"),
    ],
    "trinket-dealer": [
        ("Whisper lamp", 5, "an ordinary oil lamp with an extraordinary story", "Snik certified, whatever that is worth to you"),
        ("Lucky bone whistle", 3, "makes a sound dogs hate and goblins love", "found in a barrow, cleaned thoroughly, mostly"),
        ("Map of the Ashlands", 12, "shows both roads and one that isn't there", "the cartographer's apprentice owes me money"),
        ("Compass, brass", 30, "points north except near the old shrine, where it gets shy", "nobody has explained that to me yet"),
        ("Key shaped like a dagger", 10, "opens certain old sea-chests", "the late Old Kree's, and it's a long story for the price of a beer"),
    ],
}

SHOPKEEPERS = {
    "blacksmith": ("Dorn", "a grumpy dwarven blacksmith of Karhold, fifty years at the forge, no patience for chatter but endless patience for steel"),
    "alchemist": ("Magistra Vool", "a brilliant, beleaguered alchemist who runs a failing college and despises amateurs"),
    "herbalist": ("Sage Willowbark", "a gentle herbalist who talks to her plants and believes every ailment has a leaf that answers it"),
    "provisioner": ("Hettie", "a warm, nosy provisioner who knows every rumor within a day's ride"),
    "trinket-dealer": ("Snik", "a fast-talking goblin trader who believes his junk is treasure and takes low offers personally"),
}

PLACES = [
    ("Karhold", "a mountain town of forges and coal smoke", ["the pass north", "Emberhold by the low road"]),
    ("Emberhold", "a walled city with a bell tower and two markets", ["Karhold", "the river crossing"]),
    ("Nighthaven", "a quiet elven village under silver boughs", ["the Echoing Glade", "Moonglade proper"]),
    ("the Echoing Glade", "a hidden clearing where the veil runs thin", ["Nighthaven"]),
    ("Merrow Point", "cliff farms and salt-thyme hives over the sea", ["the cliff road south"]),
    ("Brannock", "a trading city on black water, famous for its sewers and its cheese", ["the under-docks", "the high road"]),
]

QUEST_SOURCE = os.path.join(NPC, "dprashar-output.json")


def price_str(p):
    return f"{p} gold"


def shop_convo(shop, rng):
    keeper, desc = SHOPKEEPERS[shop]
    items = rng.sample(ITEMS[shop], 3)
    star = items[0]
    lines = [
        f"Description: {desc}.",
        f"Personality: plainspoken, busy, knows every item on the shelves.",
        f"Scenario: {keeper}'s shop, shelves stocked, the day's trade underway.",
        "<START>",
        f"{keeper}: *looks up from the counter* {rng.choice(['Morning, if it is one.', 'In or out, the door costs heat.', 'Welcome. Mind the step.'])} Say what you need.",
        f"Player: What do you have for sale?",
        (f"{keeper}: Depends on your coin. {star[0]}, {price_str(star[1])} -- {star[2]}. "
         f"There's {items[1][0]} at {price_str(items[1][1])}, and {items[2][0]} for {price_str(items[2][1])} "
         f"if your purse is light. No credit, no exceptions, no whining."),
        f"Player: Tell me about the {star[0]}.",
        f"{keeper}: *a short nod, almost approval* {star[3].capitalize()}. That's what I know, and I don't pad the truth to sell it. {price_str(star[1])}, firm.",
        f"Player: I'll take it.",
        f"{keeper}: {rng.choice(['*already wrapping* Sensible.', '*counts the coins twice* Pleasure doing business.', '*hands it over with both hands* Care for it and it outlives you.'])} Anything else, or are we square?",
    ]
    return "\n".join(lines) + "\n"


def quest_convo(quest, rng):
    giver = rng.choice(list(SHOPKEEPERS.values()))[0]
    lines = [
        f"Description: {giver}, a quest-giver in a fantasy world with work that needs doing.",
        f"Scenario: {giver} has a task on offer, and is weighing whether you can be trusted with it.",
        "<START>",
        f"{giver}: *studies you a moment before deciding* You've the look of someone between jobs. Good. I have one, and it's called {quest['Title']}.",
        f"Player: Do you have a quest for me?",
        f"{giver}: {quest['Text'].strip()} What do you say?",
        f"Player: What do I need to do?",
        f"{giver}: Plainly put: {quest['Objective']} Nothing fancier than that, and nothing less.",
        f"Player: And the reward?",
        f"{giver}: {rng.choice(['Coin, and my name behind yours, which spends better in some rooms than gold.', 'Honest pay for honest work, and a favor owed, which lasts longer.', 'Enough. And I say that as someone who has watched people die for less.'])}",
    ]
    return "\n".join(lines) + "\n"


def place_convo(place, rng):
    name, desc, exits = place
    guide = rng.choice(list(SHOPKEEPERS.values()))[0]
    lines = [
        f"Description: {guide}, who knows the roads and answers plainly.",
        f"Scenario: A traveler asking directions at a roadside stop.",
        "<START>",
        f"Player: Where are we?",
        f"{guide}: {name} -- {desc}. {rng.choice(['Mind your step and it treats folk well enough.', 'Not the prettiest stop on the road, but an honest one.', 'Keep your coin close and it keeps you safe enough.'])}",
        f"Player: How do I get to {exits[0]}?",
        f"{guide}: {rng.choice(['Follow the markers', 'Take the road out past the last building', 'Head out with the morning carts'])} toward {exits[0]}. "
        + (f"From there you can make {exits[1]} if the weather holds." if len(exits) > 1 else "The road ends there, so make the trip count."),
        f"Player: Anything to watch out for?",
        f"{guide}: {rng.choice(['The weather turns fast; take a coat you trust.', 'Keep to the marked road after dark and you will be fine.', 'Tolls are fair; the men offering shortcuts are not.'])}",
    ]
    return "\n".join(lines) + "\n"


def object_convo(shop, rng):
    keeper, desc = SHOPKEEPERS[shop]
    item = rng.choice(ITEMS[shop])
    lines = [
        f"Description: {desc}.",
        f"Scenario: {keeper}'s shop; a particular item sits on the table between you.",
        "<START>",
        f"Player: Is that thing on your table for sale?",
        f"{keeper}: *follows your gaze* That? That's the {item[0]}. {item[2].capitalize()}. And yes, {price_str(item[1])} -- it is.",
        f"Player: What's its story?",
        f"{keeper}: {item[3].capitalize()}. That's the honest version, and I charge nothing for it. The item itself is {price_str(item[1])}.",
        f"Player: I'll think about it.",
        f"{keeper}: {rng.choice(['Think slow. It has nowhere else to be, and neither have I.', 'It will be here tomorrow if your coin is.', 'Thinking is free. The item is not.'])}",
    ]
    return "\n".join(lines) + "\n"


def main():
    quests = json.load(open(QUEST_SOURCE, encoding="utf-8"))
    rng = random.Random(23)
    out = []
    for shop in ITEMS:
        for _ in range(40):
            out.append(shop_convo(shop, rng))
        for _ in range(20):
            out.append(object_convo(shop, rng))
    for q in rng.sample(quests, 400):
        out.append(quest_convo(q, rng))
    for _ in range(200):
        out.append(place_convo(rng.choice(PLACES), rng))
    rng.shuffle(out)
    with open(OUT, "w", encoding="utf-8") as f:
        for convo in out:
            f.write(json.dumps({"text": convo}) + "\n")
    print(f"wrote {len(out)} DB-grounded conversations to {OUT}")


if __name__ == "__main__":
    main()
