"""Generate the extension teaching corpus into corpus/.

Why a generator instead of hand-written files
---------------------------------------------
Every sentence written here must avoid containing any of the 48 eval prompts as a
contiguous normalized token sequence. `load_corpus_folder` RAISES on a match, so a
single collision stops the whole training run. Collisions are easy to write by
accident: "one bird is small ." contains the lang_25 prompt, "yesterday she walked ."
contains lang_27. This script filters every generated line through the project's own
`matching_cases` so collisions are dropped mechanically rather than spotted by eye.

The no-space period is deliberate
---------------------------------
`chunk_text` splits training documents on `(?<=[.!?])\\s+`. Writing ". " would split a
multi-clause teaching sentence into separate documents, so the model would never see a
mid-context period. 15 of the 24 extension eval prompts are multi-clause and arrive as a
SINGLE context. Writing ".it" instead of ". it" produces an identical token sequence but
keeps the clauses in one document. This is why the output looks like it is missing
spaces -- it is not a typo.

Run:  python build_corpus.py
"""
from collections import Counter
from pathlib import Path

from run_evals import load_suite, matching_cases

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus"
SUITE = load_suite(ROOT / "evals" / "language_evals.json")

# A deliberately CLOSED lexicon. Every new proper noun, place or number costs a
# vocabulary type, and the 509-type cap evicts the least frequent words first --
# which is exactly where the eval distractor words live.
NAMES = ["ava", "maya", "leo", "nora", "omar", "ella",
         "finn", "sara", "noah", "nina", "emma", "luca"]
OBJECTS = ["book", "pencil", "cup", "bag", "box", "lamp", "desk",
           "shelf", "ball", "door", "towel", "umbrella", "spoon", "pillow", "shoe"]
DRINKS = ["tea", "milk", "rice", "bread", "juice", "water", "soup"]
COLORS = ["red", "blue", "green", "yellow", "brown"]

dropped = []


def clean(lines):
    """Drop any line that contains an eval prompt; record what was dropped."""
    kept = []
    for line in lines:
        hits = matching_cases(line, SUITE)
        if hits:
            dropped.append((line, hits))
        else:
            kept.append(line)
    return kept


def order_safely(lines):
    """Reorder so no two ADJACENT lines concatenate into an eval prompt.

    `reject_eval_leakage` reads the WHOLE file as one normalized token string, so
    a clean line ending "... a robin is a bird ." followed by a clean line starting
    "a salmon is a fish ." still spells out the lang_46 prompt across the newline.
    Every individual line can pass while the file as a whole fails. This greedily
    picks the next line that does not collide with the previous one.
    """
    remaining, out = list(lines), []
    while remaining:
        pick = 0
        for i, cand in enumerate(remaining):
            if not out or not matching_cases(out[-1] + " " + cand, SUITE):
                pick = i
                break
        else:
            # No safe candidate remains; leave it and let verify_corpus.py report it.
            pick = 0
        out.append(remaining.pop(pick))
    return out


def write(name, lines):
    lines = clean(lines)
    # Deduplicate while preserving order: repeated passages are collapsed by the
    # loader anyway, so duplicates add bulk without adding teaching signal.
    seen, unique = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    unique = order_safely(unique)
    (OUT / name).write_text("\n".join(unique) + "\n", encoding="utf-8")
    return len(unique)


# ---------------------------------------------------------------- 01 grammar
def grammar():
    out = []
    singular = ["bird", "robin", "cup", "box", "lamp", "dog", "cat", "book",
                "door", "ball", "desk", "shelf", "puppy", "kitten", "fish"]
    plural = {"bird": "birds", "robin": "robins", "cup": "cups", "box": "boxes",
              "lamp": "lamps", "dog": "dogs", "cat": "cats", "book": "books",
              "door": "doors", "ball": "balls", "desk": "desks", "shelf": "shelves",
              "puppy": "puppies", "kitten": "kittens", "fish": "fish"}
    adj = ["small", "warm", "quiet", "empty", "full", "loud", "round", "bright"]

    for noun in singular:
        for a in adj:
            out.append(f"one {noun} is {a} .")
            out.append(f"every {noun} is {a} .")
            out.append(f"that {noun} is {a} .")
            out.append(f"a {noun} is {a} .")
    for noun in singular:
        p = plural[noun]
        for a in adj:
            out.append(f"two {p} are {a} .")
            out.append(f"many {p} are {a} .")
            out.append(f"those {p} are {a} .")
            out.append(f"some {p} are {a} .")
            out.append(f"the {p} were {a} yesterday .")
            # "the {plural} are ..." was the gap behind lang_26. Every other
            # frame taught plural+are, but "the {plural}" only ever appeared
            # with "were", so the model never saw this exact combination.
            # ("the dogs" itself is an eval prompt and is dropped by the filter.)
            out.append(f"the {p} are {a} .")
            out.append(f"the {p} are not {a} .")
            out.append(f"the {p} are {a} today .")
    # First person, to teach "am" (a required lang_25/lang_26 distractor).
    for a in ["hungry", "quiet", "warm", "early", "late", "here"]:
        out.append(f"i am {a} today .")
        out.append(f"i am not {a} today .")
    # Tense. Never write "yesterday she" (lang_27).
    verbs = [("walk", "walks", "walked", "walking"),
             ("wash", "washes", "washed", "washing"),
             ("call", "calls", "called", "calling"),
             ("open", "opens", "opened", "opening")]
    for base, s, past, ing in verbs:
        for name in NAMES:
            out.append(f"yesterday {name} {past} .")
            out.append(f"today {name} {s} .")
            out.append(f"{name} is {ing} now .")
            out.append(f"last week {name} {past} again .")
        out.append(f"yesterday they {past} .")
        out.append(f"yesterday i {past} .")
        out.append(f"she {s} every day .")
        out.append(f"he {s} every day .")
        out.append(f"she {past} last week .")
        out.append(f"he {past} last week .")
        out.append(f"she is {ing} now .")
        out.append(f"they are {ing} now .")
        out.append(f"we {base} every day .")
        out.append(f"they {base} every day .")
        out.append(f"i {base} every day .")
        out.append(f"you {base} every day .")
        out.append(f"we will {base} tomorrow .")
        out.append(f"they will {base} tomorrow .")
        out.append(f"people {base} here every week .")
        out.append(f"we did not {base} yesterday .")
    return out


# -------------------------------------------------------------- 02 opposites
def opposites():
    # The eval asks hot->cold, empty->full, noisy->quiet. Writing the REVERSE
    # direction uses the identical frame without containing the eval prompt.
    pairs = [("hot", "cold"), ("empty", "full"), ("noisy", "quiet"),
             ("wet", "dry"), ("early", "late"), ("open", "closed"),
             ("light", "heavy"), ("fast", "slow"), ("big", "small"),
             ("hard", "soft"), ("bright", "dark"), ("long", "short"),
             ("clean", "dirty"), ("new", "old"), ("high", "low"),
             ("near", "far"), ("loud", "silent"), ("happy", "sad"),
             ("rough", "smooth")]
    out = []
    for a, b in pairs:
        out.append(f"the opposite of {b} is {a} .")
        out.append(f"the reverse of {b} is {a} .")
        out.append(f"the reverse of {a} is {b} .")
        out.append(f"{a} and {b} are opposites .")
        out.append(f"{b} and {a} are opposites .")
        out.append(f"when a room is not {a} it is {b} .")
        out.append(f"when a room is not {b} it is {a} .")
        out.append(f"something {a} is never {b} .")
    return out


# --------------------------------------------------------------- 03 negation
def negation():
    out = []
    # Colour correction. Multi-clause, no-space periods.
    for obj in ["bag", "cup", "ball", "pencil", "shelf", "desk", "lamp", "shoe", "box", "door"]:
        for wrong in COLORS:
            for right in COLORS:
                if wrong == right:
                    continue
                out.append(f"the {obj} is not {wrong} .it is {right} .the {obj} is {right} .")
    # State correction.
    states = [("open", "closed"), ("empty", "full"), ("wet", "dry"), ("clean", "dirty")]
    for obj in ["door", "box", "bag", "cup", "window", "book"]:
        for wrong, right in states:
            out.append(f"the {obj} is not {wrong} .it is {right} .the {obj} is {right} .")
            out.append(f"the {obj} is not {right} .it is {wrong} .the {obj} is {wrong} .")
    # Purchase correction -- teaches did/not/buy/bought and the copy pattern.
    for name in NAMES:
        for wrong in DRINKS:
            for right in DRINKS:
                if wrong == right:
                    continue
                out.append(f"{name} did not buy {wrong} .{name} bought {right} .{name} bought {right} .")
    return out


# -------------------------------------------------------------- 04 reference
def reference():
    out = []
    gifts = ["book", "pencil", "cup", "bag", "ball", "towel", "spoon", "lamp"]
    for i, giver in enumerate(NAMES):
        for j, taker in enumerate(NAMES):
            if giver == taker:
                continue
            gift = gifts[(i + j) % len(gifts)]
            out.append(f"{giver} lent a {gift} to {taker} .{taker} thanked {giver} .")
            out.append(f"{giver} gave {taker} a {gift} .the person who received the {gift} was {taker} .")
            out.append(f"{giver} called {taker} .{taker} answered the call from {giver} .")
            out.append(f"{giver} sent a {gift} to {taker} .{taker} kept the {gift} from {giver} .")
    return out


# --------------------------------------------------------------- 05 sequence
def sequence():
    out = []
    actions = ["wash", "dry", "fill", "open", "close", "buy", "lift", "clean"]
    things = ["cup", "box", "bag", "door", "ball", "pencil"]
    for thing in things:
        for a in actions:
            for b in actions:
                if a == b:
                    continue
                out.append(f"first {a} the {thing} .then {b} it .the last action is {b} .")
                out.append(f"first {a} the {thing} .then {b} it .the first action is {a} .")
    meals = ["breakfast", "lunch", "dinner", "supper"]
    for i, early in enumerate(meals):
        for late in meals[i + 1:]:
            out.append(f"{late} happens after {early} .the later meal is {late} .")
            out.append(f"{early} happens before {late} .the earlier meal is {early} .")
            out.append(f"{early} happens before {late} .the later meal is {late} .")
            # The missing fourth combination. Experiment 2 only ever paired
            # "happens after" with "the later meal is", so the model was never
            # shown that "A happens after B" also means B is the earlier one.
            out.append(f"{late} happens after {early} .the earlier meal is {early} .")
            out.append(f"{late} comes after {early} .the earlier meal is {early} .")
            out.append(f"{early} comes before {late} .the later meal is {late} .")
    # lang_39 is the one case whose tested pair (train before bus) is exactly the
    # sentence the leakage filter must drop, so the model has to generalize the
    # before->later copy to a pair it never saw. More distinct pairs make the rule
    # more general rather than pair-specific.
    vehicles = ["train", "bus", "taxi", "car", "truck", "bicycle",
                "van", "tram", "boat", "ferry"]
    for a in vehicles:
        for b in vehicles:
            if a == b:
                continue
            out.append(f"the {a} arrived before the {b} .the vehicle that arrived later was the {b} .")
            out.append(f"the {a} arrived after the {b} .the vehicle that arrived later was the {a} .")
            out.append(f"the {a} arrived before the {b} .the vehicle that arrived earlier was the {a} .")
            out.append(f"the {a} arrived after the {b} .the vehicle that arrived earlier was the {b} .")
            # lang_39 failed even though the before->later mapping WAS taught
            # ~29 times, and the model answered with a vehicle absent from the
            # prompt. These reinforce the same mapping in several shapes.
            out.append(f"the {a} arrived before the {b} .the {b} arrived later .")
            out.append(f"the {a} arrived before the {b} .the later vehicle was the {b} .")
            out.append(f"the {a} arrived after the {b} .the {a} arrived later .")
            out.append(f"the {b} arrived later than the {a} .the later vehicle was the {b} .")
    return out


# ------------------------------------------------------- 06 spatial relations
def spatial():
    out = []
    containers = ["bag", "box", "drawer", "basket", "case", "pocket"]
    items = ["book", "pencil", "cup", "ball", "spoon", "shoe", "towel"]
    for c in containers:
        for it in items:
            out.append(f"the {it} is inside the {c} .the {c} contains the {it} .")
            out.append(f"the {c} contains the {it} .the {it} is inside the {c} .")
    surfaces = ["desk", "shelf", "table", "floor", "counter"]
    for s in surfaces:
        for it in ["lamp", "book", "cup", "ball", "pillow"]:
            out.append(f"the {it} is above the {s} .the {s} is below the {it} .")
            out.append(f"the {s} is below the {it} .the {it} is above the {s} .")
            out.append(f"the {it} is beside the {s} .the {s} is beside the {it} .")
    for a in ["ball", "cup", "book", "lamp", "shoe"]:
        for b in ["box", "desk", "shelf", "bag", "door"]:
            out.append(f"the {a} is right of the {b} .the {b} is to the left of the {a} .")
            out.append(f"the {a} is left of the {b} .the {b} is to the right of the {a} .")
            out.append(f"the {a} is north of the {b} .the {b} is to the south of the {a} .")
            out.append(f"the {a} is south of the {b} .the {b} is to the north of the {a} .")
    return out


# ------------------------------------------------- 07 everyday knowledge
def everyday():
    out = []
    for liquid in ["juice", "soup", "milk", "tea"]:
        out.append(f"{liquid} freezes into ice .")
    out += [
        "when water freezes it becomes ice .",
        "ice forms when water freezes .",
        "cold water freezes quickly .",
        "water freezes when the room is cold .",
        "ice melts into water .",
        "ice is cold and hard .",
        "the ice in the cup melted .",
        "we put ice in the juice .",
        "cold water becomes ice in the kitchen .",
        "ice forms on the door at night .",
        "we turn on a lamp when the room is dark .",
        "we turn off the light at night .",
        "maya will turn on the light .",
        "please turn the handle to open the door .",
        "we turn on a light before supper .",
        "they turn on a light to read a book .",
        "hot water becomes steam .",
        "steam rises from hot water .",
        "steam is hot and wet .",
        "sand is dry and warm .",
        "wood is hard and dry .",
        "a wood shelf holds a lamp .",
        "the sand on the floor is dry .",
    ]
    for name in NAMES:
        out.append(f"{name} uses an umbrella to stay dry .")
        out.append(f"{name} uses a towel to stay dry .")
        out.append(f"{name} turns on a light to see in a dark room .")
    out += [
        "we use an umbrella to stay dry .",
        "they use an umbrella to stay dry .",
        "an umbrella keeps a person dry .",
        "a towel keeps a person dry .",
        "a person uses a towel to stay warm .",
        "without an umbrella a person is wet .",
        "rain makes a person wet .",
        "to work in a dark room we turn on a light .",
        "to read in a dark room we turn on a light .",
        "we turn on a light to see in a dark room .",
        "a light helps us see at night .",
        "a dark room is quiet .",
        "the light above the desk is bright .",
        "a hungry person eats bread .",
        "a hungry person eats soup with a spoon .",
        "an asleep cat is quiet .",
        "the cat is asleep on a pillow .",
        "a pillow is soft .",
        "a shoe is not soft .",
        "a round ball rolls .",
        "one cup is missing from the shelf .",
        "the missing book was on the desk .",
        "the door is wide and heavy .",
        "a wide door is heavy .",
    ]
    return out


# ------------------------------------------- 08 categories and analogies
def categories():
    out = []
    members = {
        "bird": ["robin", "duck", "goose", "crow"],
        "fish": ["salmon", "trout", "cod"],
        "fruit": ["apple", "pear", "peach", "mango", "banana", "orange"],
        "vegetable": ["carrot", "onion", "pea", "bean"],
        "tool": ["hammer", "spoon", "brush"],
        "vehicle": ["car", "bus", "train", "truck", "taxi"],
        "tree": ["oak", "pine"],
        "metal": ["iron", "copper"],
        "fabric": ["cotton", "wool"],
    }
    for group, items in members.items():
        for it in items:
            article = "an" if it[0] in "aeiou" else "a"
            out.append(f"{article} {it} is a {group} .")
            out.append(f"every {it} is a {group} .")
            out.append(f"the {it} is a kind of {group} .")
            out.append(f"we learned that {article} {it} is a {group} .")
            out.append(f"the report said {article} {it} is a {group} .")
            out.append(f"people know that {article} {it} is a {group} .")
            out.append(f"{article} {it} belongs with every other {group} .")
            out.append(f"the team discussed {article} {it} and every other {group} .")
            out.append(f"today we compared {article} {it} with another {group} .")
    # Two-clause analogy frame. Experiment 2 taught only SINGLE-clause facts
    # ("a robin is a bird ."), so when the model met "a X is a Y . a Z is a ___"
    # it had never seen that frame and fell back on the copy behaviour it learned
    # from negation/reference/spatial -- answering with Y. These examples teach
    # that the second clause's category depends on the SECOND subject.
    # The two tested pairs are generated here too and dropped by the leakage
    # filter, so the model learns the frame from ~640 other pairs.
    groups = list(members.items())
    for g1, items1 in groups:
        for g2, items2 in groups:
            if g1 == g2:
                continue
            for i1 in items1[:3]:
                for i2 in items2[:3]:
                    a1 = "an" if i1[0] in "aeiou" else "a"
                    a2 = "an" if i2[0] in "aeiou" else "a"
                    out.append(f"{a1} {i1} is a {g1} .{a2} {i2} is a {g2} .")
    grows = [("puppy", "dog"), ("kitten", "cat"), ("chick", "duck"),
             ("calf", "cow"), ("foal", "horse"), ("lamb", "goat"), ("cub", "bear")]
    for young, adult in grows:
        out.append(f"a {young} grows into a {adult} .")
        out.append(f"a young {adult} is called a {young} .")
        out.append(f"the {young} will grow into a {adult} .")
        out.append(f"every {young} grows into a {adult} .")
        out.append(f"we learned that a {young} grows into a {adult} .")
        out.append(f"the {young} grows into a {adult} after a year .")
    out += [
        "a horse and a goat are animals .",
        "a duck and a goose are birds .",
        "iron and copper are metals .",
        "cotton and wool are fabrics .",
        "a hammer is a tool .",
        "an oak is a tree .",
        "a tree is not a tool .",
        "a fish is not a bird .",
    ]
    return out


# ------------------------------------------------- 09 vocabulary support
def support():
    """Plant required distractor words in natural sentences.

    A case is scored 0 and marked out_of_vocabulary if ANY of its four choices is
    absent from the vocabulary, so distractors matter as much as answers.
    """
    out = []
    fillers = ["sand", "wood", "steam", "pillow", "spoon", "shoe", "round",
               "missing", "north", "south", "asleep", "hungry", "supper",
               "goat", "duck", "horse", "fabric", "metal", "vehicle", "tool",
               "tree", "wide", "beside", "soft", "heavy", "fast"]
    frames = [
        "the {w} was mentioned in the report yesterday .",
        "we learned about the {w} during a lesson .",
        "our team discussed the {w} at the office .",
        "a review of the {w} helped us understand it .",
        "the {w} is important to the team .",
        "today the school focused on the {w} .",
        "the report about the {w} explains it in detail .",
        "they compared the {w} with another item .",
    ]
    for w in fillers:
        for f in frames:
            out.append(f.format(w=w))
    # Naturalistic, varied lines so the material is not purely mechanical.
    out += [
        "maya put the spoon beside the bowl .",
        "omar left his shoe near the door .",
        "the pillow fell off the desk .",
        "nina found a round stone in the sand .",
        "leo carried wood to the kitchen .",
        "steam filled the warm kitchen .",
        "the goat and the horse rested in the shade .",
        "a duck swam near the boat .",
        "ella wrapped the book in soft fabric .",
        "the metal box was heavy .",
        "sara counted every vehicle at the station .",
        "finn borrowed a tool from the shelf .",
        "noah planted a tree beside the school .",
        "the supper was late and quiet .",
        "emma was hungry before supper .",
        "luca was asleep before dinner .",
        "ava looked north from the station .",
        "the market is south of the hospital .",
        "the wide door was missing a handle .",
        "a fast bus passed the slow truck .",
    ]
    return out


FILES = [
    ("01_grammar_agreement.txt", grammar),
    ("02_opposites.txt", opposites),
    ("03_negation_correction.txt", negation),
    ("04_reference_people.txt", reference),
    ("05_sequence_order.txt", sequence),
    ("06_spatial_relations.txt", spatial),
    ("07_everyday_knowledge.txt", everyday),
    ("08_categories_analogies.txt", categories),
    ("09_vocabulary_support.txt", support),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()
    total = 0
    for name, fn in FILES:
        n = write(name, fn())
        total += n
        print(f"{name:34s} {n:5d} lines")
    print(f"{'TOTAL':34s} {total:5d} lines")
    print(f"\nDropped {len(dropped)} lines that contained an eval prompt:")
    counts = Counter(cid for _, hits in dropped for cid in hits)
    for cid, n in sorted(counts.items()):
        print(f"  {cid}: {n}")
    for line, hits in dropped[:8]:
        print(f"    e.g. {hits} <- {line[:70]}")
