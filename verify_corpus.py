"""Read-only pre-flight check for the extension corpus. Run BEFORE training.

This reproduces the expensive parts of the training pipeline in about a second, using
the notebook's OWN loader functions (extracted from custom_llm.py by AST, the same
technique test_corpus.py uses) rather than reimplementing them. Nothing here can drift
from the real pipeline.

Because SEED=42 is fixed and no step is stochastic, the vocabulary simulated here is
the EXACT vocabulary the training run will build -- so the out-of-vocabulary table
below is a prediction of the real eval coverage, not an estimate.

Run:  python verify_corpus.py
Exit status 0 means every green-light condition passed.
"""
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from run_evals import (load_suite, matching_cases, reject_eval_leakage,
                       reserve_classroom_passages, validate_corpus_location)

ROOT = Path(__file__).resolve().parent
SUITE = load_suite(ROOT / "evals" / "language_evals.json")

# ---- pull the notebook's own loader functions out of custom_llm.py -------------
tree = ast.parse((ROOT / "custom_llm.py").read_text())
wanted = {"word_tokens", "chunk_text", "load_corpus_folder", "classroom_corpus"}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
namespace = {"Path": Path, "hashlib": hashlib, "re": re,
             "language_suite": SUITE, "reject_eval_leakage": reject_eval_leakage,
             "validate_corpus_location": validate_corpus_location}
exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook-loader", "exec"), namespace)
load_folder = namespace["load_corpus_folder"]
chunk_text = namespace["chunk_text"]
word_tokens = namespace["word_tokens"]
classroom_corpus = namespace["classroom_corpus"]

# These mirror custom_llm.py section 2 / section 4 exactly.
SEED, BLOCK_SIZE, VOCAB_CAP = 42, 48, 509
CORPUS_DIR = ROOT / "corpus"

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


# =============================================================== 1. LEAKAGE
print("\n=== 1. Eval leakage ===")
files = sorted(p for p in CORPUS_DIR.glob("*.txt"))
line_hits, file_hits = [], []
for path in files:
    text = path.read_text(encoding="utf-8")
    # Per line: pinpoints the offending line number.
    for n, line in enumerate(text.splitlines(), 1):
        hits = matching_cases(line, SUITE)
        if hits:
            line_hits.append((path.name, n, hits, line))
    # Per whole file: catches two adjacent lines concatenating into a prompt,
    # which is how reject_eval_leakage actually reads the file.
    hits = matching_cases(text, SUITE)
    if hits:
        file_hits.append((path.name, hits))

check("no eval prompt on any single line", not line_hits,
      "; ".join(f"{f}:{n} {h}" for f, n, h, _ in line_hits[:5]))
check("no eval prompt across the whole file text", not file_hits,
      "; ".join(f"{f} {h}" for f, h in file_hits[:5]))

# The real thing: this is the call the notebook makes, and it raises on a match.
try:
    extension_chunks, manifest = load_folder(str(CORPUS_DIR), BLOCK_SIZE - 1)
    check("load_corpus_folder() accepts the corpus", True,
          f"{len(manifest['files'])} files, {len(extension_chunks)} passages")
except Exception as exc:  # noqa: BLE001 - we want to surface the notebook's own error
    extension_chunks, manifest = [], {"files": []}
    check("load_corpus_folder() accepts the corpus", False, str(exc))

# Per chunk, after chunking.
chunk_hits = [c for c in extension_chunks if matching_cases(c, SUITE)]
check("no eval prompt in any chunked passage", not chunk_hits,
      f"{len(chunk_hits)} bad chunks")

# ================================================= 2. VOCABULARY SIMULATION
print("\n=== 2. Vocabulary simulation (exact, SEED=42) ===")
base_chunks = chunk_text(classroom_corpus(), BLOCK_SIZE - 1)
base_chunks, separation = reserve_classroom_passages(base_chunks, SUITE)
print(f"  reserved {separation['excluded_passages']} classroom passages "
      f"for {len(separation['case_ids'])} eval cases")

all_chunks = base_chunks + extension_chunks
docs = sorted(set(all_chunks))
random.Random(SEED).shuffle(docs)
cut = int(.9 * len(docs))
train_docs, val_docs = docs[:cut], docs[cut:]

counts = Counter(t for doc in train_docs for t in word_tokens(doc))
retained = sorted((t for t in counts if len(t) <= 128), key=lambda t: (-counts[t], t))[:VOCAB_CAP]
vocabulary = ["<UNK>", "<BOS>", "<EOS>"] + sorted(retained)
stoi = {t: i for i, t in enumerate(vocabulary)}
dropped_types = sorted(set(counts) - set(retained))

print(f"  unique passages   : {len(docs)} (train {len(train_docs)} / val {len(val_docs)})")
print(f"  extension share   : {len(set(extension_chunks)) / max(1, len(docs)):.1%} of unique passages")
print(f"  distinct types    : {len(counts)}  (cap {VOCAB_CAP})")
check(f"vocabulary fits under the cap with margin", len(counts) <= 480,
      f"{len(counts)} types, headroom {VOCAB_CAP - len(counts)}")
check("no type evicted by the 509 cap", not dropped_types, f"dropped {dropped_types[:10]}")

# ====================================================== 3. THE OOV TABLE
print("\n=== 3. Would each of the 48 eval cases be scorable? ===")
non_scorable = []
for case in SUITE["cases"]:
    prompt_tokens = word_tokens(case["prompt"])
    unknown_prompt = sorted({t for t in prompt_tokens if t not in stoi})
    unknown_choices = [c for c in case["choices"] if word_tokens(c)[0] not in stoi]
    too_long = len(prompt_tokens) + 1 > BLOCK_SIZE
    if unknown_prompt or unknown_choices or too_long:
        non_scorable.append((case["id"], case["category"], unknown_prompt,
                             unknown_choices, too_long))

if non_scorable:
    for cid, cat, up, uc, tl in non_scorable:
        why = []
        if up:
            why.append(f"prompt words {up}")
        if uc:
            why.append(f"choices {uc}")
        if tl:
            why.append("context too long")
        print(f"  {cid} ({cat}): " + "; ".join(why))
check("all 48 cases would be scorable", not non_scorable,
      f"{len(non_scorable)} still out of vocabulary")

# ============================================ 4. BALANCE / THIN COVERAGE
print("\n=== 4. Required words: passage coverage ===")
required = set()
for case in SUITE["cases"]:
    required.update(word_tokens(case["prompt"]))
    for c in case["choices"]:
        required.add(word_tokens(c)[0])

passage_counts = Counter()
for doc in docs:
    for t in set(word_tokens(doc)):
        passage_counts[t] += 1

# A word appearing in only 1-2 unique passages can land entirely in the 10%
# validation split and never enter the vocabulary at all.
thin = sorted((passage_counts[w], w) for w in required if passage_counts[w] < 8)
if thin:
    print("  words in fewer than 8 distinct passages:")
    for n, w in thin:
        print(f"    {w}: {n}")
check("every required word appears in >= 8 distinct passages", not thin,
      f"{len(thin)} thin words")

# ================================================ 5. MID-CONTEXT PERIODS
print("\n=== 5. Multi-clause structure ===")
# chunk_text splits on ". " -- if the no-space periods were lost, every
# cross-clause category (negation, sequence, spatial, reference) is silently dead.
mid = sum(1 for c in extension_chunks if "." in word_tokens(c)[:-1])
check("multi-clause passages survived chunking", mid > 1000,
      f"{mid} passages contain a mid-context period")

print("\n=== Per-file passage counts ===")
for rec in manifest.get("files", []):
    print(f"  {rec['file']:34s} {rec['passages']:5d} passages "
          f"({rec['unique_passages']} unique)")

print()
if failures:
    print(f"RED LIGHT -- {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("GREEN LIGHT -- safe to train.")
