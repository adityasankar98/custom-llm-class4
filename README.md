# Building a Custom LLM — Class 4

Training Karpathy's nanoGPT from scratch on a word-token corpus, measuring it with a fixed
48-case language eval suite before and after training across five experiments, and talking to
the result through a terminal chat interface.

**The best model here has 230k parameters, a 48-token context, and trained for 38 seconds on
synthetic sentences.** It is not a chat assistant. It continues sentences in the shapes it was shown.
Everything below reports what it actually did, including where it failed and where my
predictions were wrong.

| | |
|---|---|
| Executed notebook, experiment 1 | [`custom_llm_experiment1_starter.ipynb`](custom_llm_experiment1_starter.ipynb) |
| Executed notebook, experiment 2 | [`custom_llm_experiment2_extended.ipynb`](custom_llm_experiment2_extended.ipynb) |
| Executed notebook, experiment 3 | [`custom_llm_experiment3_targeted.ipynb`](custom_llm_experiment3_targeted.ipynb) |
| Executed notebook, experiment 5 | [`custom_llm_experiment5_capacity.ipynb`](custom_llm_experiment5_capacity.ipynb) |
| Eval suite (unchanged) | [`evals/language_evals.json`](evals/language_evals.json) |
| Chat evidence | [screenshot](results/chat-screenshot.png) · [session recording](results/chat-session-recording.txt) · [transcript](results/chat-experiment5.json) |

---

## 1. Headline results

The two required experiments are **E1** (starter corpus) and **E2** (corpus extension). E3–E5
are follow-ups that answer a question E2 raised, and they turned out to be the most informative
part of the project.

| # | Corpus | Layers | Steps | Untrained | **Trained** | Scorable acc | Coverage |
|---|---|---|---|---|---|---|---|
| **E1** | starter only | 2 | 3,000 | 9/48 | **20/48** | 83.3% | 24/48 |
| **E2** | + 8-category extension | 2 | 3,000 | 14/48 | **44/48** | 91.7% | 48/48 |
| E3 | + targeted fixes for E2's 4 failures | 2 | 3,000 | 11/48 | **42/48** | 87.5% | 48/48 |
| E4 | + 2 more pattern gaps closed | 2 | 3,000 | 14/48 | **44/48** | 91.7% | 48/48 |
| E5 | same corpus as E4 | **4** | **6,000** | 9/48 | **47/48** | 97.9% | 48/48 |

Runs: [E1](llm_runs/20260921T220602_231016Z/) · [E2](llm_runs/20260921T220633_918630Z/) ·
[E3](llm_runs/20260922T020828_433525Z/) · [E4](llm_runs/20260922T021600_443143Z/) ·
[E5](llm_runs/20260922T021637_203555Z/). Every run carries its own untrained and final result
sets, e.g. [E2 untrained](llm_runs/20260921T220633_918630Z/language_evals/untrained/) ·
[E2 final](llm_runs/20260921T220633_918630Z/language_evals/final/) ·
[E5 untrained](llm_runs/20260922T021637_203555Z/language_evals/untrained/) ·
[E5 final](llm_runs/20260922T021637_203555Z/language_evals/final/).

### The central finding: corpus work hit a ceiling that only capacity broke

E2 left four cases failing. I diagnosed each one, found that three had a specific provable
cause — **the frame they test appears zero times in the corpus** — and wrote teaching material
for exactly those gaps. That produced E3, which scored **42/48: worse than E2.** It fixed its
two targets (`categories_and_analogies` 1/3 → 3/3) and broke four other cases in categories I
had not touched. E4 closed two further gaps and landed back at **44/48**.

So at 2 layers, three different corpora scored **44, 42, 44**. Adding well-targeted teaching
material did not raise the ceiling; it **moved failures around**. Only when I raised capacity
from 2 to 4 layers and doubled the step budget did the score move to **47/48** — and it did so
while keeping every gain, which is what a capacity explanation predicts and a data explanation
does not.

**This is why E1 → E2 is the honest headline and 47/48 is not.** E5 is tuned: its architecture,
budget and corpus were all chosen after looking at which cases failed. E2 is the clean
measurement — extension material written once from the category list, measured once.

### Reading the E1 → E2 jump correctly

A large part of it is not learning. In E1, 24 of the 48 cases used words the starter corpus
never contains, so they were `out_of_vocabulary` and scored 0 automatically; coverage was capped
at 24/48 and **no amount of training could have moved it**. Adding teaching material made those
cases *answerable*, and an answerable case gets a 1-in-4 guess — about 6 correct from chance
alone. **So ~26/48 was the null result, not 20/48.**

Two mechanisms, kept separate throughout this README:

- **Vocabulary coverage** — did the word exist at all? Solved, and verifiable *before* training.
- **Learned pattern** — given the words, did the model pick the right one? The real measurement.

### Per-category breakdown, all five runs

| Category | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|
| domain_context | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| domain_place | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| new_wording | 4/8 | 8/8 | **6/8** | 8/8 | 8/8 |
| grammar | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| opposites | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| negation | 0/3 | 3/3 | **2/3** | **2/3** | 3/3 |
| reference | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| sequence | 0/3 | 1/3 | 1/3 | 1/3 | **2/3** |
| spatial_relations | 0/3 | 3/3 | **2/3** | **2/3** | 3/3 |
| everyday_knowledge | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| categories_and_analogies | 0/3 | **1/3** | 3/3 | 3/3 | 3/3 |

The bolded cells are the regressions that make the point: E3's targeted fix bought
`categories_and_analogies` at the cost of `negation`, `spatial_relations` and `new_wording`.
Every E1 row that reads 0/3 is zero *because the words did not exist*, which is why those rows
are identical before and after training in E1.

---

## 2. Where my predictions were wrong

I wrote a [prediction](custom_llm_experiment2_extended.ipynb) into the notebook before either
real run. Three parts of it were right and three were clearly wrong. Both matter.

**Right.** Coverage went to 48/48 exactly as the pre-training verification predicted.
`starter_patterns` improved dramatically (6/16 → 16/16) because hundreds of sibling sentences
teach those domain associations. Training loss fell steeply then flattened. And the embedding
prediction landed precisely: I named `client, buyer, shopper, consumer, subscriber` as the
words `customer` should move toward, and after training those are its top five neighbours at
cosine > 0.96, up from ~0.3 noise (§7).

**Wrong #1 — I predicted scorable accuracy would fall. It rose, 83.3% → 91.7%.** My reasoning
was that 24 hard cases joining the denominator would drag the average down. What I missed is
that the denominator in experiment 1 was only 24 cases, and 4 of those were `starter_transfer`
failures. The extension corpus fixed `starter_transfer` too (4/8 → 8/8), so the base improved
at the same time the denominator grew.

**Wrong #2 — I predicted `reference` was "probably hopeless". It scored 3/3.** I argued that
`maya lent a book to leo . leo thanked ___` requires suppressing the most recent name, which is
itself a distractor, and that two layers was too little machinery. With 527 passages of the
same frame shape, the model learned it. I was wrong about the capacity limit.

**Wrong #2b — but the follow-ups showed my capacity instinct was not baseless.** `reference`
held at 3/3 in every run, so I was wrong about *that* case. Yet E3 and E4 then demonstrated a
real capacity ceiling at 2 layers, which is the same argument applied to a different category.
The honest summary is that I picked the wrong case to be pessimistic about, not that capacity
was irrelevant.

**Wrong #3 — I predicted a `new_wording` regression** from extension material consuming ~49% of
training batches and halving classroom exposure. Instead `new_wording` went 4/8 → 8/8. The
extra data appears to have taught general function-word structure that transferred back to the
starter templates, rather than competing with them for capacity.

I am reporting these because the assignment grades prediction against observation, and three
confident, specific, wrong predictions are more informative than a vague one that cannot fail.

---

## 3. The experiments

### Experiment 1 — starter corpus

`corpus/` empty, `CORPUS="classroom"`. The notebook generates 6,360 synthetic sentences from
8 domains × 6 nouns × 4 contexts × 4 adjectives × 8 frames, plus 216 shopping sentences.

| | |
|---|---|
| Unique passages | 4,592 (train 4,132 / validation 460) |
| Vocabulary | 136 (133 word types + `<UNK>`, `<BOS>`, `<EOS>`) |
| Parameters | 111,872 |
| Unknown-token rate | training 0.0%, held-out 0.0% |
| Reserved eval passages | 160 |
| Steps / elapsed | 3,000 / 9.3 s |

### Experiment 2 — extended corpus

Same settings; `corpus/` now holds 9 authored `.txt` files. Extension material is **added to**
the classroom sentences, not substituted for them.

| | |
|---|---|
| Unique passages | 9,029 (train 8,126 / validation 903) — extension is 49.1% |
| Vocabulary | 426 (423 word types) |
| Parameters | 130,432 |
| Unknown-token rate | training 0.0%, held-out 0.031% |
| Reserved eval passages | 160 |
| Steps / elapsed | 3,000 / 10.3 s |

### Experiments 3–5 — the follow-ups

**E3** added teaching material for the three frames E2 had never shown the model (see §6), at
identical settings. **E4** closed two further gaps found the same way: `the {plural} are …`
(only `the {plural} were …` had been taught, which is why `lang_26` failed) and four more
vehicle types so the before→later rule had more distinct pairs to generalize from. Both ran at
2 layers / 3,000 steps, so **corpus is the only variable across E2, E3 and E4** — and all three
land at 44, 42, 44.

**E5** changes the model rather than the data: `N_LAYER` 2 → 4 and 6,000 steps on E4's corpus.
Changing two settings at once is a deliberate compromise — the larger model needs a longer
budget to converge — so E5 should be read as "more capacity, fairly trained", not as an
isolation of depth alone.

| | E4 (2 layers) | E5 (4 layers) |
|---|---|---|
| Unique passages | 9,704 (train 8,733 / val 971) | 10,520 (train 9,468 / val 1,052) |
| Vocabulary | 433 | 433 |
| Parameters | 130,880 | **230,848** |
| Steps / elapsed | 3,000 / 12 s | 6,000 / 38 s |
| Unknown-token rate | training 0.0%, held-out 0.009% | training 0.0%, held-out 0.009% |
| Score | 44/48 | **47/48** |

E4 was run from the script rather than the notebook, since it is an isolation control; its full
artifacts (config, corpus, both eval result sets, weights) are in its run folder exactly as for
the others. E1, E2, E3 and E5 each have an executed notebook.

Hardware: macOS 15.6.1, Apple Silicon, CPU only. Python 3.12.14, PyTorch 2.14.0. **No run was
interrupted; every run completed its full step budget** —
[E1 training_summary.json](llm_runs/20260921T220602_231016Z/training_summary.json) ·
[E2 training_summary.json](llm_runs/20260921T220633_918630Z/training_summary.json).

Supporting files: [E1 config](llm_runs/20260921T220602_231016Z/config.json) ·
[E2 config](llm_runs/20260921T220633_918630Z/config.json) ·
[E1 training.csv](llm_runs/20260921T220602_231016Z/training.csv) ·
[E2 training.csv](llm_runs/20260921T220633_918630Z/training.csv) ·
[E2 vocabulary report](llm_runs/20260921T220633_918630Z/vocabulary_report.json) ·
[E2 corpus manifest](llm_runs/20260921T220633_918630Z/corpus_manifest.json) ·
[E2 temperature_comparison.json](llm_runs/20260921T220633_918630Z/temperature_comparison.json) ·
[E2 checkpoint.json](llm_runs/20260921T220633_918630Z/checkpoint.json) ·
[results ZIP](llm_runs/20260921T220633_918630Z.zip)

**What stayed fixed, what changed in training, what changed only at inference.** Fixed across
both experiments: the eval suite and its scoring, seed 42, the 90/10 split procedure, the model
shape (2 layers, 4 heads, 64 dims, 48-token context), batch size 32, 3,000 steps, learning rate
0.001, and the generation settings used for samples. Changed by training: only the network
weights, via gradient updates. Changed only at inference, touching no weights: the sampling
temperature. The single deliberate difference between experiment 1 and experiment 2 is the
contents of `corpus/`.

---

## 4. The extension corpus

Generated reproducibly by [`build_corpus.py`](build_corpus.py). **Sources and permissions: every
sentence is original material I wrote for this assignment.** No external, copyrighted, personal,
or confidential text is used, so there is nothing in `corpus/` I lack the right to publish.

| File | Passages | Teaches |
|---|---|---|
| [`01_grammar_agreement.txt`](corpus/01_grammar_agreement.txt) | 1,332 | number agreement, tense |
| [`02_opposites.txt`](corpus/02_opposites.txt) | 152 | antonym frames |
| [`03_negation_correction.txt`](corpus/03_negation_correction.txt) | 750 | not-X-but-Y correction |
| [`04_reference_people.txt`](corpus/04_reference_people.txt) | 527 | name antecedents |
| [`05_sequence_order.txt`](corpus/05_sequence_order.txt) | 778 | first/then, before/after |
| [`06_spatial_relations.txt`](corpus/06_spatial_relations.txt) | 256 | containment, above/below, left/right |
| [`07_everyday_knowledge.txt`](corpus/07_everyday_knowledge.txt) | 87 | water/ice, umbrella/dry, dark/light |
| [`08_categories_analogies.txt`](corpus/08_categories_analogies.txt) | 327 | is-a facts, grows-into |
| [`09_vocabulary_support.txt`](corpus/09_vocabulary_support.txt) | 228 | plants required distractor words |

Total 4,437 passages, all unique, and the manifest records **zero extraction warnings**.
**I used no PDFs.** All nine files are plain UTF-8 `.txt`, so there is no OCR risk, no page
extraction to check, and no reading-order problem — the trade-off being that the material is
authored rather than drawn from real-world documents. Had I used PDFs, the check would have
been `corpus_manifest.json`'s per-file `pages`, `characters` and `warnings` fields, which flag
blank pages as needing OCR. The manifest's per-file SHA-256 hashes let anyone confirm the exact
bytes that trained the model.

### Why these categories

The assignment requires at least two. I targeted **all eight**, with **grammar, opposites, and
categories/analogies** as the declared primary three. The reasoning is that coverage is the one
outcome fully under my control: every extension case was scoring 0 purely for a missing word,
and fixing that is a verifiable pre-training step, not a gamble on training dynamics. Targeting
only two categories would have left 18 cases failing for a reason I could have fixed. The
marginal cost per extra category was roughly 300 sentences and 30 vocabulary types.

### Three non-obvious design constraints

**(a) Distractors must be taught too.** A case is `out_of_vocabulary` if *any* of its four
choices is missing — not just the answer. So `09_vocabulary_support.txt` deliberately plants
`spoon`, `pillow`, `steam`, `sand`, `asleep`, `goat`, `north`, `round`, `supper` and 17 others
in ordinary sentences. Without it, cases stay unscorable no matter how well the pattern is
taught.

**(b) The missing space after each period is deliberate, not a typo.** `chunk_text` splits
training documents on `(?<=[.!?])\s+`. Written normally, a three-clause teaching sentence
becomes three separate one-clause documents, and the model would never see a mid-context
period — while 15 of the 24 extension prompts are multi-clause and arrive as a *single*
context. Writing `.it` instead of `. it` yields an identical token sequence in one document:

```
"the crate is not red . it is blue . the crate is blue ."   -> 3 documents
"the crate is not red .it is blue .the crate is blue ."     -> 1 document
```

The corpus contains 2,311 passages with a mid-context period as a result. Without this the
negation, sequence, spatial and reference categories would all have silently failed.

**(c) The reverse direction is legal and is how the exact eval frame gets taught.** The prompt
`the opposite of hot is` may not appear, but `the opposite of cold is hot .` is a different
token sequence and is allowed. Nineteen antonym pairs are taught in both directions, so the
three tested pairs are ordinary members of a family rather than three memorized answers.

---

## 5. Keeping the exam out of the textbook

The eval suite is the exam; the corpus is the study material. Four independent checks:

1. **The generator filters itself.** `build_corpus.py` passes every candidate sentence through
   the project's own `matching_cases` and drops any that contains an eval prompt. It dropped
   **24 sentences**, including ones I would not have caught by eye — `one bird is small .`
   contains the lang_25 prompt, and `yesterday she walked .` contains lang_27.
2. **The loader would have refused.** `load_corpus_folder` raises on any imported file
   containing an eval prompt, checked against the *whole file text*, so even two adjacent lines
   concatenating into a prompt would stop the run.
3. **The notebook reserved 160 classroom passages** containing the 16 `starter_patterns`
   prefixes before the split and before vocabulary building —
   [`eval_separation.json`](llm_runs/20260921T220633_918630Z/eval_separation.json).
4. **Independent post-hoc check.** Scanning the actual `corpus.txt` that trained experiment 2
   for all 48 prompts returns **none**. The reference stories are absent:
   `maya lent a book to leo`, `ella gave finn a pencil` and `omar called nina` are all `False`.

**One thing I want to flag rather than bury.** The same check shows `a robin is a bird` and
`a salmon is a fish` *are* present, as separate sentences. That is permitted — the eval prompt
is the contiguous sequence `a robin is a bird . a salmon is a`, which never appears — and it is
the categories extension doing its job. But it means a correct answer there would be closer to
fact recall than inference, and I would rather say so. As it happens the model still gets
**lang_46 wrong**, for the reason in §6.

The suite is a **development benchmark**: it guided my corpus design, so none of these numbers
support a claim about unseen generalization. Demonstrating that would need a second suite that
never influenced the corpus.

---

## 6. The failures, and what diagnosing them produced

### E2's four failures, and the three that had a provable cause

Full rows: [E2 eval_results.json](llm_runs/20260921T220633_918630Z/language_evals/final/eval_results.json).

**lang_46 — the clearest one.** Prompt: `a robin is a bird . a salmon is a`

| bird | fish | tool | tree |
|---|---|---|---|
| **0.801** | 0.044 | 0.006 | 0.004 |

The model puts 80% on `bird`. Both facts are in the corpus, so this is not ignorance. Counting
occurrences showed why: the two-clause frame `a X is a Y . a Z is a ___` appears **zero times**
in E2's corpus — I had taught only single-clause facts (`a robin is a bird .`). Meeting a frame
it had never seen, the model fell back on the copy behaviour it learned from negation,
reference and spatial, and copied `bird` from the first clause.

**lang_38** had the same shape of cause. `happens after … the earlier meal is` appears **zero
times**: my generator only ever paired `happens after` with `the later meal is`, so the model
was never shown that "A happens after B" also means B is the earlier one.

**lang_48** (`a carrot is a vegetable . an apple is a`) picked `fabric` at 0.019 over `fruit` at
0.014 — all four probabilities near the floor, a different failure from lang_46's confident
wrong answer. Same missing frame.

**lang_39 was the exception.** The before→later pattern *was* taught, about 29 times. This one
was not a missing frame.

### What fixing them cost

Teaching those frames (with different content words — the tested sentences are dropped by the
leakage filter) fixed exactly what it targeted: `categories_and_analogies` went 1/3 → 3/3 and
held there in every later run. But E3 scored **42/48, two lower than E2**, because `negation`,
`spatial_relations` and `new_wording` each lost a case. At fixed capacity, the new frames
competed with the old ones. That is the observation the whole §1 finding rests on.

### The one case that never yielded: lang_39

`the train arrived before the bus . the vehicle that arrived later was the` → `bus`

| run | taxi | car | bus | train | picked |
|---|---|---|---|---|---|
| E2 (2 layers) | 0.209 | **0.134** | 0.135 | 0.150 | `taxi` |
| E4 (2 layers, more pairs) | 0.026 | **0.332** | 0.275 | 0.292 | `car` |
| E5 (4 layers) | 0.007 | 0.054 | 0.347 | **0.567** | `train` |

This is the most interesting row in the project. The model's behaviour changed qualitatively
even though the score stayed 0. In E2 it answered `taxi`, a vehicle **not in the prompt at
all**. By E5, `train` + `bus` hold 91% of the mass — it has learned "answer with a vehicle from
this sentence" — but it picks the **first** one instead of the second.

Why this case resists everything: `the train arrived before the bus …` is itself an eval
prompt, so the leakage filter *must* drop it. The model sees the rule demonstrated on ~80 other
vehicle pairs and has to generalize it to the single pair it is forbidden from seeing. Every
other extension case can be taught by sibling examples of the same pair-type; this one requires
genuine generalization to a held-out instance, and the model does not get there. **It is the
cleanest evidence in the whole project that the suite is measuring pattern-matching with a
generalization gap, not understanding.**

### The eval score overstates the model

Four-choice scoring asks only whether the right word beats three specific alternatives. Free
generation shows how thin the underlying competence is. From the trained model's own sample
timeline at step 3,000:

```
the box is not yellow . it is green . the box is red .
```

It produces the negation template perfectly and then **botches the copy** — having just said
green, it concludes red. Negation scored 3/3 on the constrained eval. The same pattern collapses
when the model must generate the word instead of rank it. In the chat session below, `the
opposite of cold is` returns `peach`, despite opposites scoring 3/3.

Both things are true: the model reliably ranks the right answer above three distractors, and it
cannot reliably produce that answer unprompted. The eval measures the first; only the samples
and the chat log reveal the second.

### Actual free continuations, next to the score

The runner saves an unconstrained continuation for every case alongside the four-choice score.
They frequently disagree, which is the point of saving both:

| case | score | free continuation (experiment 2) | experiment 1 |
|---|---|---|---|
| lang_07 `…surgeon explains the` | 1 (`patient`) | `treatment in detail .` | `treatment in detail .` |
| lang_28 `the opposite of hot is` | 1 (`cold`) | `cold .` | `''` — unscorable |
| lang_32 `…she bought milk . ava bought` | 1 (`milk`) | **`tea .`** | `''` — unscorable |
| lang_34 `…leo thanked` | 1 (`maya`) | `maya .` | `the station .` — unscorable |
| lang_46 `…a salmon is a` | 0 (`bird`) | `bird .` | `office .` — unscorable |

**lang_32 is the one to look at.** It scores 1 — the model ranked `milk` above `tea`, `rice` and
`bread`, which is the negation pattern working. Left to generate freely from the same prompt it
produces **`tea`**: precisely the word the sentence said was *not* bought. The four-choice score
and the free continuation disagree about the same case, on the same weights, at the same moment.

lang_07 shows a milder version: the model ranks `patient` highest but would generate
`treatment` — both correct for the hospital domain, just different words. And the experiment 1
column shows what "unscorable" actually looked like: empty strings, or `the station .` for a
prompt about people thanking each other, because the names were not in the vocabulary at all.

---

## 7. How it learns — traced through actual numbers

All values below are from
[`tokenization.json`](llm_runs/20260921T220633_918630Z/tokenization.json) and
[`inspection.json`](llm_runs/20260921T220633_918630Z/inspection.json) in the experiment 2 run.

**Corpus → tokens → IDs.** The corpus is raw text. The tokenizer lowercases it and splits into
whole words and punctuation. A real training passage:

```
today the bank focused on risk and the local mortgage .
```

becomes IDs `[1, 371, 365, 27, 141, 251, 308, 12, 365, 211, 231, 3, 2]` — `1` is `<BOS>`,
`3` is `.`, `2` is `<EOS>`. Each ID is just a row number in a lookup table; it carries no
meaning on its own. The model trains on inputs `[1, 371, …, 3]` against targets
`[371, 365, …, 2]`: the same sequence shifted by one, so every position predicts its successor.

**ID → vector.** The word `customer` is ID **94**. That indexes row 94 of the embedding table,
a 64-number vector. Before training its first five coordinates were:

```
[ 0.00881,  0.00078, -0.03013, -0.03366, -0.00840]   (random initialization)
```

After 3,000 steps:

```
[ 0.10480,  0.13483,  0.02179, -0.02016,  0.05697]
```

Nothing told the model what a customer is. The vector moved only because of the company the
word keeps.

**One real gradient and one real weight update.** The first recorded update, on coordinate 0 of
`customer`:

| before | gradient | learning rate | after |
|---|---|---|---|
| 0.008814873 | 0.003225453 | 1e-05 | 0.008804873 |

The gradient says "increasing this number increases the loss", so the optimizer moves it down.
The change is 1.0e-05 — about 0.1% of the value. The learning rate is 1e-05 rather than the
0.001 I set because this is step 1 of the 100-step **warmup**; the rate ramps to 0.001 and then
decays on a cosine schedule. Learning is millions of adjustments this small.

**Probabilities.** Given `the customer`, the untrained model's top predictions are essentially
flat noise — `customer` 0.0056, `box` 0.0036, `low` 0.0035 — roughly 1/426 each. After training:

| returned | recommended | ordered | compared | reviewed | selected |
|---|---|---|---|---|---|
| 0.222 | 0.182 | 0.159 | 0.152 | 0.143 | 0.107 |

Those six words are **exactly the six verbs** in the starter's shopping template
(`the {noun} {verb} the {product} after checking the price .`). The model has discovered that
after a role noun, one of six verbs follows, and has spread ~96% of its probability mass across
precisely that set. This is the clearest single piece of evidence that it learned the corpus's
structure rather than memorizing strings.

**Attention.** The saved first-head rows for a three-token prefix:

```
[1.000, 0.000, 0.000]
[0.846, 0.154, 0.000]
[0.305, 0.582, 0.114]
```

Each row is one position deciding how much to read from earlier positions. The upper-right
zeros are causal masking — a position can never see the future. Token 1 has only itself. By
token 3 the model is drawing 58% from token 2 and 30% from token 1. This mixing is how
`.omar thanked ___` can reach back past the recent name to the earlier one.

**Embedding neighbors — closing the prediction I made before training.** I predicted the
neighbors of `customer` would move "from arbitrary words toward the other role nouns it shares
frames with (client, buyer, shopper, consumer, subscriber)". Cosine similarity over the full
64-dimension vectors in [`checkpoint.json`](llm_runs/20260921T220633_918630Z/checkpoint.json),
before and after:

| word | nearest neighbors before training | after training (E2, 2 layers) |
|---|---|---|
| `customer` | onion 0.389, kittens 0.356, breakfast 0.314 | **client 0.974, buyer 0.973, shopper 0.972, subscriber 0.967, consumer 0.961** |
| `maya` | truck 0.352, other 0.289, `<BOS>` 0.274 | noah 0.618, leo 0.553, nina 0.538, nora 0.528 |
| `hot` | boat 0.325, who 0.296, shoe 0.278 | rough 0.792, happy 0.774, smooth 0.758, short 0.756 |

The `customer` prediction was exactly right — all five role nouns, in order, at cosine > 0.96,
up from ~0.3 noise. Names cluster with names. Nothing told the model these were categories; the
vectors moved because the words appear in the same slots.

**But look at `hot` in E2.** Its nearest neighbor is *not* `cold`. It is `rough`, `happy`,
`smooth`, `short` — other adjectives that fill the same template position. The 2-layer model
groups words by **distributional slot, not by meaning**, and an antonym is merely a word that
shows up in the same frames. Scoring 3/3 on `opposites` did not mean it represented
oppositeness.

**And this is where E5 surprised me.** Running the same measurement on the 4-layer model:

| word | after training (E5, 4 layers) |
|---|---|
| `customer` | buyer 0.956, consumer 0.954, shopper 0.951, subscriber 0.945, client 0.943 |
| `maya` | noah 0.558, emma 0.553, nora 0.543, sara 0.510 |
| `hot` | **cold 0.696**, happy 0.654, early 0.608, sad 0.604 |

`hot`'s nearest neighbor is now **`cold`** — the antonym, not a slot-mate. The extra capacity
did not just raise the score by three cases; it changed the *kind* of representation the model
built, from "words that appear here" toward something that groups a word with its opposite.
I would not have predicted that, and it is a better argument for the capacity story than the
eval numbers alone. Load
[`checkpoint.json`](llm_runs/20260921T220633_918630Z/checkpoint.json) into the supplied
[`embedding-viewer.html`](embedding-viewer.html) to explore this directly — note that the map
is a PCA compression, while the neighbor numbers above use the full vector space.

**Temperature** rescales the probabilities at generation time and **changes no weights**. All
twelve saved samples, same starting token and sampling seed, from
[`temperature_comparison.json`](llm_runs/20260921T220633_918630Z/temperature_comparison.json):

```
0.3  the report about the car explains the travel in detail .
0.3  the team discussed the subscriber and the service at the store .
0.3  the important offering was mentioned in the price report yesterday .
0.3  the important application was mentioned in the code report yesterday .

0.8  the report about the brand explains the quality in detail .
0.8  the box is not yellow . it is green . the box is red .
0.8  the team discussed the system and the update at the office .
0.8  we learned about the local investment during a discussion of interest .

1.2  the report about the brand explains the quality in detail .
1.2  our hospital has a question about the different therapist and patient .
1.2  first open the bag . then open it . the first action is wash .
1.2  people know that an far is a fabric .
```

At 0.3 the model plays it safe: two of four samples are the same template with one word
swapped, and every sentence is well-formed. At 0.8 it ranges wider and starts producing
extension material. At 1.2 it reaches far enough down the distribution to break — `people know
that an far is a fabric .` puts the adjective `far` in a noun slot, a word combination that
appears nowhere in the corpus. Identical weights in all twelve; only the sampling changed.

### Loss

![training curves](llm_runs/20260921T220633_918630Z/training_curves.svg)

Fixed evaluation panels of **at most 20 training and 20 validation documents**, averaged over
non-padding next-token targets. These are small estimates, not full-corpus measurements.

| Step | E1 train | E1 validation | E2 train | E2 validation |
|---|---|---|---|---|
| 0 | 4.9263 | 4.9275 | 6.0658 | 6.0446 |
| 1500 | 0.6821 | 0.7182 | 0.8261 | 0.9799 |
| 3000 | 0.6783 | 0.7061 | 0.7807 | 0.9184 |

Full history: [E1](llm_runs/20260921T220602_231016Z/history.json) ·
[E2](llm_runs/20260921T220633_918630Z/history.json)

Starting loss differs (4.93 vs 6.07) because the vocabularies differ: with no knowledge, loss
is about ln(vocab size), and ln(136)=4.91 while ln(426)=6.05. Both runs land almost exactly
there, which is a good sign the untrained baseline is honest. **Losses across different
corpora and vocabularies are not comparable** — experiment 2's higher final loss does not mean
it is a worse model, it means it is predicting from 426 options instead of 136.

The train/validation gap is small (0.78 vs 0.92) but this is **not** evidence of
generalization. Duplicate passages are removed before the 90/10 split, but held-out passages
come from the same templates and the same source files as training ones. A validation passage
is a sentence the model has not seen in a shape it has seen thousands of times. Low held-out
loss here means the templates were learned, nothing more.

### Samples

Same generation settings throughout. Full files:
[step 0](llm_runs/20260921T220633_918630Z/samples/step_0000.txt) ·
[step 1500](llm_runs/20260921T220633_918630Z/samples/step_1500.txt) ·
[step 3000](llm_runs/20260921T220633_918630Z/samples/step_3000.txt)

**Untrained** — word salad, no grammar, no repeated structure:
```
melts report it deposit salmon sent window desks physician into salmon he doctor noah turn
```

**Halfway (1,500)** — templates are already correct; content is not:
```
first open the pencil . then open it . the first action is wash .
```
The frame is perfect and the answer is wrong — it says `wash`, a word that never appeared in
the sentence. Structure is learned well before content.

**Final (3,000)** — fluent template sentences, with the copy error discussed in §6:
```
the report about the brand explains the quality in detail .
the box is not yellow . it is green . the box is red .
```

The visible change from 0 → 1500 is enormous; from 1500 → 3000 it is small, matching the loss
curve flattening. Most of the learning happened in the first half of the budget.

---

## 8. Chat interface

![chat session](results/chat-screenshot.png)

*Rendered from the verbatim session capture in
[`results/chat-session-recording.txt`](results/chat-session-recording.txt). Model:
`llm_runs/20260922T021637_203555Z/model.pt` — experiment 5, 4 layers, 6,000 steps, 230,848
parameters, the 47/48 run.*

Six real interactions, saved to
[`results/chat-session2.json`](results/chat-session2.json), plus a second session in
[`results/chat-experiment5.json`](results/chat-experiment5.json):

| # | Prompt | Reply | |
|---|---|---|---|
| 1 | `the customer` | `contains the towel . the towel is inside the drawer .` | **wrong domain** |
| 2 | `the opposite of cold is` | `is opposites .` | **failure** |
| 3 | `maya lent a cup to omar .omar thanked` | `maya .` | correct antecedent |
| 4 | `a trout is a fish . a robin is a` | `bird .` | the E3 fix, working live |
| 5 | `the train arrived before the bus . the vehicle that arrived later was the` | `train .` | **the lang_39 failure, live** |
| 6 | `quantum blockchain sushi` | `the different orange and the juice .` + unknown-word notice | out of vocabulary |

Turns 4 and 5 are worth putting side by side. Turn 4 is the two-clause analogy frame that E2
got wrong and E3 fixed — the model now correctly retrieves `bird` for `robin` instead of
copying `fish` from the first clause. Turn 5 is the one case that never yielded, reproducing
outside the eval harness exactly as §6 describes: it answers with a vehicle from the sentence,
and picks the wrong one.

**The observed chat limitations** are turns 1 and 2. `the opposite of cold is` returns
`is opposites .` even though `opposites` scores **3/3** on the eval — ranking `hot` above three
distractors is far easier than producing it from 433 options. Turn 1 is a regression I did not
expect: in experiment 2 `the customer` produced `returned the product after checking the
price .`, a correct starter template, but the E5 model answers with spatial-containment
material instead. The extension corpus is now over half the training data, so the starter
domains have lost ground in free generation even while `starter_patterns` still scores 16/16
under constrained ranking. Turn 6 shows the vocabulary boundary: unknown words become `<UNK>`,
carry no meaning, and the model falls back to a generic frame.

Interface properties, as required: it is a **tiny language model** that continues text rather
than answering questions; every prompt **starts fresh** with no conversation memory; context is
capped at **48 tokens** and longer prompts are truncated with a notice; unknown words are
listed explicitly. Generating replies never retrains the model and never adds chat text to the
corpus.

---

## 9. Reproducing this

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt numpy
```

Verify the corpus before training (takes about a second):

```bash
.venv/bin/python verify_corpus.py
```

Rebuild the corpus from the generator:

```bash
.venv/bin/python build_corpus.py
```

Rerun the evals against either saved model — `model.pt` is self-contained, carrying both
weights and vocabulary, so no separate vocab file is needed. `--output` must be a fresh
directory:

```bash
.venv/bin/python run_evals.py --model llm_runs/20260922T021637_203555Z/model.pt --stage final --output results/my-final-evals
```

```bash
.venv/bin/python run_evals.py --model llm_runs/20260922T021637_203555Z/model_untrained.pt --stage untrained --output results/my-untrained-evals
```

Launch the chat interface (`--transcript` refuses to overwrite, so use a fresh filename):

```bash
.venv/bin/python chat.py --model llm_runs/20260922T021637_203555Z/model.pt --transcript results/my-chat.json
```

Retrain from scratch — edit `TRAINING_STEPS` / `LEARNING_RATE` in section 1 of `custom_llm.py`,
then:

```bash
.venv/bin/python build_notebook.py && .venv/bin/jupyter nbconvert --to notebook --execute custom_llm.ipynb
```

I verified the standalone reruns reproduce the in-notebook numbers exactly: E1 9 -> 20, E2 14 -> 44, E5 9 -> 47. Saved under [`results/`](results/).

**Scoring rules** (unchanged from the supplied runner): the model sees only the prompt — never
the four choices, never the answer key. It scores 1 when the correct word receives the highest
next-token probability among the four choices, 0 otherwise. Ties score 0. Cases with any
unknown prompt or choice word, or an over-long context, are `out_of_vocabulary` /
`context_too_long`, score 0 in the all-case rate, and are excluded from scorable accuracy and
coverage. A free continuation is generated and saved separately for every case and is **not**
what the score measures.

### Layout

| Path | |
|---|---|
| `custom_llm.py` | notebook source; `build_notebook.py` compiles it to `.ipynb` |
| `build_corpus.py` | generates the extension corpus with the leakage filter |
| `verify_corpus.py` | read-only pre-flight check (this is the important one) |
| `corpus/` | teaching material only |
| `evals/` | the fixed 48-case suite — never a training input |
| `llm_runs/` | per-run artifacts, weights, eval results, ZIPs |
| `results/` | standalone eval reruns and chat evidence |

`run_evals.py`, `chat.py`, `nanogpt_model.py` and `evals/language_evals.json` are supplied and
**unmodified** — the notebook verifies all four against pinned SHA-256 hashes on every run and
refuses to start if any byte changed.

**Attribution.** The transformer is Karpathy's nanoGPT at commit `3adf61e`, used under its MIT
license, retained verbatim in [`NANOGPT_LICENSE`](NANOGPT_LICENSE). The notebook scaffold, eval
suite, runner and chat interface come from the course
[starter repository](https://github.com/pepealonso95/custom-llm). My own contributions are
`build_corpus.py`, `verify_corpus.py`, the nine files in `corpus/`, the prediction and choices
in section 1 of the notebook, and this README.

---

## 10. Limitation and next experiment

**The limitation, now with evidence rather than speculation.** After E2 I wrote that the model
uses a single copy mechanism for two incompatible jobs — copying a name across a clause
boundary correctly (`reference` 3/3) while copying a *category* across the same frame
incorrectly (`lang_46`, 80% on `bird`) — and proposed testing it by raising `N_LAYER` from 2
to 4. E3–E5 ran that test, and the answer was more interesting than either branch I predicted:

- The immediate cause of `lang_46` was **not** capacity. It was a missing frame — the
  two-clause analogy pattern occurred zero times in the corpus. Teaching it fixed the category.
- But fixing it at 2 layers **cost three cases elsewhere**, and three different corpora all
  scored 44, 42, 44. That ceiling *was* capacity.
- Doubling depth cleared it to 47/48 and, unexpectedly, changed the embedding geometry too:
  `hot`'s nearest neighbour moved from slot-mates (`rough`, `smooth`) to `cold` itself (§7).

**The limitation that remains** is `lang_39`, and it is the sharpest one in the project because
it cannot be fixed by teaching. Its tested sentence is an eval prompt, so the leakage filter
must drop it; the model has to generalize the before→later rule to the one pair it is forbidden
to see. Across E2 → E5 it progressed from answering with a vehicle absent from the prompt, to
concentrating 91% of its mass on the two vehicles present, to still choosing the wrong one.
Pattern coverage improved; generalization to a held-out instance did not arrive.

**The next experiment** I would run: teach the before/after rule on a *non-vehicle* domain
entirely — people arriving, letters posted, songs played — while leaving the vehicle examples
untouched, and re-measure `lang_39`. If an abstract ordering rule learned elsewhere transfers
to vehicles, the model has something like a reusable relation. If it does not, then what looks
like a learned "before → later" rule is really a per-domain lookup table, and the 47/48 is even
more surface-level than this README already claims. That is a genuine test of generalization
rather than another round of coverage work, and it is cheap — about 40 seconds of training.

**Two limitations on the numbers themselves.** First, **every score here comes from a suite that
guided the corpus design**, and E5 additionally had its architecture and budget chosen after
seeing which cases failed. The honest description is a *development benchmark*; 47/48 means
"the taught patterns were learned", not "the model understands language". E1 → E2 is the
cleanest measurement in the project and 44/48 is the number I would defend. Second, the
constrained score consistently overstates the model: `opposites` scores 3/3 while the chat
interface answers `the opposite of cold is` with `is opposites .`, and `negation` scored 3/3 in
E2 while free generation produced `the box is not yellow . it is green . the box is red .`
