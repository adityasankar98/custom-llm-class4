# Building a Custom LLM

Class 4, Fall 26 · From Zero to AI Agents

Train a tiny language model and understand how **data, tokens, vectors, embeddings,
neural networks, and learning** fit together. This ready-made notebook closely follows
[Andrej Karpathy's microgpt](https://karpathy.ai/microgpt.html). Your work is to choose,
run, inspect, and explain. Writing the model from scratch is optional.

[Open in Google Colab](https://colab.research.google.com/github/pepealonso95/custom-llm/blob/main/custom_llm.ipynb)
· [Assignment Google Doc](https://docs.google.com/document/d/1MQ3YQl2ywWZF7W5_l_91FiIp7pTYPO_3viI2JVapRcc/edit)
· [Assignment text](ASSIGNMENT.md)
· [Notebook](custom_llm.ipynb)

## Start here

1. Open the notebook in Colab and save your own copy. The default CPU runtime is enough.
2. In section 1, choose **corpus**, **training steps**, and **learning rate**. Write your reasons and prediction.
3. Try 10 steps to check setup, then use 1,000 as a starting training budget. Select **Run All**.
4. Read the explanations and inspect the actual token IDs, vectors, probabilities, weight update, samples, and loss plot.
5. Save the results ZIP **and** download the executed notebook separately. Keep its outputs visible.
6. Put your notebook, evidence, and explanation in your own public GitHub repository. Submit its URL through the [course portal](https://submissions-portal-eight.vercel.app).

For local Jupyter or VS Code, use a Python 3 kernel. The model itself uses only the
standard library. You can also run `python custom_llm.py` after editing the three
settings at its top. The `.py` file contains the same code and teaching text as the notebook.

## What you should understand

| Idea | Evidence you will inspect |
|---|---|
| Corpus and data | Five documents, unique-line count, train/validation split |
| Tokens and IDs | A character mapped to an integer and back; next-token targets |
| Vectors and embeddings | One token's 16 numbers before and after training |
| Neural networks | Weighted sums, ReLU, layers, and adjustable parameters |
| Learning | Loss, a real gradient, and the first change to an embedding parameter |
| Attention and context | A trained attention row that uses only present and earlier tokens |
| Generation | Before/after probabilities and samples at three temperatures |

The core story is **examples → predictions → loss → weight updates → changed predictions**.
An embedding coordinate need not have a nameable human meaning. Plausible outputs
do not demonstrate factual knowledge. This tiny model generates short strings,
not general-purpose chat answers.

## Explore the embedding space in 3D

Download [embedding-viewer.html](embedding-viewer.html) and open it in your browser.
It is a single offline file with the reference model's actual 16-dimensional token
embeddings built in. Drag to rotate, scroll to zoom, and select a character to
inspect its vector and its closest neighbors in the original 16-dimensional space.

- Switch between random initialization and the trained model. The PCA projection
  uses the same center, axes, and scale for both states.
- Turn on **Show movement** to connect the endpoints. These lines are not the
  model's actual training trajectory.
- Choose **Open your checkpoint** and select `checkpoint.json` from your results
  ZIP to inspect your own model. Nothing is uploaded. Imported checkpoints do not
  include initial embeddings, so their before/after controls are disabled.
- The 3D projection loses information. Its retained variance is shown on screen;
  character proximity is not evidence of word-level meaning.

Teaching prompt: choose a character, compare its numbers and neighbors before and
after training, and explain why a nearby point in 3D need not be a close vector in
16D. These are token lookup embeddings, not the contextual states after attention.

For maintainers: `python3 build_embedding_viewer.py` refreshes the bundled reference
data and checks it against the recorded probe. `node test_embedding_viewer.cjs`
checks the checkpoint match, PCA, similarities, and import validation.

## Your corpus

The default is the names corpus used by microgpt, downloaded from
[Karpathy's makemore dataset](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt).
You may instead upload a UTF-8 `.txt` file and set `CORPUS` to its path.

- One short document per line: for example a name, place name, or product label.
- At least 100 distinct nonempty lines, each **1–15 characters** long.
- Duplicate lines are removed before a seeded 90/10 document split.
- Longer lines are rejected explicitly because the model has a 16-position context.
- Use data you are allowed to share. The results ZIP includes your corpus.

The vocabulary enumerates allowed characters from the supplied file; held-out
documents never provide weight updates. Evaluation uses fixed panels of up to
20 documents per split. Loss is the mean of each document's mean next-token loss.
Compare losses within a run; different corpora and vocabularies are not a leaderboard.

## Results to keep

Each run creates a fresh folder under `llm_runs/` and a ZIP containing:

- `samples/step_0000.txt`, halfway samples, and final samples
- `training_curves.svg`, `history.json`, and `training.csv`
- `tokenization.json` and `inspection.json` with actual vectors, probabilities, attention, and the first weight update
- `temperature_comparison.json`
- `config.json`, `training_summary.json`, `corpus.txt`, and `split.json`
- `checkpoint.json` with weights and vocabulary for inspection

The checkpoint is not an exact training-resume file because it does not include
optimizer state. If you interrupt training, run the remaining inspection and save
cells and report the completed step count. To start another experiment, run from
the top so the model and optimizer reset. The ZIP does **not** capture the currently
open notebook; save that separately after the final cell.

Use [STUDENT_README.md](STUDENT_README.md) as a starting structure for your submission.
The final work is your explanation of your run, not a copy of this project README.

## Verified reference run

The entire notebook was executed in order on Python 3.13 on an Apple Silicon Mac,
using the supplied names corpus, 1,000 steps, and learning rate 0.01.
The measured training/evaluation section took about one minute on that machine.
This is one observed runtime, not a promise for your computer or Colab.

| Step | Training panel loss | Validation panel loss |
|---|---:|---:|
| 0 | 3.3488 | 3.3815 |
| 500 | 2.3910 | 2.3644 |
| 1000 | 2.3121 | 2.2187 |

Example first samples changed from `cqnvdhpnsosuwqer` and `cxiwttdhzyt` to
`anisa` and `karin`. The full outputs, including less convincing samples, are in
[the executed reference notebook](examples/custom_llm.executed.ipynb).
Use your own outputs in your submission. A 10-step setup run was also executed successfully.

## Relationship to Karpathy's microgpt

Read the [original explanation](https://karpathy.github.io/2026/02/12/microgpt/)
and [source](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).
The neural network and scalar-autograd implementation follow that design closely:
one layer, four heads, 16-dimensional token/position embeddings, a 16-position
context, RMSNorm, ReLU, residual additions, Adam, and next-character prediction.

The classroom version adds data checks, deduplication and held-out evaluation,
inspection checkpoints, stable log-loss, a separate sampling random generator,
and saved results. The extra notebook text and evidence helpers explain the small
model; they are not additional architecture students must implement.

Optional: after understanding this model, use
[Karpathy's GPT video project](https://github.com/karpathy/ng-video-lecture)
for PyTorch and longer Shakespeare text. GPU optimization, larger datasets, and
writing an autograd engine are optional extensions.
