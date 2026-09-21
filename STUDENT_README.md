# My Custom LLM Experiment

Replace the prompts below with your choices, actual outputs, and explanations.

Grading uses deliverable quality **4 points**, testing & evaluation **3 points**,
and working result **3 points**. Your model's eval percentage is not your grade.
Complete, valid eval evidence and a reasoned comparison matter; no minimum pass
rate or numerical improvement is required. Missing evidence earns less credit.

## My choices and prediction

State your corpus and source, training steps, and learning rate. Give a brief reason
for each. What patterns did you expect before running the notebook?

Did you use the classroom corpus, expand it with files in corpus/, or choose
folder-only mode? Name your permitted sources and the unique passages they added.
How did you check extracted PDF text and resolve warnings? Link corpus_manifest.json
when sharing is permitted. Do not publish private source text or derived outputs.

## My run

Link your **executed notebook**, config, and training summary. Record the actual
completed steps, elapsed time, hardware, parameter count, vocabulary size, document
count, and split sizes. Clearly identify interruptions or failures.

Link vocabulary_report.json and report training/held-out unknown-token rates.
Did the 509-type vocabulary retain the words that mattered for your experiment?
The split is by short passage, not source file; explain what that evaluation can test.

## My evidence

Embed your training_curves.svg. Include all measured losses in a table and link
history.json. State the evaluation panel sizes. Show the untrained, halfway, and
final samples and link the full sample files, including empty or garbled outputs.

Link tokenization.json and inspection.json. Show one word, its ID, and its
64-number vector before and after training. Show the saved parameter's before value,
gradient, and after value, plus the probability comparison for the same prefix.

Include the three temperatures' samples and link temperature_comparison.json.

## My fixed language evals

Link the unchanged `evals/language_evals.json`, runner, and every untrained/final
`eval_results.csv` and `eval_summary.json` from both experiments. Report all 48 cases,
all-case success, scorable accuracy, coverage, and scores by group/category. Show
actual free continuations as well as the multiple-choice score; they are different.

Fill this table with your actual values and links. Keep the category breakdowns
and complete per-case outputs alongside it; this summary does not replace them.

| Experiment | Stage | Correct / 48 | Scorable / 48 | Accuracy among scorable cases | Full results |
|---|---|---|---|---|---|
| Starter corpus | Untrained | | | | |
| Starter corpus | Trained | | | | |
| Expanded corpus | Untrained | | | | |
| Expanded corpus | Trained | | | | |

Which starter patterns worked? Did familiar words still work in new phrasings?
Which extension skills lacked words or examples? Name at least two categories you
chose, describe your new teaching material, and compare the starter and expanded
corpus runs. Do not hide failures or claim an improvement without measured evidence.

Link `eval_separation.json`. Explain how you kept prompts, answers, scoring rules,
results, and chat logs out of training and vocabulary building. Mention the limits
of exact-match leakage checks. These public tests guided development; they are not
an untouched final test.

## My chat interface

Give the exact notebook or terminal launch instructions and identify the saved
model/run. Include a screenshot or recording and at least three actual prompts
and replies from `chat_transcript.json`. Explain one limitation, unknown words,
the context limit, and whether prompts share history. Replies must come from your
trained nanoGPT, not canned text or a different model API.

## What I learned

Use actual values from your run to explain:

1. What is my corpus, what can it teach, and what is missing? Why hold data out?
2. How do a token, token ID, vector, and embedding differ?
3. What makes this a neural network? How did loss, gradients, and the optimizer change its weights?
4. What does attention combine, and why can it not look at future tokens?
5. How do probabilities become generated text? What changed with temperature, and did any weights change then?
6. Did the samples and both loss curves support my prediction? What can I honestly conclude?

## One limitation and my next experiment

Describe one observed limitation. Propose one change to data or a setting, explain
why, and predict the effect. Include the required second run with a corpus extension in at least two language-eval categories. Further experiments are optional.

## Reproduce and inspect

Explain how to open your notebook and locate the corpus and evidence. Verify the
repository signed out before submitting. Do not clear notebook outputs.
