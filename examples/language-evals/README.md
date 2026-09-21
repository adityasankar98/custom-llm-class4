# Measured language-eval reference

This is one actual CPU run of the revised notebook: starter classroom corpus,
3,000 training steps, learning rate 0.001, training seed 42. It reserved 160 generated
passages containing eval prefixes before the split; 4,592 unique passages remained.
These results are separate from the historical notebook in `examples/`.

| Group | Untrained | Trained | Scorable cases |
|---|---:|---:|---:|
| Starter patterns | 6/16 | 16/16 | 16/16 |
| New wording | 3/8 | 4/8 | 8/8 |
| Extend corpus | 0/24 | 0/24 | 0/24 |
| All cases | 9/48 | 20/48 | 24/48 |

The 24 extension cases could not be validly scored because required words were
missing. They count as zero in all-case success, but are not evidence that the model
chose a wrong word from four known choices. Read the per-case vocabulary diagnostics.

All choices, probabilities, unconstrained continuations, and explanations are in the
untrained and final result files. Saved checkpoint inference reproduced every final
case. The terminal interface was also exercised with three actual prompts; its
transcript includes an out-of-vocabulary question and unedited generated replies.

This demonstrates attainable pattern tests, a harder wording-transfer test, and
clear reasons to extend the corpus. It does not prove that adding any particular
training file will solve the extension tests. Use your own measured results in your
submission. Minor numeric differences can occur on different hardware or PyTorch
versions. The recorded hashes identify the exact suite and model used here.
