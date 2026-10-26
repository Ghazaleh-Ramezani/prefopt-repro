# CPU validation run

Produced by `bash scripts/cpu_validation.sh` on an offline fixture: a
0.67M-parameter Qwen2-architecture model with a byte-level BPE tokenizer and a
240-pair synthetic preference set, built by `scripts/make_cpu_fixture.py`.

The run checks that each stage trains, saves its checkpoint, and that the next
stage and the evaluation harness rebuild exactly the model that was trained.
`goodhart.json` in the DPO and GRPO folders traces the Goodhart gap at every
saved checkpoint, with an independently seeded reward model (`rm_gold`)
standing in for the external judge. `zero3_cpu_check.json` compares DeepSpeed
ZeRO-3 training on two CPU ranks against a single-process run of the same
steps.
On a model this small the win-rates and diagnostics exercise the code paths;
comparisons between methods come from the 1.5B ablation grid.

Each subdirectory holds that stage's `config.resolved.yaml` and
`environment.json` (git SHA, library versions, hardware), plus its evaluation
output where the stage has one.
