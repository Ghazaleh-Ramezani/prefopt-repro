# Debugging log — bringing the pipeline up end to end

The four training stages were first written against TRL's documented
interfaces, with unit tests for config parsing, reward shaping, statistics and
reporting. Running every stage in sequence — each consuming the previous
stage's checkpoint, then evaluation and diagnostics rebuilding those models —
is what tests the seams between them, and that is where these defects were.

Thirteen turned up. Twelve are code defects, each now fixed and covered by a
regression test; one (5) is an artifact of the offline fixture's tokenizer and
is recorded as such. Three of them change results without raising
anything: 3, which froze the model body under LoRA; 6, which evaluated
DPO and GRPO checkpoints as a model that was never trained; and 12, which
scored every reward-model judgement on text in a format the reward model was
never trained on.

Defects 1–5 surfaced on the first full run; 6–11 came from a second, adversarial
audit of the evaluation and diagnostics code; 12–13 came from preparing the
first run on real data and a GPU.

Environment: 2 CPU cores, 7 GB RAM, no GPU, no Hugging Face Hub access. A
0.67 M-parameter Qwen2-architecture model (4 layers, hidden 128) and a 240-pair
synthetic preference set are built locally by `scripts/make_cpu_fixture.py`, and
`scripts/cpu_validation.sh` reproduces the whole run. The fixture is seeded, so
two runs of the script produce byte-identical results. On a model that small
the numbers exercise the code paths; the control-flow defects they expose are
real.

---

## 1. SFT crashes on any split carrying a string `prompt` column

```
jinja2.exceptions.UndefinedError: 'str object' has no attribute 'role'
```

raised from `apply_chat_template` during `SFTTrainer`'s dataset preparation.

`load_sft_dataset` returned every column of the source split.
`load_preference_dataset` filtered its columns with `_keep_columns`, but the
SFT loader never got the same treatment. TRL's `maybe_apply_chat_template`
treats a `prompt` key as *conversational* — a list of `{role, content}`
messages — so a plain-string `prompt` sitting next to `messages` makes it
render a string as a message list.

This is not specific to the local fixture: the configured default,
`HuggingFaceH4/ultrafeedback_binarized`, has exactly this shape in its
`train_sft` split.

**Fix:** filter the SFT split to `messages` before it reaches the trainer
(`SFT_COLUMNS` in `data.py`).

## 2. RM and DPO crash on implicit-prompt preference data

```
TypeError: string indices must be integers, not 'str'
  trl/data_utils.py:101   last_role = example["prompt"][-1]["role"]
```

TRL accepts preference data in one of two shapes. With an *explicit* prompt,
all three of `prompt`, `chosen` and `rejected` are message lists and
`chosen`/`rejected` hold only the assistant turn. With an *implicit* prompt,
`chosen`/`rejected` are complete conversations that already contain the user
turn, and there is no `prompt` key at all.

`PREFERENCE_COLUMNS` kept `prompt` alongside full-conversation
`chosen`/`rejected`, which is neither shape, so TRL indexed the string as a
message list. Both affected stages share the one loader, so both failed the
same way. `ultrafeedback_binarized` is again exactly this shape.

**Fix:** detect the implicit shape at load time and drop the `prompt` column
(`_drop_redundant_prompt`).

## 3. Gradient checkpointing silently froze the model body under LoRA

This is the one worth reading.

The reward model trained without error and reported **0.2667** pairwise
accuracy on held-out pairs — below the 0.5 a coin would get, on a dataset whose
chosen responses are eight times longer than its rejected ones. Training loss
sat at 0.703, which is ln 2 to three decimals: the loss of a model whose
reward differences are all zero.

Buried in the output was:

```
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  torch/utils/checkpoint.py:242
```

Two candidate explanations: no gradient was flowing, or the metric was wrong.
These predict different things, so they can be separated.

**Isolating it.** A thirty-line probe builds the same model and LoRA config,
runs one forward and backward pass, and counts how many trainable parameters
receive a non-zero gradient — across checkpointing on/off and `use_reentrant`
true/false:

```
gc OFF                              : 9/17 trainable params got a non-zero gradient
gc ON  reentrant=True  (repo before) : 1/17 trainable params got a non-zero gradient
gc ON  reentrant=False              : 9/17 trainable params got a non-zero gradient
gc ON  reentrant=True  + input_grads: 9/17 trainable params got a non-zero gradient
```

One parameter out of seventeen — the classifier head, which sits outside the
checkpointed blocks. Every LoRA adapter inside the transformer body was getting
nothing.

**Cause.** Reentrant checkpointing decides whether to recurse into a
checkpointed block by checking whether that block's *inputs* require grad.
Under LoRA the base weights are frozen, so they do not, and autograd skips the
body. `configs/base.yaml` set `gradient_checkpointing: true` and no stage passed
`gradient_checkpointing_kwargs`, so Transformers used its default of
`use_reentrant=True`. Nothing raised. The loss simply did not move.

**Confirming the metric was innocent.** With `use_reentrant=False` the warning
disappeared, but accuracy stayed at 0.283 and the loss stayed at ln 2 — a
0.67 M-parameter randomly-initialised model at lr 5e-6 has nothing to show after
three epochs either way. Raising the learning rate to 5e-3 and training for 8
epochs gave **loss 0.054, pairwise accuracy 1.0000**. So the accuracy
computation was correct all along, and the two failures were independent: a
broken gradient path *and* a learning rate far too small for a toy model.

That ordering matters. Had the fix been applied without the second check, the
still-below-chance accuracy would have looked like the fix failing.

**Fix:** `train.gradient_checkpointing_use_reentrant`, default `False`, passed
as `gradient_checkpointing_kwargs` by all four stages. The flag must stay
`False` while the base model is frozen, which is every stage here.

**Test:** `test_default_checkpointing_reaches_every_lora_adapter` runs the
probe as a regression test with the repo's default config, and fails if the
default is switched back to reentrant.

**Why it would have been expensive to find later.** It produces no error, no
NaN and no warning that names the real problem. On an H100 the symptom is a
sweep whose win-rates all land within noise of the SFT baseline — which reads
as "the method does not work on this dataset", not as "the body never trained".

## 4. `evaluate` and `diagnostics` could not load any checkpoint this repo produces

Under PEFT, `trainer.save_model()` writes adapter weights only: every run
directory holds `adapter_config.json` and no `config.json`.
`AutoModelForCausalLM.from_pretrained` cannot open that — the base model has to
be loaded first and the adapter applied on top.

`generate()` already *accepted* a `base_model` argument and both call sites
passed `cfg.model.base_model`, but the body never used it: it called
`from_pretrained(model_path)` directly. `RewardModelJudge` and `mean_token_kl`
had the same problem and no `base_model` parameter at all.

So every number downstream of training — win-rate, bootstrap CIs, swap
consistency, all five diagnostics, `RESULTS.md` — was unreachable. The unit
tests did not catch it because they cover config parsing, reward shaping,
statistics and reporting, none of which load a model.

**Fix:** one `load_for_inference(path, base_model, head=...)` in `modeling.py`
that detects an adapter directory and resolves it, used by all three call
sites. It also replaces a hardcoded `torch_dtype=torch.bfloat16` with a probe
for bf16 support, so evaluation runs on CPU.

A second, smaller problem surfaced in the same place: `evaluate.py` and
`diagnostics.py` had their own argparse without the `--set` override flag the
four training stages take through `cli.parse_and_prepare`. A run trained with
overrides therefore could not be evaluated under the settings it was trained
with — the parser rejected the flag outright, and `scripts/run_ablations.sh`
sweeps with exactly those overrides. Both now accept `--set`.

## 5. `token_type_ids` passed to `generate()` — fixture artifact

```
ValueError: The following `model_kwargs` are not used by the model: ['token_type_ids']
```

The local fixture's `PreTrainedTokenizerFast` emits `token_type_ids`, which a
causal LM's `generate()` rejects rather than ignores. Qwen's own tokenizer does
not emit them, so this would not occur in the configured setup.

Recorded for completeness, and the one-line guard in `evaluate.generate` is
kept because it costs nothing and makes the path robust to other tokenizers.
The same error from inside TRL's `GRPOTrainer` was resolved by setting
`model_input_names` on the fixture tokenizer, not by changing library code.

## 6. DPO and GRPO checkpoints were reloaded onto the wrong model

`load_policy` merges the SFT adapter into the base, and DPOTrainer then trains
a fresh LoRA adapter on top of those merged weights. The saved DPO adapter is
therefore a delta on the *SFT-merged* model — but its `adapter_config.json`
names the raw base, and the loader applied it there. The result loads without
error, generates text, scores, and is a model that was never trained: it has
the DPO delta and none of the SFT one.

It also moves the diagnostics. On an early, unseeded build of the fixture an
audit probe measured a maximum logit difference of 0.13 between the reloaded
model and the trained one, and `length_reward_corr` for the DPO policy read
+0.476 mis-loaded against −0.473 rebuilt correctly: the sign flipped. On the
seeded fixture now published the gap is smaller — maximum logit difference
7e-4, `length_reward_corr` 0.448 mis-loaded against 0.385 rebuilt correctly.
At 1.5B, where SFT is a full training stage, the mis-loaded model is missing
all of it.

**Fix:** each run already records its starting adapter as `init_adapter` in
`config.resolved.yaml`. `modeling.adapter_chain` walks that record back to the
base and `load_for_inference` applies and merges each adapter in order, so
evaluation, diagnostics and the reward judge all rebuild the model that was
trained. A recorded `init_adapter` that no longer exists raises rather than
falling back to the raw base.

**Tests:** `tests/test_adapter_chain.py` builds SFT and DPO adapters on a tiny
local model and checks that the reloaded model reproduces the trained logits
to 1e-5, and that applying the DPO adapter to the raw base — the old
behaviour — gives a measurably different model.

## 7. Unknown top-level config keys were silently ignored

The README states that unknown keys raise at startup. That held inside each
section — `_from_dict` rejects them — but not at the top level, so a misspelt
section such as `trian:` was dropped and the run used every default in it.
`load_config` now validates top-level keys against `RunCfg` too
(`test_unknown_top_level_key_fails_loudly`).

## 8. `spearman` invented correlation from tied values

The length/reward correlation ranked with `argsort(argsort(x))`, which gives
ordinal ranks and breaks ties by input order. A constant input therefore got
ranks 0, 1, 2, … and correlated perfectly with any sorted partner — an audit
probe returned 1.0 for constant completion lengths — and tied values
disagreed with `scipy.stats.spearmanr` (0.80 against 0.0 in one probe).
Ranks now average over ties, a constant input returns NaN as scipy does, and
unequal lengths raise (`tests/test_eval_logic.py`).

## 9. The judge parser accepted verdicts it should not have

`_parse_verdict` stripped non-letters and matched on a prefix, so
"Both are fine" scored as a B win and "Answer A" as an A win. Worse, in
`win_rate` two unparseable verdicts satisfied the agreement test — neither
"A", "B" nor "TIE", and equal to each other — and counted as a *consistent
loss*. The parser now accepts only an exact first token of A, B or TIE and
returns INVALID otherwise; `win_rate` scores an INVALID pair as a tie, excludes
it from swap consistency, and reports `invalid_verdicts`, so a judge that stops
following the format shows up in the results instead of shifting them.

## 10. `win_rate` truncated misaligned inputs

`zip()` over prompts, policy outputs and baseline outputs stops at the
shortest list, so a generation step that dropped outputs produced a win-rate
on fewer prompts than reported, with `n` silently smaller. Misaligned inputs
now raise.

## 11. Evaluation sent adapter directories to vLLM

`generate` used vLLM whenever it was installed, passing the run directory as
the model. A run directory holds a LoRA adapter (on top of an SFT adapter, per
6), with no `config.json`, so vLLM fails on it — and with an error that is not
an `ImportError`, so the HF fallback below it was never reached. On a GPU
machine with vLLM installed, which the README recommends for GRPO rollouts,
evaluation would have crashed on every checkpoint. vLLM is now used only for
full model directories; adapter runs go through `load_for_inference`
(`test_generate_skips_vllm_for_adapter_directories`).

## 12. The reward-model judge scored text in a format the reward model never saw

`RewardModelJudge.score` built its input as `f"{prompt}\n\n{response}"`.
TRL's `RewardTrainer` does something else with a conversational pair: it
renders `chosen` and `rejected` through the tokenizer's chat template
(`maybe_apply_chat_template`) and then tokenizes that text. The scalar head is
therefore trained only on chat-formatted conversations, and every score the
harness computed — the RM-judged win-rate in `evaluate`, `length_reward_corr`
and the RM win-rate in `diagnostics`, the proxy side of `goodhart` — came from
text outside its training distribution. Nothing errors: the scores are a
different function of the response, not obviously wrong ones.

It came up while choosing an independent public reward model as the external
judge for the first run on real data. Its model card specifies chat-template
input, and checking our own reward model's training path against that showed
the same requirement had been missed.

On the seeded fixture, with training unchanged (DPO eval reward margin 0.445
and KL 0.0017 in both runs), scoring the chat-formatted text moves:

| | before | after |
|---|---|---|
| DPO RM-judged win-rate vs SFT | 0.958 (0.917, 1.000) | 1.000 (1.000, 1.000) |
| DPO `length_reward_corr` | 0.385 | 0.503 |
| GRPO `length_reward_corr` | 0.643 | 0.818 |
| GRPO RM-judged win-rate vs SFT | 0.500 | 0.417 |
| DPO Goodhart gap, last checkpoint | 0.000 | 0.042 |

The verbosity signal the fixture plants (chosen responses are longer) shows up
more strongly once the reward model reads the format it learned on. The size
of the effect on real data is measured by the small run's judge checks
(`notebooks/small_real_run.ipynb`), which score the same held-out pairs both
ways.

**Fix:** `RewardModelJudge.format()` renders `[user, assistant]` through the
chat template exactly as `RewardTrainer` does, and `score()` tokenizes that.
The plain join is kept only for tokenizers with no chat template.
`test_judge_scores_the_text_the_reward_trainer_trained_on` compares the judge's
input with TRL's own `maybe_apply_chat_template` output.

## 13. Evaluation ran in emulated bf16 on GPUs without native bf16

`inference_dtype` chose bfloat16 whenever `torch.cuda.is_bf16_supported()` was
true. That probe defaults to `including_emulation=True`, so on a T4 (compute
capability 7.5) it returns True and generation, the judge and the KL
measurement ran in emulated bf16, even in a run that trained in float32 because
the GPU has no bf16 units. `native_bf16()` now requires compute capability 8.0+
(`test_emulated_bf16_is_not_used_for_inference`).

---

## What the run establishes

All four stages execute in sequence, each consuming the previous stage's
checkpoint, and evaluation, diagnostics and the reward judge rebuild the
models that were trained. `scripts/cpu_validation.sh` reproduces this from a
clean checkout on a CPU, and its artifacts are in `results/cpu_validation/`.

The reward model reaches 1.0 held-out pairwise accuracy once the gradient path
is intact and the learning rate suits a randomly initialised model. DPO moves
its implicit-reward metrics in the right direction (eval `rewards/accuracies`
1.0, margin 0.44), and position-swapped evaluation, bootstrap intervals and
all five reward-hacking diagnostics compute on its output: RM-judged win-rate
100% over two generation seeds, swap consistency 1.0, no invalid verdicts
(95.8% before the judge read the chat format, per 12).

**Next:** the ablation grid — beta sweep, group size, reward-model quality — at
1.5B on GPU. That is where the comparisons between methods come from; this run
establishes the wiring they depend on.

Two signals from the GRPO stage to watch at scale. `completions/clipped_ratio`
is 0.90–0.98 on the fixture, meaning almost every rollout hits
`max_completion_length`, so rewards are computed on truncated text — a reason
to set the completion length from the data rather than the default. And the
logged KL stays near zero, as expected for a policy that barely moves from its
reference in a few steps; on a real run it should rise, and a flat KL with
rising reward is the over-optimisation signature the diagnostics are built for.
