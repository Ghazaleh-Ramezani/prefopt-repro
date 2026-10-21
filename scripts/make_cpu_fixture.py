"""Build an offline fixture for exercising the whole pipeline on a CPU.

Writes a 0.67M-parameter Qwen2-architecture model with a byte-level BPE
tokenizer and a synthetic preference dataset to `--out` (default
`fixtures/cpu`). Nothing is downloaded, so the full SFT -> RM -> DPO -> GRPO
-> evaluate -> diagnostics -> report path runs on a laptop with no GPU and no
Hub access, which is what `scripts/cpu_validation.sh` does.

The numbers such a model produces say nothing about DPO or GRPO. What the run
checks is that every stage consumes the previous stage's output and that the
evaluation path rebuilds the model each stage trained.

Chosen responses are deliberately longer and more structured than rejected
ones, so the verbosity bias real preference data carries is present here too,
which gives `length_reward_corr` something to detect.

    python scripts/make_cpu_fixture.py --out fixtures/cpu
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import datasets as hfds
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

TOPICS = [
    "photosynthesis", "gradient descent", "the water cycle", "binary search",
    "inflation", "antibiotics", "tectonic plates", "neural networks", "osmosis",
    "the Doppler effect", "supply and demand", "entropy", "vaccines", "recursion",
    "black holes", "enzymes", "compound interest", "convection", "alloys", "sorting",
]
VERBS = ["explain", "describe", "summarise", "outline", "clarify", "walk me through"]
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + "
    "'<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _prompt(i: int) -> str:
    return f"{VERBS[i % len(VERBS)].capitalize()} {TOPICS[i % len(TOPICS)]} in simple terms."


def _chosen(i: int) -> str:
    t = TOPICS[i % len(TOPICS)]
    return (
        f"Here is a clear explanation of {t}. First, the core idea is that {t} "
        f"describes a process that changes over time. Second, the key mechanism "
        f"involves an exchange between two parts of the system. Third, the "
        f"practical consequence is that small changes early on produce large "
        f"differences later. In short, {t} is best understood as a sequence of "
        f"steps where each step depends on the one before it."
    )


def _rejected(i: int) -> str:
    return f"{TOPICS[i % len(TOPICS)]} is a thing that happens. Look it up."


def _messages(prompt: str, reply: str) -> list[dict]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]


def _row(i: int) -> dict:
    # One schema for every split: `chosen`/`rejected` are full conversations
    # (the implicit-prompt preference format) and `messages` serves SFT,
    # matching the shape of ultrafeedback_binarized.
    p = _prompt(i)
    return {
        "prompt": p,
        "chosen": _messages(p, _chosen(i)),
        "rejected": _messages(p, _rejected(i)),
        "messages": _messages(p, _chosen(i)),
    }


def build(out: Path, n_train: int = 240, n_test: int = 60, seed: int = 0) -> None:
    model_dir, data_dir = out / "tiny-model", out / "prefdata"
    train = [_row(i) for i in range(n_train)]
    test = [_row(i + 1000) for i in range(n_test)]  # disjoint prompt ids

    corpus = []
    for r in train + test:
        corpus += [r["prompt"], r["chosen"][-1]["content"], r["rejected"][-1]["content"]]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=2000,
            special_tokens=SPECIALS,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        bos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        chat_template=CHAT_TEMPLATE,
        # Causal LMs reject token_type_ids in generate(); Qwen's own tokenizer
        # does not emit them either.
        model_input_names=["input_ids", "attention_mask"],
    )
    fast.save_pretrained(model_dir)

    cfg = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    # Seeded so that every run of cpu_validation.sh starts from the same
    # weights and reproduces the published artifacts.
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(model_dir)

    if data_dir.exists():
        shutil.rmtree(data_dir)
    # Split names are the ones `datasets` maps from a directory layout.
    for name, rows in [("train", train), ("test", test), ("validation", train)]:
        split_dir = data_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        hfds.Dataset.from_list(rows).to_parquet(str(split_dir / "data.parquet"))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"fixture: {n_params / 1e6:.2f}M parameters, {n_train} train / {n_test} test pairs -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="fixtures/cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
