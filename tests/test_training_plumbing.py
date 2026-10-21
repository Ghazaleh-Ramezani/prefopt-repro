"""Regression tests for the dataset shapes and the gradient path.

Each of these failed silently or late before it was fixed: the SFT and
preference loaders handed TRL columns it misreads, and reentrant gradient
checkpointing left every LoRA adapter in the model body without a gradient.
"""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")

from prefopt.config import RunCfg  # noqa: E402
from prefopt.data import load_preference_dataset, load_sft_dataset  # noqa: E402


def _msgs(prompt: str, reply: str) -> list[dict]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]


@pytest.fixture()
def implicit_prompt_dataset(tmp_path):
    """The ultrafeedback_binarized shape: a string `prompt` alongside
    full-conversation `chosen`, `rejected` and `messages`."""
    rows = [
        {
            "prompt": f"question {i}",
            "chosen": _msgs(f"question {i}", "a long and careful answer"),
            "rejected": _msgs(f"question {i}", "no"),
            "messages": _msgs(f"question {i}", "a long and careful answer"),
        }
        for i in range(20)
    ]
    for split in ("train", "test", "validation"):
        d = tmp_path / split
        d.mkdir()
        datasets.Dataset.from_list(rows).to_parquet(str(d / "data.parquet"))
    cfg = RunCfg()
    cfg.data.dataset = str(tmp_path)
    cfg.data.train_split, cfg.data.eval_split, cfg.data.sft_train_split = (
        "train",
        "test",
        "validation",
    )
    cfg.data.max_train_samples, cfg.data.max_eval_samples = 20, 20
    return cfg


def test_sft_loader_keeps_only_the_message_list(implicit_prompt_dataset):
    # A string `prompt` next to `messages` makes TRL render the string as a
    # conversation and crash in apply_chat_template.
    data = load_sft_dataset(implicit_prompt_dataset)
    assert data["train"].column_names == ["messages"]
    assert data["eval"].column_names == ["messages"]


def test_preference_loader_drops_the_string_prompt(implicit_prompt_dataset):
    # TRL accepts an explicit prompt (all message lists) or an implicit one
    # (no `prompt` key). A string prompt beside full conversations is neither.
    data = load_preference_dataset(implicit_prompt_dataset)
    for split in ("train", "eval"):
        assert set(data[split].column_names) == {"chosen", "rejected"}


def test_default_checkpointing_reaches_every_lora_adapter():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    )
    # Exactly what the four training stages pass to their trainer config.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": RunCfg().train.gradient_checkpointing_use_reentrant
        }
    )
    model.config.use_cache = False
    model.train()
    with torch.no_grad():  # LoRA B starts at zero; make it non-zero so A gets a gradient
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.1)

    model(torch.randint(0, 64, (2, 8))).logits.sum().backward()

    lora = {n: p for n, p in model.named_parameters() if "lora_" in n}
    starved = [n for n, p in lora.items() if p.grad is None or not p.grad.abs().sum()]
    assert lora and not starved, f"no gradient reached: {starved}"
