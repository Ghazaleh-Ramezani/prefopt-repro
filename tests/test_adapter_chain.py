"""Reloading a DPO/GRPO run must rebuild the model that was trained.

DPO starts from the SFT adapter merged into the base and trains a new adapter
on top. Applying the saved DPO adapter to the raw base drops the SFT delta.
This builds a tiny randomly initialised model locally (no download), so it
runs on CPU in a few seconds.
"""

from __future__ import annotations

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, Qwen2Config  # noqa: E402

from prefopt.modeling import adapter_chain, load_for_inference  # noqa: E402


def _tiny_base(path):
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)


def _perturbed_adapter(model, seed):
    """A LoRA adapter whose B matrices are non-zero, so it changes the model."""
    torch.manual_seed(seed)
    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    )
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.5)
    return peft_model


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


@pytest.fixture()
def runs(tmp_path):
    base = tmp_path / "base"
    _tiny_base(base)

    # SFT: adapter on the base.
    sft_dir = tmp_path / "runs" / "sft"
    sft = _perturbed_adapter(AutoModelForCausalLM.from_pretrained(base), seed=1)
    sft.save_pretrained(sft_dir)
    (sft_dir / "config.resolved.yaml").write_text(yaml.safe_dump({"init_adapter": None}))

    # DPO: merge SFT into the base, then a new adapter on the merged model --
    # exactly what load_policy + DPOTrainer do.
    merged = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base), sft_dir
    ).merge_and_unload()
    dpo = _perturbed_adapter(merged, seed=2)
    ids = torch.randint(0, 64, (1, 12))
    # Forward through the PeftModel with the adapter active; merging here would
    # strip the LoRA layers in place and save an empty adapter.
    trained_logits = _logits(dpo, ids)

    dpo_dir = tmp_path / "runs" / "dpo"
    dpo.save_pretrained(dpo_dir)
    (dpo_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump({"init_adapter": str(sft_dir)})
    )
    return {"base": str(base), "sft": str(sft_dir), "dpo": str(dpo_dir), "ids": ids,
            "trained_logits": trained_logits}


def test_chain_walks_back_to_the_sft_adapter(runs):
    assert adapter_chain(runs["dpo"]) == [runs["sft"], runs["dpo"]]
    assert adapter_chain(runs["sft"]) == [runs["sft"]]


def test_reload_reproduces_the_trained_model(runs):
    reloaded = load_for_inference(runs["dpo"], runs["base"], dtype=torch.float32)
    torch.testing.assert_close(
        _logits(reloaded, runs["ids"]), runs["trained_logits"], atol=1e-5, rtol=1e-5
    )


def test_adapter_on_raw_base_is_a_different_model(runs):
    # The behaviour this replaces: the DPO adapter applied straight to the raw
    # base. It loads without error and scores a model that was never trained.
    raw = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(runs["base"]), runs["dpo"]
    ).merge_and_unload()
    gap = (_logits(raw, runs["ids"]) - runs["trained_logits"]).abs().max().item()
    assert gap > 1e-3


def test_missing_init_adapter_fails_loudly(runs, tmp_path):
    (tmp_path / "runs" / "dpo" / "config.resolved.yaml").write_text(
        yaml.safe_dump({"init_adapter": str(tmp_path / "gone")})
    )
    with pytest.raises(FileNotFoundError, match="init_adapter"):
        adapter_chain(runs["dpo"])
