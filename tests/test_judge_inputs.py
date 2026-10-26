"""What the reward-model judge and the evaluation harness actually feed the
models: the conversation format and the inference dtype."""

from __future__ import annotations

import pytest

pytest.importorskip("trl")

import torch  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
from trl.data_utils import maybe_apply_chat_template  # noqa: E402

from prefopt import modeling  # noqa: E402
from prefopt.evaluate import RewardModelJudge  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + "
    "'<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _tokenizer(chat_template: str | None) -> PreTrainedTokenizerFast:
    vocab = {"<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        chat_template=chat_template,
    )


def _judge(tok: PreTrainedTokenizerFast) -> RewardModelJudge:
    judge = object.__new__(RewardModelJudge)  # format() needs no model
    judge.tok = tok
    return judge


def test_judge_scores_the_text_the_reward_trainer_trained_on():
    # RewardTrainer renders each conversational pair with the chat template
    # before tokenizing. The judge used to score "prompt\n\nresponse", a
    # format the scalar head never saw in training.
    tok = _tokenizer(CHAT_TEMPLATE)
    prompt, response = "Explain osmosis.", "Water crosses the membrane toward the solute."
    pair = {
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "No."},
        ],
    }
    trained_on = maybe_apply_chat_template(pair, tok)["chosen"]
    assert _judge(tok).format(prompt, response) == trained_on
    assert _judge(tok).format(prompt, response) != f"{prompt}\n\n{response}"


def test_plain_join_only_without_a_chat_template():
    assert _judge(_tokenizer(None)).format("p", "r") == "p\n\nr"


def test_emulated_bf16_is_not_used_for_inference(monkeypatch):
    # A T4 (compute capability 7.5) reports is_bf16_supported() == True
    # through emulation; inference there should run in float32.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    assert modeling.inference_dtype() == torch.float32

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    assert modeling.inference_dtype() == torch.bfloat16
