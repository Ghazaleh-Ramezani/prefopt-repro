"""Check ZeRO-3 sharded training against single-process training, on CPU.

Runs a few optimizer steps of causal-LM loss on the offline fixture model
twice: once as a plain single-process PyTorch loop, once under DeepSpeed
ZeRO stage 3 with two ranks (gloo, DeepSpeed's CPU accelerator). Both see
the same global batch at every step. The check passes when

  * each rank holds only its partition of every parameter (ZeRO-3), and
  * the loss trajectory and the final parameters match the single-process
    run to numerical tolerance.

    python scripts/make_cpu_fixture.py --out fixtures/cpu
    DS_ACCELERATOR=cpu deepspeed --num_gpus 2 scripts/zero3_cpu_check.py   # or:
    DS_ACCELERATOR=cpu torchrun --nproc_per_node 2 scripts/zero3_cpu_check.py

Writes results/zero3_cpu_check.json from rank 0.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STEPS, MICRO, LR = 6, 2, 1e-3


def batches(tok, n_ranks):
    import datasets as hfds

    rows = hfds.load_dataset("parquet", data_files=str(FIX / "prefdata" / "train" / "data.parquet"))["train"]
    texts = [r["chosen"][0]["content"] + "\n" + r["chosen"][-1]["content"] for r in rows]
    global_bs = MICRO * n_ranks
    out = []
    for s in range(STEPS):
        chunk = texts[s * global_bs : (s + 1) * global_bs]
        enc = tok(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        out.append(enc)
    return out


def loss_fn(model, enc):
    labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
    return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss


def reference_run(tok, n_ranks):
    """Single process, full global batch per step, mean of per-rank micro-batch losses."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(FIX / "tiny-model", dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    losses = []
    for enc in batches(tok, n_ranks):
        opt.zero_grad()
        parts = [{k: v[r * MICRO : (r + 1) * MICRO] for k, v in enc.items()} for r in range(n_ranks)]
        loss = sum(loss_fn(model, p) for p in parts) / n_ranks
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, {n: p.detach().clone() for n, p in model.named_parameters()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    ap.add_argument("--out", default="results/zero3_cpu_check.json")
    args = ap.parse_args()

    import deepspeed

    deepspeed.init_distributed(dist_backend="gloo")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    tok = AutoTokenizer.from_pretrained(FIX / "tiny-model")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = {
        "train_micro_batch_size_per_gpu": MICRO,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0},
        "zero_allow_untested_optimizer": True,
        "gradient_clipping": 0.0,
    }
    # The single-process reference runs first, before ZeRO-3 is switched on.
    reference = reference_run(tok, world) if rank == 0 else None
    torch.distributed.barrier()

    from transformers.integrations import HfDeepSpeedConfig

    # Keeping this object alive tells from_pretrained that ZeRO-3 is active,
    # so weights are partitioned across ranks as they load.
    ds_hf = HfDeepSpeedConfig(cfg)  # noqa: F841
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(FIX / "tiny-model", dtype=torch.float32)
    n_full = sum(p.ds_numel for p in model.parameters())
    n_local = sum(p.ds_tensor.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=cfg)

    losses = []
    for enc in batches(tok, world):
        part = {k: v[rank * MICRO : (rank + 1) * MICRO] for k, v in enc.items()}
        loss = loss_fn(engine, part)
        engine.backward(loss)
        engine.step()
        mean = loss.detach().clone()
        torch.distributed.all_reduce(mean)
        losses.append((mean / world).item())

    params = {}
    for n, p in engine.module.named_parameters():
        with deepspeed.zero.GatheredParameters([p]):
            params[n] = p.detach().clone()

    if rank == 0:
        ref_losses, ref_params = reference
        max_param_diff = max((params[n] - ref_params[n]).abs().max().item() for n in ref_params)
        max_loss_diff = max(abs(a - b) for a, b in zip(losses, ref_losses))
        report = {
            "world_size": world,
            "zero_stage": 3,
            "backend": "gloo (DeepSpeed CPU accelerator)",
            "deepspeed_version": deepspeed.__version__,
            "torch_version": torch.__version__,
            "parameters_total": int(n_full),
            "parameters_held_by_rank0": int(n_local),
            "rank0_share": n_local / n_full,
            "steps": STEPS,
            "loss_zero3": losses,
            "loss_single_process": ref_losses,
            "max_abs_loss_difference": max_loss_diff,
            "max_abs_parameter_difference": max_param_diff,
            "match": bool(max_loss_diff < 1e-4 and max_param_diff < 1e-4),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if not k.startswith("loss_")}, indent=2))


FIX = Path(os.environ.get("FIX", "fixtures/cpu"))

if __name__ == "__main__":
    main()
