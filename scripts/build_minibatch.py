"""Build and freeze the fixed minibatch used by the curvature probe.

Steps:
  1) Tokenize FIXED_PROMPTS (truncate to max_seq_len).
  2) Optionally: greedy-generate continuations from the EARLIER checkpoint to
     better approximate the on-policy distribution at the analyzed checkpoint.
     This is a proxy for GRPO rollouts (which we don't have for public pairs).
  3) Save token id tensors to results/<pair>/minibatch.pt.

Each minibatch_size group of prompts becomes one minibatch tensor of shape
(batch_size, seq_len). All sequences in a minibatch are right-padded with the
tokenizer's pad token; an attention mask is saved alongside.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import FIXED_PROMPTS, device, load_config, results_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    mb_cfg = cfg["minibatch"]
    out = results_root(cfg["pair_name"])

    tok = AutoTokenizer.from_pretrained(cfg["earlier_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    seq_len = int(mb_cfg["max_seq_len"])
    bs = int(mb_cfg["batch_size"])
    nb = int(mb_cfg["num_minibatches"])
    needed = bs * nb
    if needed > len(FIXED_PROMPTS):
        raise ValueError(
            f"need {needed} prompts (bs={bs}, nb={nb}) but only have "
            f"{len(FIXED_PROMPTS)} fixed prompts"
        )
    prompts = FIXED_PROMPTS[:needed]

    sequences: list[str]
    if mb_cfg.get("use_model_continuations", True):
        print("[minibatch] generating greedy continuations from earlier checkpoint...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["earlier_model"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device())
        model.eval()
        max_new = int(mb_cfg["max_new_tokens"])
        sequences = []
        with torch.no_grad():
            for i, p in enumerate(prompts):
                ids = tok(p, return_tensors="pt", truncation=True,
                          max_length=seq_len - max_new).to(device())
                gen = model.generate(
                    **ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                text = tok.decode(gen[0], skip_special_tokens=True)
                sequences.append(text)
                print(f"  [{i+1}/{len(prompts)}] {len(text)} chars")
        del model
        torch.cuda.empty_cache()
    else:
        sequences = list(prompts)

    enc = tok(
        sequences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = enc["input_ids"]            # (needed, seq_len)
    attention_mask = enc["attention_mask"]  # (needed, seq_len)

    minibatches = []
    for b in range(nb):
        s = slice(b * bs, (b + 1) * bs)
        minibatches.append(
            {
                "input_ids": input_ids[s].clone(),
                "attention_mask": attention_mask[s].clone(),
            }
        )

    payload = {
        "minibatches": minibatches,
        "tokenizer": cfg["earlier_model"],
        "seq_len": seq_len,
        "batch_size": bs,
        "pad_token_id": tok.pad_token_id,
        "used_continuations": bool(mb_cfg.get("use_model_continuations", True)),
        "raw_sequences": sequences,
    }
    torch.save(payload, out / "minibatch.pt")
    print(f"[minibatch] saved {nb} minibatches of size {bs} to {out/'minibatch.pt'}")


if __name__ == "__main__":
    main()
