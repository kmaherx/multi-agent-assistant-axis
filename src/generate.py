"""Model loading + batched generation."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0  # greedy
    top_p: float = 1.0
    do_sample: bool = False


def load_model_and_tokenizer(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def _apply_chat(tokenizer, messages_list):
    """Apply the chat template to a list of message lists; return list[str]."""
    return [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]


@torch.inference_mode()
def batched_generate(
    model,
    tokenizer,
    messages_list,
    gen_config: GenConfig,
    steering_ctx: contextlib.AbstractContextManager | None = None,
    batch_size: int = 16,
) -> list[str]:
    """Apply chat template, tokenize with left padding, generate, decode only new tokens."""
    prompts = _apply_chat(tokenizer, messages_list)

    out_texts: list[str] = []
    ctx = steering_ctx if steering_ctx is not None else contextlib.nullcontext()
    with ctx:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)
            gen_kwargs = dict(
                max_new_tokens=gen_config.max_new_tokens,
                do_sample=gen_config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if gen_config.do_sample:
                gen_kwargs["temperature"] = gen_config.temperature
                gen_kwargs["top_p"] = gen_config.top_p
            outputs = model.generate(**enc, **gen_kwargs)
            new_tokens = outputs[:, enc["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            out_texts.extend(decoded)
    return out_texts
