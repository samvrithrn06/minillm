"""Check minillm against Hugging Face transformers (float32, greedy decoding).

Setup (extra dependencies, only needed for this script):
    pip install torch transformers

Usage:
    python scripts/compare_hf.py --model-dir models/SmolLM2-360M
    python scripts/compare_hf.py --model-dir models/SmolLM2-360M --prompt "Once upon a time" --new-tokens 50

Both implementations are fed the *same* token ids (from the HF tokenizer), so any
difference comes from the model code, not tokenization.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--new-tokens", type=int, default=30)
    ap.add_argument("--max-seq", type=int, default=512)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from minillm import KVCache, LlamaModel, generate

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    hf = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.float32).eval()
    enc = tok(a.prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].tolist()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    with torch.no_grad():
        hf_logits = hf(**enc).logits[0].numpy()  # (T, vocab)
        hf_out = hf.generate(**enc, max_new_tokens=a.new_tokens, do_sample=False,
                             repetition_penalty=1.0, pad_token_id=pad)
    hf_new = hf_out[0, len(prompt_ids):].tolist()

    model = LlamaModel.from_pretrained(a.model_dir, max_seq=a.max_seq)
    our_logits = model.forward(prompt_ids, KVCache(model.cfg), last_only=False)
    our_new, _ = generate(model, prompt_ids, a.new_tokens)

    diff = np.abs(hf_logits - our_logits)
    top1 = (hf_logits.argmax(-1) == our_logits.argmax(-1)).mean()
    n = min(len(hf_new), len(our_new))
    first_div = next((i for i in range(n) if hf_new[i] != our_new[i]), None)

    print(f"prompt tokens: {len(prompt_ids)}")
    print(f"logits max |diff|:  {diff.max():.2e}   mean |diff|: {diff.mean():.2e}")
    print(f"top-1 agreement on prompt positions: {top1:.1%}")
    print(f"generated {len(our_new)} tokens (minillm) vs {len(hf_new)} (HF)")
    if first_div is None and len(hf_new) == len(our_new):
        print("greedy generation: IDENTICAL")
    elif first_div is None:
        print(f"greedy generation: identical over the first {n} tokens (lengths differ, e.g. EOS)")
    else:
        print(f"greedy generation: first difference at token {first_div}")
        print("  HF:     ", repr(tok.decode(hf_new)))
        print("  minillm:", repr(tok.decode(our_new)))


if __name__ == "__main__":
    main()
