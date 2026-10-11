"""Command line interface:  minillm generate ...  /  minillm bench ..."""
from __future__ import annotations

import argparse
import sys

from .bench import dump_json, format_markdown, run_bench
from .cache import BlockPool, PagedKVCache
from .generate import SamplingParams, generate
from .loader import LazyWeights
from .model import LlamaConfig, LlamaModel
from .quant import QUANT_MODES
from .speculative import speculative_generate


def _tokenizer(model_dir: str):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("The CLI needs the 'tokenizers' package:  pip install 'minillm[cli]'")
    return Tokenizer.from_file(f"{model_dir}/tokenizer.json")


def _cmd_generate(a) -> None:
    tok = _tokenizer(a.model_dir)
    model = LlamaModel.from_pretrained(a.model_dir, quant=a.quant, max_seq=a.max_seq)
    ids = tok.encode(a.prompt).ids
    if a.draft_dir:
        draft = LlamaModel.from_pretrained(a.draft_dir, quant=a.draft_quant, max_seq=a.max_seq)
        out, stats = speculative_generate(model, draft, ids, a.max_new_tokens, k=a.spec_k)
    else:
        cache = None
        if a.paged:
            pool = BlockPool(model.cfg, num_blocks=-(-model.cfg.max_seq // 16), block_size=16)
            cache = PagedKVCache(pool, model.cfg.max_seq)
        params = SamplingParams(a.temperature, a.top_k, a.top_p, a.seed)
        out, stats = generate(model, ids, a.max_new_tokens, params, cache=cache)
    print(a.prompt + tok.decode(out))
    print("\n--- stats ---")
    for k, v in stats.items():
        print(f"{k:>24}: {v:.3f}" if isinstance(v, float) else f"{k:>24}: {v}")


def _cmd_bench(a) -> None:
    tok = _tokenizer(a.model_dir)
    cfg = LlamaConfig.from_hf_json(a.model_dir, max_seq=a.max_seq)
    with open(a.text_file) as f:
        ids = tok.encode(f.read()).ids
    draft = None
    if a.draft_dir:
        draft = LlamaModel.from_pretrained(a.draft_dir, quant="none", max_seq=a.max_seq)
    report = run_bench(
        cfg, LazyWeights(a.model_dir), ids,
        quants=a.quants, batch_sizes=a.batch_sizes, new_tokens=a.new_tokens,
        draft=draft, spec_k=a.spec_k,
    )
    print(format_markdown(report))
    if a.out:
        dump_json(report, a.out)
        print(f"\nsaved {a.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="minillm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate text from a local checkpoint")
    g.add_argument("--model-dir", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new-tokens", type=int, default=64)
    g.add_argument("--max-seq", type=int, default=1024)
    g.add_argument("--quant", choices=QUANT_MODES, default="none")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--paged", action="store_true", help="use the paged KV cache")
    g.add_argument("--draft-dir", help="small draft model dir: enables speculative decoding (greedy)")
    g.add_argument("--draft-quant", choices=QUANT_MODES, default="none")
    g.add_argument("--spec-k", type=int, default=4)
    g.set_defaults(fn=_cmd_generate)

    b = sub.add_parser("bench", help="benchmark quantization, batching, speculation")
    b.add_argument("--model-dir", required=True)
    b.add_argument("--text-file", required=True, help="plain text used for perplexity and prompts")
    b.add_argument("--max-seq", type=int, default=512)
    b.add_argument("--new-tokens", type=int, default=32)
    b.add_argument("--quants", nargs="+", choices=QUANT_MODES, default=list(QUANT_MODES))
    b.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    b.add_argument("--draft-dir", help="draft model dir for the speculative-decoding benchmark")
    b.add_argument("--spec-k", type=int, default=4)
    b.add_argument("--out", help="write the full report as JSON")
    b.set_defaults(fn=_cmd_bench)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
