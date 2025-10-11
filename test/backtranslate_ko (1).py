#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtranslate_ko.py
-------------------
Generate ~N Korean paraphrases via back-translation using Hugging Face MarianMT models.
- Offline-capable after first download (models cached).
- Supports multiple pivot languages (en, ja, zh, fr, de).
- Sampling controls (temperature/top_p/top_k) for diversity.
- Simple deduplication with Jaccard similarity on whitespace tokens.

Usage
-----
python backtranslate_ko.py --text "와, 또 그걸 해냈네. 정말 대단하다." --num 20 --pivots en ja zh --temperature 1.2 --top_p 0.9

Or with stdin:
echo "와, 또 그걸 해냈네. 정말 대단하다." | python backtranslate_ko.py -n 20

Requirements
------------
pip install -U transformers sentencepiece torch accelerate

Notes
-----
- First run will download MarianMT models (~100-300MB each). They are cached under ~/.cache/huggingface.
- If you face memory issues, reduce pivots or batch_size.
"""

import argparse
import sys
import random
from typing import List, Tuple, Iterable, Set

# Lazy import to make startup fast
from transformers import MarianMTModel, MarianTokenizer

# Available pivot language pairs (pivot -> (ko->pivot, pivot->ko) model names)
MODEL_MAP = {
    "en": ("Helsinki-NLP/opus-mt-tc-big-ko-en", "Helsinki-NLP/opus-mt-tc-big-en-ko"),
    "ja": ("Helsinki-NLP/opus-mt-ko-ja", "Helsinki-NLP/opus-mt-ja-ko"),
    "zh": ("Helsinki-NLP/opus-mt-ko-zh", "Helsinki-NLP/opus-mt-zh-ko"),
    "fr": ("Helsinki-NLP/opus-mt-ko-fr", "Helsinki-NLP/opus-mt-fr-ko"),
    "de": ("Helsinki-NLP/opus-mt-ko-de", "Helsinki-NLP/opus-mt-de-ko"),
}

def jaccard(a: str, b: str) -> float:
    sa: Set[str] = set(a.split())
    sb: Set[str] = set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def load_pair(src2tgt_name: str, tgt2src_name: str):
    tok_ab = MarianTokenizer.from_pretrained(src2tgt_name)
    mdl_ab = MarianMTModel.from_pretrained(src2tgt_name)
    tok_ba = MarianTokenizer.from_pretrained(tgt2src_name)
    mdl_ba = MarianMTModel.from_pretrained(tgt2src_name)
    return (tok_ab, mdl_ab, tok_ba, mdl_ba)

def translate_batch(tokenizer, model, texts: List[str], **gen_kwargs) -> List[str]:
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        **inputs,
        do_sample=gen_kwargs.get("do_sample", True),
        temperature=gen_kwargs.get("temperature", 1.0),
        top_p=gen_kwargs.get("top_p", 0.9),
        top_k=gen_kwargs.get("top_k", 50),
        max_new_tokens=gen_kwargs.get("max_new_tokens", 96),
        num_beams=gen_kwargs.get("num_beams", 1),
    )
    return [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]

def backtranslate_once(ko_text: str, pair, **gen_kwargs) -> str:
    tok_ab, mdl_ab, tok_ba, mdl_ba = pair
    # ko -> pivot
    pivot_text = translate_batch(tok_ab, mdl_ab, [ko_text], **gen_kwargs)[0]
    # pivot -> ko
    ko_bt = translate_batch(tok_ba, mdl_ba, [pivot_text], **gen_kwargs)[0]
    return ko_bt

def generate_variants(
    text: str,
    pivots: List[str],
    num: int,
    per_pivot_min: int = 1,
    jaccard_threshold: float = 0.9,
    seed: int = 42,
    **gen_kwargs
) -> List[str]:
    random.seed(seed)
    # Load all required models first (shared cache)
    loaded = {}
    for p in pivots:
        if p not in MODEL_MAP:
            raise ValueError(f"Unsupported pivot: {p}. Choose from {list(MODEL_MAP.keys())}")
        loaded[p] = load_pair(*MODEL_MAP[p])

    variants: List[str] = []
    tried = 0
    pivot_cycle = (p for _ in range(10**9) for p in random.sample(pivots, k=len(pivots)))

    while len(variants) < num and tried < num * 50:
        pivot = next(pivot_cycle)
        pair = loaded[pivot]
        # add controlled randomness each round
        gen_kwargs_round = dict(gen_kwargs)
        # light jitter
        gen_kwargs_round["temperature"] = float(gen_kwargs.get("temperature", 1.0)) * random.uniform(0.9, 1.3)
        gen_kwargs_round["top_p"] = min(0.98, float(gen_kwargs.get("top_p", 0.9)) + random.uniform(-0.05, 0.05))
        gen_kwargs_round["top_k"] = int(max(10, float(gen_kwargs.get("top_k", 50)) + random.uniform(-10, 10)))
        candidate = backtranslate_once(text, pair, **gen_kwargs_round).strip()
        tried += 1

        # basic dedup (skip if too similar to original or to existing)
        if candidate == text or jaccard(candidate, text) >= jaccard_threshold:
            continue
        if any(jaccard(candidate, v) >= jaccard_threshold for v in variants):
            continue
        variants.append(candidate)

    return variants

def main():
    ap = argparse.ArgumentParser(description="Korean back-translation paraphrase generator")
    ap.add_argument("--text", "-t", type=str, help="Input Korean sentence. If omitted, read from stdin")
    ap.add_argument("--num", "-n", type=int, default=20, help="Number of variants to generate")
    ap.add_argument("--pivots", "-p", nargs="+", default=["en"], help="Pivot languages to use")
    ap.add_argument("--temperature", type=float, default=1.1, help="Sampling temperature")
    ap.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p")
    ap.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    ap.add_argument("--max_new_tokens", type=int, default=96, help="Max new tokens for generation")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--print_pivots", action="store_true", help="Also print pivot language used per line")
    args = ap.parse_args()

    if not args.text:
        args.text = sys.stdin.read().strip()
    if not args.text:
        print("No input text. Provide --text or pipe from stdin.", file=sys.stderr)
        sys.exit(1)

    try:
        variants = generate_variants(
            text=args.text,
            pivots=args.pivots,
            num=args.num,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Always include the original first (for reference)
    print("# ORIGINAL")
    print(args.text)
    print("# VARIANTS")
    for i, v in enumerate(variants, 1):
        print(f"{i:02d}. {v}")

if __name__ == "__main__":
    main()
