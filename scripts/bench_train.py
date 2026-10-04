"""Pure-Python BPE training benchmark on FineWeb-Edu (Phase 1, Day 3).

usage: python scripts/bench_train.py --mb 2 --vocab 512
"""

import argparse
import time
from pathlib import Path

from llm_scratch.bpe.regex_tokenizer import RegexTokenizer
from llm_scratch.fineweb import DEFAULT_PARQUET, load_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--mb", type=float, default=2.0)
    ap.add_argument("--vocab", type=int, default=512)
    args = ap.parse_args()

    text = load_text(args.parquet, int(args.mb * 1024 * 1024))
    n_docs = len(text.split("\n\n"))
    print(
        f"loaded {len(text.encode('utf-8')) / 1e6:.1f} MB, {n_docs} docs, {len(text):,} chars"
    )

    tok = RegexTokenizer()
    t0 = time.perf_counter()
    tok.train(text, vocab_size=args.vocab)
    elapsed = time.perf_counter() - t0

    n_merges = len(tok.merges)
    print(
        f"train: vocab={args.vocab} merges={n_merges} time={elapsed:.1f}s "
        f"({elapsed / max(n_merges, 1):.2f}s/merge)"
    )

    t0 = time.perf_counter()
    ids = tok.encode(text)
    encode_elapsed = time.perf_counter() - t0
    print(f"compression: {len(text) / len(ids):.2f} chars/token")
    print(f"encode: {encode_elapsed:.1f}s, roundtrip ok={tok.decode(ids) == text}")


if __name__ == "__main__":
    main()
