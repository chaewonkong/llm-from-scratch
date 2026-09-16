"""Pure-Python BPE training benchmark on FineWeb-Edu (Phase 1, Day 3).

usage: python scripts/bench_train.py --mb 2 --vocab 512
"""
import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from llm_scratch.bpe.py.tokenizer import RegexTokenizer

DEFAULT_PARQUET = Path.home() / "data/fineweb-edu/sample/10BT/000_00000.parquet"


def load_text(path: Path, n_bytes: int) -> str:
    """Read row groups until we have n_bytes of UTF-8 text (never loads the whole 2GB file)."""
    pf = pq.ParquetFile(path)
    docs, size = [], 0
    for rg in range(pf.num_row_groups):
        for doc in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            docs.append(doc)
            size += len(doc.encode("utf-8"))
            if size >= n_bytes:
                return "\n\n".join(docs)
    return "\n\n".join(docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--mb", type=float, default=2.0)
    ap.add_argument("--vocab", type=int, default=512)
    args = ap.parse_args()

    text = load_text(args.parquet, int(args.mb * 1024 * 1024))
    n_docs = len(text.split("\n\n"))
    print(f"loaded {len(text.encode('utf-8')) / 1e6:.1f} MB, {n_docs} docs, {len(text):,} chars")

    tok = RegexTokenizer()
    t0 = time.perf_counter()
    train_ids = tok.train(text, vocab_size=args.vocab)
    elapsed = time.perf_counter() - t0

    n_merges = len(tok.merges)
    print(f"train: vocab={args.vocab} merges={n_merges} time={elapsed:.1f}s "
          f"({elapsed / max(n_merges, 1):.2f}s/merge)")
    print(f"compression: {len(text) / len(train_ids):.2f} chars/token")

    t0 = time.perf_counter()
    ids = tok.encode(text)
    print(f"encode: {time.perf_counter() - t0:.1f}s, roundtrip ok={tok.decode(ids) == text}")


if __name__ == "__main__":
    main()
