# BPE training benchmarks

Pure-Python `RegexTokenizer` (`src/llm_scratch/bpe/py/tokenizer.py`) on FineWeb-Edu
sample-10BT, first `000_00000.parquet`. Run with `scripts/bench_train.py`.
These are the baseline numbers for the Rust port to beat.

| date       | corpus            | vocab | merges | train   | s/merge | chars/token | encode | roundtrip |
|------------|-------------------|-------|--------|---------|---------|-------------|--------|-----------|
| 2026-09-15 | 2 MB (486 docs)   | 300   | 44     | 20.6 s  | 0.47    | 1.43        | 1.9 s  | ok        |
| 2026-09-16 | 2.1 MB (486 docs) | 4096  | 3840   | 982.5 s | 0.26    | 3.47        | 3.0 s  | ok        |
| 2026-09-16 | 52.5 MB (11071 docs) | 4096 | 3840 | 23514.8 s | 6.12 | 3.44 | 74.5 s | ok |

Refactored (`Counter`-based pair counting, uncommitted as of 2026-09-19)

| date       | corpus            | vocab | merges | train   | s/merge | chars/token | encode | roundtrip | note       |
|------------|-------------------|-------|--------|---------|---------|-------------|--------|-----------|------------|
| 2026-09-19 | 2.1 MB (486 docs) | 300   | 44     | 2.7 s   | 0.06    | 1.43        | 1.9 s  | ok        | refactored |
| 2026-09-19 | 2.1 MB (486 docs) | 4096  | 3840   | 185.3 s | 0.05    | 3.47        | 3.0 s  | ok        | refactored |
| 2026-09-19 | 52.5 MB (11071 docs) | 4096 | 3840 | 1844.1 s | 0.48 | 3.44 | 74.7 s | ok | refactored |

Notes

- Per-merge cost drops over training because `train` recounts every pair over the
  whole token sequence each iteration, and that sequence shrinks as merges apply
  (2.09M bytes -> ~600K tokens at the end). Early merges are the expensive ones.
- The 50 MB run confirms the linear extrapolation: 25x the corpus gave 23.9x the
  train time (6.5 h), 23.5x the s/merge and 24.8x the encode time.
- Compression barely moves with corpus size (3.47 -> 3.44 chars/token). At vocab 4096
  the bottleneck is vocab size, not data; more data is not worth more Python hours.
- Reference point for the Rust port: 50 MB / vocab 4096 must beat 6.5 h train and
  74.5 s encode (~0.7 MB/s).
- Refactored (2026-09-19): `train` now counts pairs over unique regex chunks weighted
  by frequency instead of over the full token sequence. Merges, vocab, compression and
  encode time are unchanged; only train time drops (7.6x at 2 MB / 300, 5.3x at
  2 MB / 4096, 12.8x at 50 MB / 4096). The gain grows with corpus size because the
  number of unique chunks grows much slower than the total chunk count.
- Updated reference point for the Rust port: 50 MB / vocab 4096 must beat 31 min
  (1844 s) train and 74.7 s encode.
