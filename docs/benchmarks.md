# BPE training benchmarks

Pure-Python `RegexTokenizer` (`src/llm_scratch/bpe/py/tokenizer.py`) on FineWeb-Edu
sample-10BT, first `000_00000.parquet`. Run with `scripts/bench_train.py`.
These are the baseline numbers for the Rust port to beat.

| date       | corpus            | vocab | merges | train   | s/merge | chars/token | encode | roundtrip |
|------------|-------------------|-------|--------|---------|---------|-------------|--------|-----------|
| 2026-09-15 | 2 MB (486 docs)   | 300   | 44     | 20.6 s  | 0.47    | 1.43        | 1.9 s  | ok        |
| 2026-09-16 | 2.1 MB (486 docs) | 4096  | 3840   | 982.5 s | 0.26    | 3.47        | 3.0 s  | ok        |
| 2026-09-16 | 52.5 MB (11071 docs) | 4096 | 3840 | 23514.8 s | 6.12 | 3.44 | 74.5 s | ok |

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
