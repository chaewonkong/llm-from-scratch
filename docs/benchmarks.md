# BPE training benchmarks

Pure-Python `RegexTokenizer` (`src/llm_scratch/bpe/py/tokenizer.py`) on FineWeb-Edu
sample-10BT, first `000_00000.parquet`. Run with `scripts/bench_train.py`.
These are the baseline numbers for the Rust port to beat.

| date       | corpus            | vocab | merges | train   | s/merge | chars/token | encode | roundtrip |
|------------|-------------------|-------|--------|---------|---------|-------------|--------|-----------|
| 2026-09-15 | 2 MB (486 docs)   | 300   | 44     | 20.6 s  | 0.47    | 1.43        | 1.9 s  | ok        |
| 2026-09-16 | 2.1 MB (486 docs) | 4096  | 3840   | 982.5 s | 0.26    | 3.47        | 3.0 s  | ok        |

Notes

- Per-merge cost drops over training because `train` recounts every pair over the
  whole token sequence each iteration, and that sequence shrinks as merges apply
  (2.09M bytes -> ~600K tokens at the end). Early merges are the expensive ones.
- Extrapolation for 50 MB at vocab 4096: ~25x the 2 MB run, roughly 7 hours.
