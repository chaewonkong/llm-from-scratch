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

Encode cache (unique-chunk memoization in `encode_ordinary`, `src/llm_scratch/bpe/py/regex_tokenizer.py`, uncommitted as of 2026-09-27)

| date       | corpus            | vocab | merges | train   | s/merge | chars/token | encode | roundtrip | note         |
|------------|-------------------|-------|--------|---------|---------|-------------|--------|-----------|--------------|
| 2026-09-27 | 2.1 MB (486 docs) | 4096  | 3840   | 182.9 s | 0.05    | 3.47        | 0.6 s  | ok        | encode cache |

Incremental pair counts (`train` updates `count` per merge instead of recounting, commit `14204ba`, 2026-10-03)

| date       | corpus            | vocab | merges | train   | s/merge | chars/token | encode | roundtrip | note              |
|------------|-------------------|-------|--------|---------|---------|-------------|--------|-----------|-------------------|
| 2026-10-03 | 2.1 MB (486 docs) | 4096  | 3840   | 62.1 s  | 0.02    | 3.47        | 0.7 s  | ok        | incremental count |

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
- Encode cache (2026-09-27): `encode_ordinary` now runs `_encode_chunk` once per unique
  regex chunk and reuses the result for repeats, the same idea as the train refactor.
  Compression (3.47) and roundtrip are unchanged; encode drops 5x at 2 MB / 4096
  (3.0 s -> 0.6 s). Train time is unchanged (185.3 s -> 182.9 s, noise).
  `scripts/bench_train.py` now imports `RegexTokenizer` from `regex_tokenizer.py`.
- Incremental pair counts (2026-10-03): `train` builds `count` once before the loop.
  After each merge it subtracts the old pairs and adds the new pairs, but only for
  words the merge actually changed. Before this change, every iteration recounted
  pairs over all unique chunks. Compression (3.47) and roundtrip are unchanged.
  Train drops 2.9x at 2 MB / 4096 (182.9 s -> 62.1 s). Encode moves 0.6 s -> 0.7 s,
  which is noise because encode was not touched. Each iteration still calls `_merge`
  on every word and picks the best pair with a linear `min` over `count`, so those
  two steps are the next bottlenecks.
