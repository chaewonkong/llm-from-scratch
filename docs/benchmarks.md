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

Inverted index + heap (`pair -> word idx` index, commit `b317614`; max-heap with lazy deletion for best pair, commit `243c710`, 2026-10-03)

| date       | corpus            | vocab | merges | train   | s/merge | chars/token | encode | roundtrip | note                 |
|------------|-------------------|-------|--------|---------|---------|-------------|--------|-----------|----------------------|
| 2026-10-03 | 2.1 MB (486 docs) | 300   | 44     | 0.7 s   | 0.02    | 1.43        | 0.4 s  | ok        | inverted index + heap |
| 2026-10-03 | 2.1 MB (486 docs) | 4096  | 3840   | 1.6 s   | <0.01   | 3.47        | 0.5 s  | ok        | inverted index + heap |
| 2026-10-03 | 52.5 MB (11071 docs) | 4096 | 3840 | 16.6 s | <0.01 | 3.44 | 7.2 s | ok | inverted index + heap |

Fewer heap pushes (day 3.8: push only pairs whose count went up, re-push decreased pairs lazily on pop, commit `d8271cb`, 2026-10-04)

| date       | corpus            | vocab | merges | train   | s/merge | chars/token | encode | roundtrip | note          |
|------------|-------------------|-------|--------|---------|---------|-------------|--------|-----------|---------------|
| 2026-10-04 | 2.1 MB (486 docs) | 300   | 44     | 0.6 s   | 0.01    | 1.43        | 0.4 s  | ok        | lazy heap push |
| 2026-10-04 | 2.1 MB (486 docs) | 4096  | 3840   | 1.2 s   | <0.01   | 3.47        | 0.6 s  | ok        | lazy heap push |
| 2026-10-04 | 52.5 MB (11071 docs) | 4096 | 3840 | 13.0 s | <0.01 | 3.44 | 7.1 s | ok | lazy heap push |

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
- Inverted index + heap (2026-10-03): two changes, measured together. First,
  `pair_word_idx_map` maps each pair to the words that contain it, so a merge calls
  `_merge` only on those words instead of on every word. Second, a max-heap
  (`(-count, pair)`) with lazy deletion picks the best pair: after each merge, every
  pair whose count changed is pushed again, and a popped entry whose count no longer
  matches `count` is skipped. Ties still break on the smallest pair, the same as the
  old `min(count, key=lambda k: (-count[k], k))`. `tests/test_merges_reference.py`
  (`-m slow`) passes, so the merges are identical to the reference.
  Compression (3.47 / 3.44) and roundtrip are unchanged.
- Train drops 39x at 2 MB / 4096 (62.1 s -> 1.6 s) and 111x at 50 MB / 4096
  (1844.1 s -> 16.6 s). The incremental-count commit was never run at 50 MB, so the
  111x covers incremental counts, the inverted index and the heap together. Against
  the first baseline, 50 MB / 4096 went from 6.5 h to 16.6 s (~1400x).
- Where the time goes (2026-10-03, measured, see "Train profile" below): at 2 MB / 4096
  setup (regex, `word_counter`, initial count) is 0.24 s and the merge loop is 1.42 s.
  The vocab 300 run is 0.70 s because the first 44 merges are the heavy ones: they
  touch 66K words, against 179K for all 3840 merges. The s/merge column now rounds to
  0.00, so it says little.
- Train time now grows slower than the corpus: 25x the data gives 10.4x the train
  time (1.6 s -> 16.6 s) and 14x the encode time (0.5 s -> 7.2 s). Both depend on
  the number of unique chunks more than on total size.
- Encode was not touched, but 50 MB encode is 7.2 s versus 74.7 s in the last 50 MB
  run (2026-09-19). That 10x comes from the encode cache (2026-09-27), which was never
  run at 50 MB. The 2 MB encode change (0.7 s -> 0.5 s) is noise.
- Updated reference point for the Rust port: 50 MB / vocab 4096 must beat 16.6 s
  train and 7.2 s encode (~7 MB/s). Python is now close enough that the Rust port
  should be compared on a larger corpus or vocab as well.
- Fewer heap pushes (2026-10-04, day 3.8): after a merge, only pairs that contain the
  new token are pushed, since only those counts go up. Pairs whose count went down are
  not pushed. Their old heap entry has a higher count, so it still pops first. On pop,
  if `count` is lower than the entry, the pair is pushed again with its current count.
  If the pair is gone from `count`, the entry is dropped. Compression (1.43 / 3.47 /
  3.44) and roundtrip are unchanged.
- Train drops 1.33x at 2 MB / 4096 (1.6 s -> 1.2 s) and 1.28x at 50 MB / 4096
  (16.6 s -> 13.0 s). The 3.6 s saved at 50 MB is close to the 3.3 s that heap push
  took in the profile below, so heap push was the step this change removed. The
  profile below is from `243c710` and was not re-run for this commit.
- Encode was not touched; 0.4 / 0.6 / 7.1 s versus 0.4 / 0.5 / 7.2 s is noise.
- Updated reference point for the Rust port: 50 MB / vocab 4096 must beat 13.0 s
  train and 7.1 s encode.

Train profile (2026-10-03, commit `243c710`)

Phase timers around a line-for-line copy of `train` / `encode_ordinary`. The copy's
merges and encoded ids match the real methods; timer overhead is ~3% (50 MB: 17.41 s
instrumented vs 16.92 s real `train()`). Median of 5 reps (2 MB) and 3 reps (50 MB).

| phase                                    | 2 MB / 4096 | %     | 50 MB / 4096 | %     |
|------------------------------------------|-------------|-------|--------------|-------|
| 1. `re.findall`                          | 0.128 s     | 7.7   | 3.097 s      | 17.8  |
| 2. `word_counter` (chunk hashing)        | 0.052 s     | 3.1   | 1.453 s      | 8.3   |
| 3. initial pair count + inverted index (+ `heapify`) | 0.063 s | 3.8 | 0.474 s | 2.7 |
| 4. merge loop                            | 1.420 s     | 85.2  | 12.386 s     | 71.1  |
|    heap pop (incl. stale entries)        | 0.212 s     | 12.8  | 0.962 s      | 5.5   |
|    `_merge`                              | 0.155 s     | 9.3   | 1.060 s      | 6.1   |
|    count subtract (old pairs)            | 0.238 s     | 14.3  | 1.694 s      | 9.7   |
|    count add + index add (new pairs)     | 0.314 s     | 18.9  | 2.268 s      | 13.0  |
|    `gone` set diff + index discard       | 0.355 s     | 21.3  | 3.062 s      | 17.6  |
|    heap push (touched pairs)             | 0.142 s     | 8.5   | 3.317 s      | 19.1  |
|    index pop                             | 0.004 s     | 0.3   | 0.023 s      | 0.1   |

| counter        | 2 MB    | 50 MB      |
|----------------|---------|------------|
| regex chunks   | 412,795 | 10,240,641 |
| unique words   | 36,660  | 248,596    |
| initial pairs  | 2,873   | 8,532      |
| word visits (all merges) | 178,895 | 1,128,256 |
| heap pushes    | 244,597 | 1,035,891  |
| heap pops      | 135,944 | 387,260    |

Encode profile, 50 MB / 4096: `re.findall` 3.10 s, `set(chunks)` 0.36 s,
`_encode_chunk` over unique chunks 3.04 s, building the id list 0.93 s.

- The merge loop is 71% of train at 50 MB. All of it is Python dict / set / tuple /
  heap work on small int pairs. No single step dominates; the largest is heap push (19%).
- Heap bookkeeping is mostly waste: 387K pops for 3840 merges means ~99% of popped
  entries are stale, and ~1M pushes are made to support that.
- `re.findall` is 18% of train and 42% of encode at 50 MB, and runs once in each. It is
  already C (the `regex` package), so a rewrite does not automatically speed it up.
- Amdahl bound at 50 MB / 4096: replacing only the merge loop with code that takes zero
  time still leaves 5.02 s of phases 1-3 (3.4x at most). Beyond 3.4x, phases 1-2 must
  change too.
