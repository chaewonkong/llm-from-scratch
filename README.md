# LLM From Scratch

Building a GPT-style LLM from scratch, one component at a time.

## Status

- [x] BPE tokenizer (byte-level, GPT-4 style regex split) in pure Python
- [ ] BPE tokenizer in Rust (PyO3/maturin), in progress
- [ ] Model, training, inference

## Layout

```
src/llm_scratch/bpe/   Python BPE (Tokenizer, RegexTokenizer)
rust/bpe/              Rust port (bpe-rs)
scripts/               benchmarks
tests/                 tests (`pytest -m slow` for reference merge check)
docs/                  benchmark notes
```

## Benchmarks

BPE training on FineWeb-Edu (50 MB, vocab 4096), pure Python:

| version                           | train  | encode |
|-----------------------------------|--------|--------|
| baseline                          | 6.5 h  | 74.5 s |
| incremental counts + index + heap | 13.0 s | 7.1 s  |

~1800x faster train through algorithmic changes alone. Full history and profiles in
[docs/benchmarks.md](docs/benchmarks.md).

## Quick start

Requires an NVIDIA GPU with CUDA 13.0 (`torch` is pinned to the `cu130` build).

```bash
uv sync
uv run pytest
uv run python scripts/bench_train.py --mb 2 --vocab 4096
```
