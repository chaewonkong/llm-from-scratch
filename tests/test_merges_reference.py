import json
from pathlib import Path

import pytest

from llm_scratch.bpe.py.regex_tokenizer import RegexTokenizer
from llm_scratch.fineweb import DEFAULT_PARQUET, load_text

FIXTURE = Path(__file__).parent / "fixtures/merges_d75e8ed_2mb_v4096.json"


@pytest.mark.slow
@pytest.mark.skipif(not DEFAULT_PARQUET.exists(), reason="FineWeb-Edu parquet not found")
def test_merges_match_reference():
    """Merges must match the pre-incremental train() (commit d75e8ed) exactly."""
    # given
    ref = json.loads(FIXTURE.read_text())
    meta = ref["meta"]
    text = load_text(DEFAULT_PARQUET, int(meta["mb"] * 1024 * 1024))
    tokenizer = RegexTokenizer()

    # when
    tokenizer.train(text, vocab_size=meta["vocab_size"])

    # then
    merges = [list(p) for p in tokenizer.merges]
    first_diff = next(
        (i for i, (x, y) in enumerate(zip(merges, ref["merges"])) if x != y), None
    )
    assert merges == ref["merges"], f"first diff at merge {first_diff}"
