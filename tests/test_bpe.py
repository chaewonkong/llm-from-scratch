from llm_scratch.bpe.py.bpe import BPE


def test_most_common_pairs():
    bpe = BPE()
    ids = [1,1,2,3,4,2,3,1,2,1,2]
    assert bpe.most_common_pairs(ids) == (1,2)

def test_merge():
    bpe = BPE()
    ids = [1,2,1,3,1,2]
    assert bpe.merge(ids, (1,2), 99) == [99, 1, 3, 99]

def test_round_trip():
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."

    bpe = BPE()
    bpe.train(text, 256)
    assert bpe.decode(bpe.encode(text)) == text