from llm_scratch.bpe.py.tokenizer import RegexTokenizer


def test_roundtrip():
    # given
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. 안녕하세요 휴먼 저는 한국인입니다. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=100256)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))


    # then
    assert text == round_trip_result