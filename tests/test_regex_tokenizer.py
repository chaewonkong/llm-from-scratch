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


def test_roundtrip_korean():
    # given
    text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다."
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=100256)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))


    # then
    assert text == round_trip_result

def test_roundtrip_emoji():
    # given
    text = "😀😀🍕🍺🍕🍺🍕🍺😂😂😂🐶🐱🐶🐱❤️❤️🍕🍺🎉🎉🎉😀🐶🐱🍕🍺❤️😂😂😀😀🍕🍺🍕🍺🍕🍺😂😂😂🐶🐱🐶🐱❤️❤️🍕🍺🎉🎉🎉😀🐶🐱🍕🍺❤️😂😂👍🏽👍🏽👨‍👩‍👧👨‍👩‍👧🇰🇷🇰🇷🇰🇷👍🏽🍕🍕👨‍👩‍👧🇰🇷"
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=100256)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))


    # then
    assert text == round_trip_result

def test_roundtrip_empty():
    # given
    text = "                                                                                                                                                       "
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=100256)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))


    # then
    assert text == round_trip_result