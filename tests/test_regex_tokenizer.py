import pytest

from llm_scratch.bpe.py.tokenizer import RegexTokenizer


def test_roundtrip():
    # given
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. 안녕하세요 휴먼 저는 한국인입니다. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=512)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))


    # then
    assert text == round_trip_result


def test_roundtrip_korean():
    # given
    text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다."
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=512)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))


    # then
    assert text == round_trip_result

def test_roundtrip_emoji():
    # given
    text = "😀😀🍕🍺🍕🍺🍕🍺😂😂😂🐶🐱🐶🐱❤️❤️🍕🍺🎉🎉🎉😀🐶🐱🍕🍺❤️😂😂😀😀🍕🍺🍕🍺🍕🍺😂😂😂🐶🐱🐶🐱❤️❤️🍕🍺🎉🎉🎉😀🐶🐱🍕🍺❤️😂😂👍🏽👍🏽👨‍👩‍👧👨‍👩‍👧🇰🇷🇰🇷🇰🇷👍🏽🍕🍕👨‍👩‍👧🇰🇷"
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=512)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))


    # then
    assert text == round_trip_result

def test_roundtrip_space():
    # given
    text = "                                                                                                                                                       "
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=512)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))


    # then
    assert text == round_trip_result

def test_roundtrip_empty():
    # given
    text =""
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=512)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))

    # then
    assert text == round_trip_result

def test_vocab_size():
    # given
    vocab_size = 300
    text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다."
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=vocab_size)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))

    # then
    assert text == round_trip_result
    assert vocab_size == len(tokenizer.vocab)

def test_merge_vocab_size():
    # given
    vocab_size = 300
    text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다."
    tokenizer = RegexTokenizer()

    tokenizer.train(text=text, vocab_size=vocab_size)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(text))

    # then
    assert text == round_trip_result
    assert vocab_size == len(tokenizer.merges) + 256

def test_small_fixed():
    # given
    text = "aaabdaaabac"
    tokenizer = RegexTokenizer()
    merges = {(97, 97): 256, (256, 97): 257, (257, 98): 258, (258, 100): 259, (259, 258): 260, (260, 97): 261, (261, 99): 262}

    # when
    tokenizer.train(text=text, vocab_size=300)
    print(tokenizer.merges)


    # then
    assert tokenizer.merges == merges

def test_roundtrip_untrained_data():
    # given
    train_text ="Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. 안녕하세요 휴먼 저는 한국인입니다. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception. 대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다."
    untrained_text = "학습 텍스트를 다시 encode한 결과가 train 종료 시점의 ids와 동일한지"
    vocab_size = 1256
    tokenizer = RegexTokenizer()

    tokenizer.train(train_text, vocab_size)

    # when
    round_trip_result = tokenizer.decode(tokenizer.encode(untrained_text))
    assert untrained_text == round_trip_result

def test_encod_matches_training_ids():
    # given
    text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다."
    tokenizer = RegexTokenizer()

    train_ids = tokenizer.train(text=text, vocab_size=512)

    # when
    encoded_ids = tokenizer.encode(text)

    # then
    assert train_ids == encoded_ids

def test_special_token_collision_raises():
    tokenizer = RegexTokenizer(special_tokens={"<|endoftext|>": 256})
    with pytest.raises(AssertionError):
        tokenizer.train(text="aaaa bbbb aaaa", vocab_size=300)

def test_special_token_at_vocab_end():
    vocab_size = 300
    tokenizer = RegexTokenizer(special_tokens={"<|endoftext|>": vocab_size})
    tokenizer.train(text="aaaa bbbb aaaa", vocab_size=vocab_size)

    assert tokenizer.decode([vocab_size]) == "<|endoftext|>"