use crate::tokenizer::{Tokenizer, TrainError};
use fancy_regex::Regex;
use std::collections::HashMap;

const PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

pub struct RegexTokenizer {
    merges: HashMap<(u32, u32), u32>, // (int,int): int
    vocab: HashMap<u32, Vec<u8>>,     // int:bytes
    pattern: String,                  // TODO: change,
    special_tokens: HashMap<String, u32>,
}

impl RegexTokenizer {
    pub fn new() -> Self {
        RegexTokenizer {
            merges: HashMap::<(u32, u32), u32>::new(),
            vocab: HashMap::<u32, Vec<u8>>::new(),
            pattern: PATTERN.to_string(),
            special_tokens: HashMap::<String, u32>::new(),
        }
    }

    pub fn new_with_special_tokens(special_tokens: HashMap<String, u32>) -> Self {
        RegexTokenizer {
            merges: HashMap::<(u32, u32), u32>::new(),
            vocab: HashMap::<u32, Vec<u8>>::new(),
            pattern: PATTERN.to_string(),
            special_tokens: special_tokens,
        }
    }
}

impl Tokenizer for RegexTokenizer {
    fn train(&self, text: &str, vocab_size: &u32) -> Result<(), TrainError> {
        todo!()
    }

    fn encode_with(
        &self,
        text: &str,
        allowed_special: crate::tokenizer::AllowedSpecial,
    ) -> Result<Vec<u32>, crate::tokenizer::EncodeError> {
        todo!()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, crate::tokenizer::EncodeError> {
        todo!()
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, crate::tokenizer::DecodeError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::{AllowedSpecial, DecodeError, EncodeError};

    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_roundtrip() {
        let text = "ｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. 안녕하세요 휴먼 저는 한국인입니다. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.";
        let tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_roundtrip_korean() {
        let text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다.";
        let tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_roundtrip_emoji() {
        let text = "😀😀🍕🍺🍕🍺🍕🍺😂😂😂🐶🐱🐶🐱❤️❤️🍕🍺🎉🎉🎉😀🐶🐱🍕🍺❤️😂😂😀😀🍕🍺🍕🍺🍕🍺😂😂😂🐶🐱🐶🐱❤️❤️🍕🍺🎉🎉🎉😀🐶🐱🍕🍺❤️😂😂👍🏽👍🏽👨‍👩‍👧👨‍👩‍👧🇰🇷🇰🇷🇰🇷👍🏽🍕🍕👨‍👩‍👧🇰🇷";
        let tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_roundtrip_space() {
        let text = "                                                                                                                                                       ";
        let tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_roundtrip_empty() {
        let text = "";
        let tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_vocab_size() {
        let vocab_size = 300;
        let text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다.";
        let tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &vocab_size).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        );

        assert_eq!(vocab_size as usize, tokenizer.vocab.len())
    }

    #[test]
    fn test_len_merge_plus_256_equals_to_vocab_size() {
        let vocab_size = 300;
        let text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다.";
        let tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &vocab_size).unwrap();

        assert_eq!(vocab_size as usize, tokenizer.merges.len() + 256);
    }

    #[test]
    fn test_small_text_with_fixed_merges() {
        let text = "aaabdaaabac";
        let tokenizer: RegexTokenizer = RegexTokenizer::new();
        let merges: HashMap<(u32, u32), u32> = HashMap::from([
            ((97, 97), 256),
            ((97, 98), 257),
            ((256, 257), 258),
            ((97, 99), 259),
            ((100, 258), 260),
            ((258, 260), 261),
            ((261, 259), 262),
        ]);

        tokenizer.train(text, &300).unwrap();
        assert_eq!(merges, tokenizer.merges)
    }

    #[test]
    fn test_roundtrip_of_untrained_data() {
        let train_text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. 안녕하세요 휴먼 저는 한국인입니다. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception. 대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다.";
        let untrained_text = "학습 텍스트를 다시 encode한 결과가 train 종료 시점의 ids와 동일한지";

        let vocab_size = 1256;
        let tokenizer: RegexTokenizer = RegexTokenizer::new();
        tokenizer.train(train_text, &vocab_size).unwrap();

        assert_eq!(
            untrained_text,
            tokenizer
                .decode(&tokenizer.encode(untrained_text).unwrap())
                .unwrap()
        );
    }

    #[test]
    fn test_special_token_collision_throws() {
        let tokenizer: RegexTokenizer = RegexTokenizer::new_with_special_tokens(HashMap::from([(
            "<|endoftext|>".to_string(),
            256,
        )]));

        assert!(matches!(
            tokenizer.train("aaaa bbbb aaaa", &300),
            Err(TrainError::SpecialTokenCollision {
                token,
                id: 256
            }) if token == "<|endoftext|>",
        ));
    }

    #[test]
    fn test_special_token_at_vocab_end() {
        let vocab_size = 300;
        let tokenizer: RegexTokenizer = RegexTokenizer::new_with_special_tokens(HashMap::from([(
            "<|endoftext|>".to_string(),
            vocab_size,
        )]));
        tokenizer.train("aaaa bbbb aaaa", &vocab_size).unwrap();

        assert_eq!(tokenizer.decode(&[vocab_size]).unwrap(), "<|endoftext|>")
    }

    #[test]
    fn test_tie_break_is_order_independent() {
        let t1 = RegexTokenizer::new();
        let t2 = RegexTokenizer::new();

        t1.train("ab cd ab cd", &258).unwrap();
        t1.train("cd ab cd ab", &258).unwrap();

        assert_eq!(t1.merges, t2.merges);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn test_roundtrip_random(text in any::<String>()) {
            let tokenizer = RegexTokenizer::new();

            tokenizer.train("The quick brown fox 안녕하세요 🐶 123\n\t", &300).unwrap();

            prop_assert_eq!(
                tokenizer.decode(&tokenizer.encode(&text).unwrap()).unwrap(),
                text
            );
        }
    }

    #[test]
    fn test_merge_never_crosses_chunk_boundary() {
        let tokenizer = RegexTokenizer::new();
        tokenizer.train("dog. dog. dog. dog. dog.", &300).unwrap();
        assert!(!tokenizer.vocab.values().any(|v| v == b"g."));
    }

    #[test]
    fn test_encode_is_chunkwise() {
        let tokenizer = RegexTokenizer::new();
        tokenizer.train("dog. dog. dog. dog. dog.", &300).unwrap();
        let text = "dog. dog.";

        let re = Regex::new(PATTERN).unwrap();

        let chunk_encoded: Vec<u32> = re
            .find_iter(text)
            .flat_map(|m| tokenizer.encode(m.unwrap().as_str()).unwrap())
            .collect();

        assert_eq!(tokenizer.encode(text).unwrap(), chunk_encoded);
    }

    #[test]
    fn test_decode_truncated_utf8_does_not_crash() {
        let tokenizer = RegexTokenizer::new();
        tokenizer.train("한국", &258).unwrap();
        assert_eq!(tokenizer.decode(&[236]).unwrap(), "\u{fffd}")
    }

    #[test]
    fn test_decode_invalid_id_err() {
        let tokenizer = RegexTokenizer::new();
        tokenizer.train("한국", &258).unwrap();

        assert!(matches!(
            tokenizer.decode(&[99999]),
            Err(DecodeError::InvalidTokenId(_))
        ));
    }

    #[test]
    fn test_encode_allowed_special_is_all() {
        let tokenizer = RegexTokenizer::new();
        tokenizer.train("hello world hello", &300).unwrap();
        let text = "hello<|endoftext|>world";

        let ids = tokenizer.encode_with(text, AllowedSpecial::All).unwrap();
        assert!(ids.iter().any(|v| v == &300));
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
    }

    #[test]
    fn test_encode_allowed_special_is_none() {
        let tokenizer = RegexTokenizer::new();
        tokenizer.train("hello world hello", &300).unwrap();
        let text = "hello<|endoftext|>world";

        let ids = tokenizer.encode_with(text, AllowedSpecial::None).unwrap();
        assert!(!ids.iter().any(|v| v == &300));
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
    }

    #[test]
    fn test_encode_allowed_special_is_default_none_raise() {
        let tokenizer = RegexTokenizer::new();
        tokenizer.train("hello world hello", &300).unwrap();
        let text = "hello<|endoftext|>world";

        assert!(matches!(
            tokenizer.encode(text),
            Err(EncodeError::AllowedSpecialNoneRaise)
        ));
    }

    #[test]
    fn test_encode_special_token_at_edges() {
        let tokenizer = RegexTokenizer::new_with_special_tokens(HashMap::from([(
            "<|endoftext|>".to_string(),
            300,
        )]));
        tokenizer.train("hello world hello", &300).unwrap();
        let text = "<|endoftext|>hello<|endoftext|><|endoftext|>";
        let ids = tokenizer.encode_with(text, AllowedSpecial::All).unwrap();
        assert_eq!(ids[0], 300);
        assert_eq!(ids[ids.len() - 2..], [300, 300]);
    }
}
