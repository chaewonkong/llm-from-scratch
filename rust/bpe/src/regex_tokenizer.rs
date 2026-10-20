use crate::tokenizer::{Tokenizer, TrainError};
use fancy_regex::Regex;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
};

const PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
const BYTE_SIZE: &u32 = &256;
pub struct RegexTokenizer {
    merges: HashMap<(u32, u32), u32>, // (int,int): int
    vocab: HashMap<u32, Vec<u8>>,     // int:bytes
    pattern: Regex,
    special_tokens: HashMap<String, u32>,
}

impl RegexTokenizer {
    pub fn new() -> Self {
        RegexTokenizer {
            merges: HashMap::<(u32, u32), u32>::new(),
            vocab: HashMap::<u32, Vec<u8>>::new(),
            pattern: Regex::new(PATTERN).unwrap(),
            special_tokens: HashMap::<String, u32>::new(),
        }
    }

    pub fn new_with_special_tokens(special_tokens: HashMap<String, u32>) -> Self {
        RegexTokenizer {
            merges: HashMap::<(u32, u32), u32>::new(),
            vocab: HashMap::<u32, Vec<u8>>::new(),
            pattern: Regex::new(PATTERN).unwrap(),
            special_tokens: special_tokens,
        }
    }

    fn merge(&self, ids: &[u32], pair: (u32, u32), new_id: &u32) -> Vec<u32> {
        let mut i = 0;
        let mut out = Vec::with_capacity(ids.len());
        while i < ids.len() {
            if i < ids.len() - 1 && ids[i] == pair.0 && ids[i + 1] == pair.1 {
                out.push(*new_id);
                i += 2;
            } else {
                out.push(ids[i]);
                i += 1;
            }
        }
        return out;
    }

    fn encode_chunk(&self, ids: &[u32]) -> Vec<u32> {
        todo!()
    }
}

impl Tokenizer for RegexTokenizer {
    fn train(&mut self, text: &str, vocab_size: &u32) -> Result<(), TrainError> {
        if vocab_size < BYTE_SIZE {
            return Err(TrainError::VocabSizeTooSmall);
        }
        let merge_limit = vocab_size - BYTE_SIZE;

        let mut counts: HashMap<Vec<u32>, usize> = HashMap::new();
        for m in self.pattern.find_iter(text) {
            let ids: Vec<u32> = m.unwrap().as_str().bytes().map(u32::from).collect();
            *counts.entry(ids).or_insert(0) += 1;
        }

        let mut words: Vec<(Vec<u32>, usize)> = Vec::with_capacity(counts.len());
        for (k, v) in counts.into_iter() {
            words.push((k, v));
        }

        let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
        let mut vocab: HashMap<u32, Vec<u8>> =
            (0..256u32).map(|i| (i as u32, vec![i as u8])).collect();
        let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
        let mut pair_word_idx_map: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();

        for i in 0..words.len() {
            let (word, freq) = &words[i];
            for w in word.windows(2) {
                let pair = (w[0], w[1]);
                *pair_counts.entry(pair).or_default() += freq;
                pair_word_idx_map.entry(pair).or_default().insert(i);
            }
        }

        let mut heap: BinaryHeap<(usize, Reverse<(u32, u32)>)> = BinaryHeap::new();
        for (p, c) in pair_counts.iter() {
            heap.push((*c, Reverse(*p)));
        }

        for i in 0..merge_limit {
            if pair_counts.is_empty() {
                break;
            }

            let (_, pair) = loop {
                let (count, Reverse(pair)) = heap.pop().unwrap();
                match pair_counts.get(&pair) {
                    Some(&c) if c == count => break (count, pair), // valid
                    Some(&c) if c < count => heap.push((c, Reverse(pair))),
                    _ => {}
                }
            };
            let idx = i + BYTE_SIZE;
            let mut touched: HashSet<(u32, u32)> = HashSet::new();
            while let Some(word_idxs) = pair_word_idx_map.remove(&pair) {
                for word_idx in word_idxs {
                    let freq = words[word_idx].1;
                    let new_word = self.merge(&words[word_idx].0, pair, &idx);
                    let word = std::mem::replace(&mut words[word_idx].0, new_word);
                    let new_word = &words[word_idx].0;

                    // reset
                    for w in word.windows(2) {
                        let pair = (w[0], w[1]);
                        let c = pair_counts.get_mut(&pair).expect("pair must be counted");
                        *c -= freq;
                        if *c == 0 {
                            // cleanup
                            pair_counts.remove(&pair);
                            pair_word_idx_map.remove(&pair);
                        }
                    }

                    //assign
                    for w in new_word.windows(2) {
                        let pair = (w[0], w[1]);
                        *pair_counts.entry(pair).or_default() += freq;
                        pair_word_idx_map.entry(pair).or_default().insert(word_idx);
                        if pair.0 == idx || pair.1 == idx {
                            touched.insert(pair);
                        }
                    }

                    // gone
                    let old_pairs: HashSet<(u32, u32)> =
                        word.windows(2).map(|w| (w[0], w[1])).collect();
                    let new_pairs: HashSet<(u32, u32)> =
                        new_word.windows(2).map(|w| (w[0], w[1])).collect();

                    for p in old_pairs.difference(&new_pairs) {
                        if let Some(idxs) = pair_word_idx_map.get_mut(p) {
                            idxs.remove(&word_idx);
                        }
                    }
                }
            }
            for p in touched {
                // TODO: 여기부터.
                if let Some(c) = pair_counts.get(&p) {
                    heap.push((*c, Reverse(p)));
                }
            }
            merges.insert(pair, idx);
            let merged = [vocab[&pair.0].as_slice(), vocab[&pair.1].as_slice()].concat();
            vocab.insert(idx, merged);
        }

        self.merges = merges;
        self.vocab = vocab;

        Ok(())
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
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_roundtrip_korean() {
        let text = "대규모 언어 모델(LLM)은 방대한 양의 데이터를 학습하여 자연어 및 기타 유형의 콘텐츠를 이해하고 생성하여 광범위한 작업을 수행할 수 있는 딥 러닝의 카테고리입니다. LLM은 단어 시퀀스를 처리하고 텍스트의 패턴을 포착하는 데 탁월한 신경망 아키텍처의 일종(트랜스포머라고 함)을 기반으로 구축됩니다.";
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_roundtrip_emoji() {
        let text = "😀😀🍕🍺🍕🍺🍕🍺😂😂😂🐶🐱🐶🐱❤️❤️🍕🍺🎉🎉🎉😀🐶🐱🍕🍺❤️😂😂😀😀🍕🍺🍕🍺🍕🍺😂😂😂🐶🐱🐶🐱❤️❤️🍕🍺🎉🎉🎉😀🐶🐱🍕🍺❤️😂😂👍🏽👍🏽👨‍👩‍👧👨‍👩‍👧🇰🇷🇰🇷🇰🇷👍🏽🍕🍕👨‍👩‍👧🇰🇷";
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_roundtrip_space() {
        let text = "                                                                                                                                                       ";
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &512).unwrap();
        assert_eq!(
            text,
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap()
        )
    }

    #[test]
    fn test_roundtrip_empty() {
        let text = "";
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();

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
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();

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
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();

        tokenizer.train(text, &vocab_size).unwrap();

        assert_eq!(vocab_size as usize, tokenizer.merges.len() + 256);
    }

    #[test]
    fn test_small_text_with_fixed_merges() {
        let text = "aaabdaaabac";
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();
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
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new();
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
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new_with_special_tokens(HashMap::from(
            [("<|endoftext|>".to_string(), 256)],
        ));

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
        let mut tokenizer: RegexTokenizer = RegexTokenizer::new_with_special_tokens(HashMap::from(
            [("<|endoftext|>".to_string(), vocab_size)],
        ));
        tokenizer.train("aaaa bbbb aaaa", &vocab_size).unwrap();

        assert_eq!(tokenizer.decode(&[vocab_size]).unwrap(), "<|endoftext|>")
    }

    #[test]
    fn test_tie_break_is_order_independent() {
        let mut t1 = RegexTokenizer::new();
        let t2: RegexTokenizer = RegexTokenizer::new();

        t1.train("ab cd ab cd", &258).unwrap();
        t1.train("cd ab cd ab", &258).unwrap();

        assert_eq!(t1.merges, t2.merges);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn test_roundtrip_random(text in any::<String>()) {
            let mut tokenizer = RegexTokenizer::new();

            tokenizer.train("The quick brown fox 안녕하세요 🐶 123\n\t", &300).unwrap();

            prop_assert_eq!(
                tokenizer.decode(&tokenizer.encode(&text).unwrap()).unwrap(),
                text
            );
        }
    }

    #[test]
    fn test_merge_never_crosses_chunk_boundary() {
        let mut tokenizer = RegexTokenizer::new();
        tokenizer.train("dog. dog. dog. dog. dog.", &300).unwrap();
        assert!(!tokenizer.vocab.values().any(|v| v == b"g."));
    }

    #[test]
    fn test_encode_is_chunkwise() {
        let mut tokenizer = RegexTokenizer::new();
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
        let mut tokenizer = RegexTokenizer::new();
        tokenizer.train("한국", &258).unwrap();
        assert_eq!(tokenizer.decode(&[236]).unwrap(), "\u{fffd}")
    }

    #[test]
    fn test_decode_invalid_id_err() {
        let mut tokenizer = RegexTokenizer::new();
        tokenizer.train("한국", &258).unwrap();

        assert!(matches!(
            tokenizer.decode(&[99999]),
            Err(DecodeError::InvalidTokenId(_))
        ));
    }

    #[test]
    fn test_encode_allowed_special_is_all() {
        let mut tokenizer = RegexTokenizer::new();
        tokenizer.train("hello world hello", &300).unwrap();
        let text = "hello<|endoftext|>world";

        let ids = tokenizer.encode_with(text, AllowedSpecial::All).unwrap();
        assert!(ids.iter().any(|v| v == &300));
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
    }

    #[test]
    fn test_encode_allowed_special_is_none() {
        let mut tokenizer = RegexTokenizer::new();
        tokenizer.train("hello world hello", &300).unwrap();
        let text = "hello<|endoftext|>world";

        let ids = tokenizer.encode_with(text, AllowedSpecial::None).unwrap();
        assert!(!ids.iter().any(|v| v == &300));
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
    }

    #[test]
    fn test_encode_allowed_special_is_default_none_raise() {
        let mut tokenizer = RegexTokenizer::new();
        tokenizer.train("hello world hello", &300).unwrap();
        let text = "hello<|endoftext|>world";

        assert!(matches!(
            tokenizer.encode(text),
            Err(EncodeError::AllowedSpecialNoneRaise)
        ));
    }

    #[test]
    fn test_encode_special_token_at_edges() {
        let mut tokenizer = RegexTokenizer::new_with_special_tokens(HashMap::from([(
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
