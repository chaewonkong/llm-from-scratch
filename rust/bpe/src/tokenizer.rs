#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllowedSpecial {
    All,
    None,
    #[default]
    NoneRaise,
}

#[derive(Debug)]
pub enum EncodeError {
    AllowedSpecialNoneRaise,
}

#[derive(Debug)]

pub enum DecodeError {
    InvalidTokenId(u32),
}

#[derive(Debug)]
pub enum TrainError {
    SpecialTokenCollision { token: String, id: u32 },
}
pub trait Tokenizer {
    fn train(&self, text: &str, vocab_size: &u32) -> Result<(), TrainError>;
    fn encode_with(
        &self,
        text: &str,
        allowed_special: AllowedSpecial,
    ) -> Result<Vec<u32>, EncodeError>;
    fn encode(&self, text: &str) -> Result<Vec<u32>, EncodeError>;
    fn decode(&self, tokens: &[u32]) -> Result<String, DecodeError>;
}
