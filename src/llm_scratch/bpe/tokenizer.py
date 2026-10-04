import json


class Tokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.pattern = ""
        self.special_tokens: dict[str, int] = {}
        self.vocab: dict[int, bytes] = {}

    def train(self, text: str, vocab_size: int, verbose=False):
        raise NotImplementedError

    def encode(self, text: str, allowed_special="none_raise") -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, file_name):
        with open(f"merges/{file_name}.json", mode="w") as f:
            json.dump([list(p) for p in self.merges], f)

    def load(self, model_file):
        with open(f"merges/{model_file}.json", mode="w") as f:
            pairs: list[list[int]] = json.load(f)
            self.merges = {(a, b): 256 + i for i, (a, b) in enumerate(pairs)}
