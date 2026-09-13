import itertools

import regex as re


class Tokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int,int],int] = {}
        self.pattern = ""
        self.special_tokens:dict[str,int] = {}
        self.vocab: dict[int, bytes] = {}

    def train(self, text: str, vocab_size: int, verbose=False):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, file_name):
        raise NotImplementedError

    def load(self, model_file):
        raise NotImplementedError

REG_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
BYTE_SIZE = 256



class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=REG_PATTERN, special_tokens=None) -> None:
        super().__init__()
        self.pattern: re.Pattern[str] = re.compile(pattern)
        self.special_tokens: dict[str, int] = special_tokens if special_tokens else {}

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= BYTE_SIZE
        num_merges = vocab_size + BYTE_SIZE

        chunks: list[str] = re.findall(self.pattern, text)
        ids = [list(ch.encode('utf-8')) for ch in chunks]

        merges: dict[tuple[int,int], int] = {}
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(BYTE_SIZE)}

        for i in range(num_merges):
            # count in every merge iteration
            count: dict[tuple[int, int], int] = {}

            for chunk_ids in ids:
                for pair in itertools.pairwise(chunk_ids):
                    count[pair] = count.get(pair, 0) + 1

            if not count:
                continue

            pair = max(count, key=lambda k: count[k])
            idx = i + BYTE_SIZE

            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {count[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

 
    def _merge(self, ids: list[int], pair: tuple[int,int], new_id: int) -> list[int]:
        new_ids = []

        i= 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids


    def encode(self, text)-> list[int]:
        """Ignores any special tokens"""
        text_chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids


    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            count: dict[tuple[int,int], int] = {}
            for pair in itertools.pairwise(ids):
                count[pair] = count.get(pair, 0)+1

            # if merges does not have pair, set to maximum float number;
            # send it to last
            pair = min(count, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            ids = self._merge(ids, pair, self.merges[pair])

        return ids

            

    def decode(self, ids: list[int]) -> str:
        part_bytes = []
        inversed_special_tokens = {v:k for k,v in self.special_tokens.items()}
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in inversed_special_tokens:
                part_bytes.append(inversed_special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f"invalid token id: {idx}")

        text_bytes = b"".join(part_bytes)
        return text_bytes.decode('utf-8', errors="replace")


            



