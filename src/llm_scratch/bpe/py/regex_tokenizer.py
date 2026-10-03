import itertools

import regex as re

from llm_scratch.bpe.py.tokenizer import Tokenizer

REG_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
BYTE_SIZE = 256


def word_counter(chunks: list[str]) -> list[tuple[list[int], int]]:
    count = {}
    for chunk in chunks:
        count[chunk] = count.get(chunk, 0) + 1

    return [(list(w.encode("utf-8")), f) for w, f in count.items()]


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=REG_PATTERN, special_tokens=None) -> None:
        super().__init__()
        self.pattern: re.Pattern[str] = re.compile(pattern)
        self.special_tokens: dict[str, int] = special_tokens if special_tokens else {}

    def train(self, text: str, vocab_size: int, verbose=False) -> None:
        assert vocab_size >= BYTE_SIZE
        num_merges = vocab_size - BYTE_SIZE

        words = word_counter(re.findall(self.pattern, text))

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(BYTE_SIZE)}

        count: dict[tuple[int, int], int] = {}
        pair_word_idx_map: dict[
            tuple[int, int], set[int]
        ] = {}  # {pair: set(word_idx..)}
        for i in range(len(words)):
            word, frequency = words[i]
            for pair in itertools.pairwise(word):
                count[pair] = count.get(pair, 0) + frequency
                pair_word_idx_map.setdefault(pair, set()).add(i)

        for i in range(num_merges):
            if not count:  # no available pair
                break

            pair = min(count, key=lambda k: (-count[k], k))
            pair_count = count[pair]
            idx = i + BYTE_SIZE

            for word_idx in pair_word_idx_map.pop(pair):
                word, freq = words[word_idx]
                new_word = self._merge(word, pair, idx)
                words[word_idx] = new_word, freq

                for p in itertools.pairwise(word):  # reset
                    count[p] -= freq
                    if count[p] == 0:  # cleanup
                        del count[p]
                        pair_word_idx_map.pop(p, None)

                for p in itertools.pairwise(new_word):  # assign
                    count[p] = count.get(p, 0) + freq
                    pair_word_idx_map.setdefault(p, set()).add(word_idx)

                gone = set(itertools.pairwise(word)) - set(itertools.pairwise(new_word))
                for p in gone:
                    if p in pair_word_idx_map:
                        pair_word_idx_map[p].discard(word_idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {pair_count} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

        collided = set(self.special_tokens.values()) & set(vocab)
        assert not collided, f"special token id collides with vocab id: {collided}"

    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        new_ids = []  # TODO: replace

        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def encode_ordinary(self, text) -> list[int]:
        """Ignores any special tokens"""
        chunks: list[str] = re.findall(self.pattern, text)
        unique_chunks = set(chunks)

        word_map: dict[str, list[int]] = {}
        for word in unique_chunks:
            c_ids = self._encode_chunk(list(word.encode("utf-8")))
            word_map[word] = c_ids

        ids: list[int] = []
        for chunk in chunks:
            ids.extend(word_map[chunk])
        return ids

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        # TODO: use counter
        while len(ids) >= 2:
            # if merges does not have pair, set to maximum float number;
            # send it to last
            pair = min(
                itertools.pairwise(ids), key=lambda p: self.merges.get(p, float("inf"))
            )

            if pair not in self.merges:
                break

            ids = self._merge(ids, pair, self.merges[pair])

        return ids

    def encode(self, text: str, allowed_special="none_raise") -> list[int]:
        """allowed_special = "all" | "none" | "none_raise" """

        special: dict[str, int] = {}
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            pass
        elif allowed_special == "none_raise":
            assert all(token not in text for token in self.special_tokens)
        else:
            raise ValueError(f"allowed_special={allowed_special} not acceptable")

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        part_bytes = []
        inversed_special_tokens = {v: k for k, v in self.special_tokens.items()}
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in inversed_special_tokens:
                part_bytes.append(inversed_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")

        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")
