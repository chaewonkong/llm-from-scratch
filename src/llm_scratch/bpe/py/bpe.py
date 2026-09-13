import itertools


class BPE:
    merges: dict[tuple[int,int], int] = {}

    def most_common_pairs(self, ids: list[int]) -> tuple[int, int] | None:
        left, right = -1, -1 # init with garbage
        maximum = 1
        count_map: dict[tuple[int, int], int] = {}
        for candidate in itertools.pairwise(ids):
            count = count_map.get(candidate, 0) + 1
            count_map[candidate] = count
            if count > maximum:
                maximum = count
                left, right = candidate

        if left == -1 and right == -1:
            return None
        
        return (left, right)

    def merge(self, ids: list[int], pair: tuple[int,int], new_id: int) -> list[int]:
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

    def get_tokens(self, s: str) -> list[int]:
        return list( s.encode('utf-8'))

    def train(self, text: str, vocab_size: int) -> list[int]:
        num_merges = vocab_size-256
        tokens = self.get_tokens(text)
        ids = list(tokens)
        for i in range(num_merges):
            idx = 256+i
            pair = self.most_common_pairs(ids)
            if pair is None:
                break
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
        
        return ids

    def encode(self, text: str) ->  list[int]:
        tokens = self.get_tokens(text)
        while len(tokens) > 1:
            pair = self.most_common_pairs(tokens)
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens


    def decode(self, ids: list[int]) -> str:
        vocab = {idx: bytes([idx]) for idx in range(256)} # populate with 256 codes
        for (left, right), idx in self.merges.items(): # items in merges order preserved; insert order == iteration order
            vocab[idx] = vocab[left] + vocab[right]

        tokens = b"".join(vocab[idx] for idx in ids)
        
        return tokens.decode('utf-8')
