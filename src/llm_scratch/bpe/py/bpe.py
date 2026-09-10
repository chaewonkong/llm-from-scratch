import itertools


def most_common_pairs(ids: list[int]) -> tuple[int, int] | None:
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




def merge(ids: list[int], pair: tuple[int,int], new_id: int) -> list[int]:
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





def get_tokens(s: str) -> list[int]:
    return list(map(int, s.encode('utf-8')))

def train(text: str, vocab_size: int) -> tuple[list[int], dict[tuple[int,int], int]]:
    merges = {}
    num_merges = vocab_size-256
    tokens = get_tokens(text)
    ids = list(tokens)
    for i in range(num_merges):
        idx = 256+i
        pair = most_common_pairs(ids)
        if pair is None:
            break
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    
    return ids, merges

def encode(text: str) ->  tuple[list[int], dict[tuple[int,int], int]]:
    return train(text, 256)

def decode(ids: list[int], merges: dict[tuple[int,int], int]) -> str:
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (left, right), idx in merges.items():
        vocab[idx] = vocab[left] + vocab[right]

    tokens = b"".join(vocab[idx] for idx in ids)
    
    return tokens.decode('utf-8')
