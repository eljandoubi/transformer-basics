import json
import regex as re
from collections.abc import Iterable, Iterator
from functools import lru_cache
import heapq
import numpy as np
from cs336_basics.train_bpe_tokenizer import train_bpe


# Parsing pattern used in GPT2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


class Segmenter:
    """This class scans the raw input string, finds occurrences of any of the “special tokens”
    (by working in raw UTF-8 bytes), and returns a sequence of segments"""

    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens or []

        # Build a simple byte‐trie for special tokens
        # (each key is a UTF‐8 string; store bytes-list for matching)
        self._special_trie = {}
        for tok in self.special_tokens:
            node = self._special_trie
            tok_bytes = tok.encode("utf-8")
            for b in tok_bytes:
                node = node.setdefault(b, {})
            node["_end"] = tok  # mark end‐of‐special‐token at this node

    def __call__(self, text: str) -> list[tuple[bool, str]]:
        """
        Get the list of segments (bool, str) from the text.
        """
        data = text.encode("utf-8")  # work in bytes for exact matches

        i = 0
        nbytes = len(data)
        result: list[tuple[bool, str]] = []
        buffer_bytes = bytearray()

        while i < nbytes:
            node = self._special_trie
            j = i
            last_match = None
            last_pos = i
            # walk as far as possible in the trie
            while j < nbytes and data[j] in node:
                node = node[data[j]]
                j += 1
                if "_end" in node:
                    last_match = node["_end"]
                    last_pos = j

            if last_match is not None:
                # flush any buffered “normal” bytes so far
                if buffer_bytes:
                    result.append((False, buffer_bytes.decode("utf-8", "replace")))
                    buffer_bytes = bytearray()
                # append the special token
                result.append((True, last_match))
                i = last_pos
            else:
                # no special token starts here
                buffer_bytes.append(data[i])
                i += 1

        if buffer_bytes:
            result.append((False, buffer_bytes.decode("utf-8", "replace")))
        return result


class Merger:
    """Class that keeps a list of byte‐pair merges and a rank‐lookup dict."""

    def __init__(self, merges: list[tuple[bytes, bytes]] | None = None):
        # Store merges in a plain list
        self.merges: list[tuple[bytes, bytes]] = []
        # The rank‐lookup dictionary
        self.merges_dict: dict[bytes, dict[bytes, int]] = {}

        # If an initial list was given, add each through our helper
        if merges:
            for a, b in merges:
                self.add_merge(a, b)

    def add_merge(self, a: bytes, b: bytes) -> None:
        """
        Append the pair (a,b) to self.merges and register it in self.merges_dict.
        The “rank” is simply the index in self.merges.
        """
        self.merges.append((a, b))
        rank = len(self.merges) - 1
        self.merges_dict.setdefault(a, {})[b] = rank

    def __call__(self, word_bytes: list[bytes]) -> list[bytes]:
        """
        Given a list of byte‐tokens, repeatedly merge the pair (a, b) with the lowest rank
        (according to self.merges_dict) until no more mergeable adjacent pairs remain.

        Uses a min‐heap to track only the candidate pairs, leading to roughly O(n log n)
        """
        n = len(word_bytes)
        if n < 2:
            return word_bytes[:]  # nothing to merge

        # Build helper arrays for a doubly‐linked structure over indices 0..n-1
        prev_idx = list(range(-1, n - 1))  # prev_idx[i] = i-1, except prev_idx[0] = -1
        next_idx = list(range(1, n + 1))  # next_idx[i] = i+1, except next_idx[n-1] = n (sentinel)
        alive = [True] * n  # alive[i] = whether token i is still “in play”

        # A small helper to push (rank, left_index) if (i, j) is a valid merge pair
        def push_pair(i: int, j: int):
            """If (word_bytes[i], word_bytes[j]) is in merges_dict, push (rank, i) into heap."""
            key_a = word_bytes[i]
            key_b = word_bytes[j]
            # merges_dict maps:   merges_dict[a][b] = rank (an integer)
            # if a not in merges_dict or b not a valid neighbor, skip
            rank = self.merges_dict.get(key_a, {}).get(key_b)
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        # Build the initial heap of all adjacent (i, i+1) pairs that exist in merges_dict
        heap: list[tuple[int, int]] = []
        for i in range(n - 1):
            push_pair(i, i + 1)

        # This counter keeps track of how many “tokens” are still alive
        # so we know when we’re down to a single dummy sentinel or no merges left.
        alive_count = n

        while heap:
            rank, i = heapq.heappop(heap)

            # If i is no longer alive, or its “next” neighbor is gone, skip
            if not alive[i]:
                continue
            j = next_idx[i]
            if j >= n or not alive[j]:
                # either j is the sentinel (n) or j has been merged away already
                continue

            # Double‐check that they still form a mergeable pair of the same rank:
            a, b = word_bytes[i], word_bytes[j]
            current_rank = self.merges_dict.get(a, {}).get(b)
            if current_rank is None or current_rank != rank:
                # either they’re no longer mergeable or their rank changed
                continue

            # ↓—— Perform the merge of indices i and j ———————————————————↓

            # 1) Concatenate b onto a, store in position i
            word_bytes[i] = a + b

            # 2) “Kill” index j by marking alive[j]=False
            alive[j] = False
            alive_count -= 1

            # 3) Splice j out of the linked structure:
            nxt = next_idx[j]
            next_idx[i] = nxt
            if nxt < n:
                prev_idx[nxt] = i

            # 4) Now i’s new neighbors might form new mergeable pairs. Push them:

            # 4a) Check if i has a “prev” neighbor
            p = prev_idx[i]
            if p >= 0 and alive[p]:
                push_pair(p, i)

            # 4b) Check if i has a “next” neighbor
            q = next_idx[i]
            if q < n and alive[q]:
                push_pair(i, q)

            # If we dropped below 2 alive tokens, there can’t be more merges
            if alive_count < 2:
                break

        # ———— Extract the surviving tokens in order ——————
        result: list[bytes] = []
        idx = 0
        # Find the first “alive” index
        while idx < n and not alive[idx]:
            idx += 1
        # Walk forward via next_idx and collect bytes
        while idx < n:
            if alive[idx]:
                result.append(word_bytes[idx])
            idx = next_idx[idx]

        return result


class BPETokenizer:
    """A BPE tokenizer that uses the provided vocab, merges, and special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merger = Merger(merges=merges)
        self.segmenter = Segmenter(special_tokens=special_tokens)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pretoken_to_id = {}
        self.special_bytes = {}
        for tok in self.segmenter.special_tokens:
            tok_bytes = tok.encode("utf-8")
            self.special_bytes[tok] = self.reverse_vocab[tok_bytes]

        self._PAT_RE = re.compile(PAT)

    def encode(self, text: str) -> list[int]:
        segments = self.segmenter(text)
        append = []
        pretoken_to_id = self.pretoken_to_id
        special_bytes = self.special_bytes
        reverse_vocab = self.reverse_vocab
        pattern = self._PAT_RE.finditer
        apply_merges = self.merger

        for is_special, token in segments:
            if is_special:
                append.append(special_bytes[token])
                continue

            for match in pattern(token):
                word = match.group(0)

                cached = pretoken_to_id.get(word)
                if cached is not None:
                    append.extend(cached)
                    continue

                # Encode UTF-8 and wrap bytes as needed
                word_bytes = [bytes([b]) for b in word.encode("utf-8")]
                merged = apply_merges(word_bytes)
                ids = [reverse_vocab[b] for b in merged]

                pretoken_to_id[word] = ids
                append.extend(ids)

        return append

    def encode_file(self, file_name: str) -> list[int]:
        with open(file_name) as f:
            text = f.read()
        return self.encode(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def stream_encode(self, file_name: str):
        with open(file_name) as f:
            yield from self.encode_iterable(f)

    def stream_sequence(self, file_name: str):
        with open(file_name) as f:
            while chunk := f.readline():
                yield np.array(self.encode(chunk))

    def decode(self, ids: list[int]) -> str:
        output_bytes = b"".join(map(self.vocab.get, ids))
        return output_bytes.decode("utf-8", errors="replace")

    @classmethod
    def from_vocab(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens_path: str,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

        with open(special_tokens_path) as f:
            special_tokens = f.readlines()

        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_token)

    @classmethod
    def from_training(cls, input_path: str, vocab_size: int, special_tokens: list[str]):
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def save(
        self,
        vocab_path: str,
        merges_path: str,
        special_tokens_path: str,
    ):
        gpt2_byte_encoder = gpt2_bytes_to_unicode()
        str_vocab = {"".join([gpt2_byte_encoder[j] for j in v]): k for k, v in self.vocab.items()}

        with open(vocab_path, "w") as vocab_f:
            json.dump(
                str_vocab,
                vocab_f,
            )

        with open(merges_path, "w") as f:
            for merge in self.merger.merges:
                f.write("".join([gpt2_byte_encoder[j] for j in merge[0]]) + " ")
                f.write("".join([gpt2_byte_encoder[j] for j in merge[1]]) + "\n")

        with open(special_tokens_path, "w") as sp:
            sp.writelines(self.segmenter.special_tokens)

    def __eq__(self, value: "BPETokenizer") -> bool:
        if self.vocab != value.vocab:
            return False
        if self.merger.merges != value.merger.merges:
            return False
        if self.segmenter.special_tokens != value.segmenter.special_tokens:
            return False
        return True


if __name__ == "__main__":
    bpe = BPETokenizer.from_training("./data/TinyStoriesV2-GPT4-train.txt", 512, ["<|endoftext|>"])
    bpe.save(
        "./tokenizer/gpt2_vocab.json",
        "./tokenizer/gpt2_merges.txt",
        "./tokenizer/special.txt",
    )
