from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import regex as re
from tqdm import tqdm
from cs336_basics.pretokenization_helper import find_chunk_boundaries

# Pre-compile patterns once at module level to avoid recompiling in each worker
PAT_REGEX = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", re.UNICODE)


CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB


def pretokenize_chunk(args: tuple[str, int, int, list[str], re.Pattern[str]]):
    file_path, start, end, special_tokens, split_pattern = args
    counts = defaultdict(int)

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start).decode("utf-8", errors="replace")

    # Split on special tokens at most once per occurrence
    parts = split_pattern.split(chunk_data)
    for part in parts:
        # Skip if exactly a special token
        if part in special_tokens:
            continue
        # Find all matches in part
        for match in PAT_REGEX.finditer(part):
            token = match.group(0)
            # Encode to UTF-8 bytes and split into single-byte elements
            b = token.encode("utf-8")
            word_bytes = tuple(bytes([x]) for x in b)
            counts[word_bytes] += 1

    return counts


def parallel_pretokenize(
    input_path: str, special_tokens: list[str], split_special_token: str = "<|endoftext|>"
) -> dict[tuple[bytes, ...], int]:
    """
    Read the file in parallel, pretokenize into word-byte tuples, and return global counts.
    """

    split_pattern = re.compile("|".join(re.escape(st) for st in special_tokens))
    # Determine chunk boundaries that don't split special tokens
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, cpu_count(), split_special_token=split_special_token[0].encode("utf-8"))

    tasks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        # Subdivide large boundary into fixed-size chunks
        pos = start
        while pos < end:
            next_end = min(pos + CHUNK_SIZE, end)
            tasks.append((input_path, pos, next_end, special_tokens, split_pattern))
            pos = next_end

    final_counts = defaultdict(int)
    # Use a process pool; workers share module-level compiled patterns
    with Pool(cpu_count()) as pool:
        for partial in tqdm(pool.imap_unordered(pretokenize_chunk, tasks), total=len(tasks), desc="Pretokenizing"):  # type: ignore
            for word, cnt in partial.items():
                final_counts[word] += cnt

    return final_counts


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], split_special_token: str = "<|endoftext|>"):
    """
    Train BPE vocabulary of size `vocab_size` (including initial bytes and special tokens).
    """
    # Initialize vocabulary with all single bytes
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # Get token counts (word as tuple of single-byte tokens)
    token_counts = parallel_pretokenize(input_path, special_tokens, split_special_token)

    # Initialize pair counts and locations
    pair_counts = Counter()
    pair_locs = defaultdict(set)
    for word, cnt in token_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += cnt
            pair_locs[pair].add(word)

    merges: list[str] = []
    # BPE merge loop
    for _ in tqdm(range(num_merges), desc="BPE merges"):  # type: ignore
        if not pair_counts:
            break
        # Find most frequent pair (break ties by lex order)
        most_pair, freq = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        if freq <= 0:
            break
        merges.append(most_pair)
        # Add new merged token to vocab
        new_token_id = len(vocab)
        new_bytes = most_pair[0] + most_pair[1]
        vocab[new_token_id] = new_bytes

        # Tokens affected by this merge
        affected = pair_locs.pop(most_pair, set())
        for old_word in list(affected):
            if old_word not in token_counts:
                continue
            cnt = token_counts.pop(old_word)
            # Remove old pairs
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i + 1])
                pair_counts[p] -= cnt
                pair_locs[p].discard(old_word)
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                    del pair_locs[p]
            # Build new word by merging occurrences
            new_word_tokens = []
            i = 0
            L = len(old_word)
            while i < L:
                if i < L - 1 and (old_word[i], old_word[i + 1]) == most_pair:
                    new_word_tokens.append(old_word[i] + old_word[i + 1])
                    i += 2
                else:
                    new_word_tokens.append(old_word[i])
                    i += 1
            new_word = tuple(new_word_tokens)
            token_counts[new_word] = cnt
            # Add new pairs from new_word
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j + 1])
                pair_counts[p] += cnt
                pair_locs[p].add(new_word)

    return vocab, merges
