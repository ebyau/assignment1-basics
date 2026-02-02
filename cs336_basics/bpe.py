# imports
import regex as re
import os
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from collections import Counter

# regex based pre-tokenization pattern (Used by GPT-2)
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def worker_count_chunk(args):
    text, special_tokens = args
    return create_word_count(text, special_tokens)


def parallel_pre_tokenization(
    input_file: str,
    desired_num_chunks: int,
    split_special_token: bytes,
    special_token: list,
) -> dict[(tuple[bytes, int])]:
    """

    Returns:
        dict: _description_
    """
    num_workers = max(1, cpu_count() // 2)
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks,
            split_special_token,
        )
        tasks = []
        for start, end in zip(boundaries, boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # On Windows, text files often have \r\n (CRLF) line endings instead of \n used on unix systems
            chunk = chunk.replace('\r\n', '\n').replace('\r', '')
            tasks.append((chunk, special_token))

    with Pool(processes=num_workers) as pool:
        per_chunk_counts = pool.map(worker_count_chunk, tasks)

    # combine all per chunk dictionaries
    total_chunks_counts = Counter()
    for chunk_dict in per_chunk_counts:
        total_chunks_counts.update(chunk_dict)

    return dict(total_chunks_counts)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def read_text_file(file_path: str) -> str:
    """Read a text file and parse the contents

    Args:
        file_path (str): Path to the text file

    Returns:
        str: Text extracted from the text file
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content


def initialize_vocab(
    special_tokens: list = [
        "<|endoftext|>",
    ]
) -> dict[int, bytes]:
    """

    Args:
        special_tokens (list, optional): Special tokens used by the tokenizer . Defaults to ["<|endoftext|>",].

    Returns:
        dict[int, bytes]: Vocabulary
    """
    vocab = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}

    # create token_id for the special token and add it to the vocabulary
    for idx in special_tokens:
        special_token_id = len(vocab)
        vocab[special_token_id] = idx.encode()
    return vocab


def create_word_count(
    text: str,
    special_tokens: list[str] = ["<|endoftext|>"],
    pattern: str = PATTERN,
) -> dict[(tuple[bytes, int])]:
    """Count frequency occurrence of words in the text corpus

    Args:
        text(str): text corpus
        pattern (str): regex pattern used to split the text corpus

    Returns:
        dict[bytes, int]: Word frequency
    """
    word_counts = {}

    # remove special tokens from the corpus
    if special_tokens:
        split_pattern = "|".join(re.escape(t) for t in special_tokens)
        chunks = re.split(split_pattern, text)
    else:
        chunks = [text]

    # iterate over the chunks
    for chunk in chunks:
        if not chunk:
            continue

        # split text according to the GPT-2 pattern
        matches = re.finditer(pattern, chunk)

        # iterate over all words and return count of freq
        for match in matches:
            text_match = match.group()
            text_tokens = tuple(bytes([idx]) for idx in text_match.encode())
            word_counts[text_tokens] = word_counts.get(text_tokens, 0) + 1

    return word_counts


def merge(
    word_tuple: tuple[bytes, ...], best_pair: tuple[bytes, bytes], new_token_id: int
) -> tuple[bytes, ...]:
    """Merge best pairs in a given word tuple

    For example best_pair = (118, 21) word_tuple = (23, 87, 118, 21)  new_token_id = 257
    After performing Merging, new word tuple becomes = (23, 87, 257)

    Args:
        word_tuple (_type_): Tuple containing byte representations of the text characters
        best_pair (_type_): Best pair to merge
        new_token_id (_type_): Token id for the newly created merged token
    """

    idx = 0
    # using list because tuples are immutable therefore w cannot add elements later on
    new_word_list = []

    merged_token_byte = best_pair[0] + best_pair[1]

    while idx < len(word_tuple):
        # go through word tuple to find 2 consecutive vale same as the best pair
        # Replace with new token id otherwise add the elements to the new word list
        if (
            idx < len(word_tuple) - 1
            and (word_tuple[idx], word_tuple[idx + 1]) == best_pair
        ):
            new_word_list.append(merged_token_byte)
            idx += 2
        else:
            new_word_list.append(word_tuple[idx])
            idx += 1

    return tuple(new_word_list)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list,
    desired_num_chunks: int = 6,
    split_special_token: bytes = b'<|endoftext|>',
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a Byte Pair Encoding (BPE) tokenizer vocabulary and merge table from a text corpus.

    Parameters
    ----------
    input_path : str
        Path to the training text file.
    vocab_size : int
        Target vocabulary size. Training stops early if no mergeable pairs remain.
    desired_num_chunks : int
        Number of chunks to split the corpus into for parallel pre-tokenization.
    split_special_token : bytes
        Byte sequence used to align chunk boundaries (e.g. b"<|endoftext|>") so chunks
        begin at document boundaries.
    special_tokens : list[str], optional
        Special tokens to preserve in the vocabulary (default: ["<|endoftext|>"]).

    Returns
    -------
    vocab : dict[int, bytes]
        Mapping from token id to the token's byte string. Initial entries include all single
        bytes plus `special_tokens`. Newly learned tokens are appended during training.
    merges_list : list[tuple[bytes, bytes]]
        Ordered list of merges learned during training. Each element is a pair of byte tokens
        (left, right) that were merged into a new vocabulary entry.

    """

    # create the initial vocab
    vocab = initialize_vocab(special_tokens)
    
    
    word_count = parallel_pre_tokenization(
        input_file=input_path,
        desired_num_chunks=desired_num_chunks,
        split_special_token=split_special_token,
        special_token=special_tokens,
    )
    
    
    merges_list = []
    pair_counts = {}
    # create character pairs
    for word, freq in word_count.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] = pair_counts.get(pair,0) + freq
            
    
    while  len(vocab) < vocab_size:
        
        # get the best pair 
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p)) 
        new_token_id = len(vocab)
        merges_list.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[new_token_id] = merged_token
        
        
        new_word_count = {}
        
        for word, freq in word_count.items():
            # check if the word contains any elements in best pair
            if best_pair[0] not in word or best_pair[1] not in word:
                new_word_count[word] = freq
                continue 
            
            # find positions and remove old pairs
            for i in range(len(word) - 1):
                if(word[i], word[i+1]) == best_pair:
                    # Remove left pair
                    if i > 0:
                        left_pair = (word[i-1],word[i])
                        pair_counts[left_pair] = pair_counts.get(left_pair,0) - freq
                        
                    
                    # remove the merged pair
                    pair_counts[best_pair] = pair_counts.get(best_pair,0) - freq
                        
                    
                    # remove right pair
                    if i+2 < len(word):
                        right_pair = (word[i+1],word[i+2])
                        pair_counts[right_pair] = pair_counts.get(right_pair,0) - freq
                        
                        
            # merged word 
            new_word = merge(word,best_pair,new_token_id)
            new_word_count[new_word] = freq
            
            # find position for merged tokens and pairs before and after
            
            for i in range(len(new_word)):
                if new_word[i] == merged_token:
                    
                    # add pair to the left of the merged token
                    if i > 0:
                        new_left_pair = (new_word[i-1], merged_token)
                        pair_counts[new_left_pair] = pair_counts.get(new_left_pair,0) + freq
                    
                    # add pair to the right of the merged token
                    if i+1 < len(new_word):
                        new_right_pair = (merged_token, new_word[i+1])
                        pair_counts[new_right_pair] = pair_counts.get(new_right_pair,0) + freq 
                
        word_count = new_word_count
        
        print(f"Iter:{new_token_id}|  Merged Pair: {best_pair} ---> {vocab[new_token_id]}|  Token ID: {new_token_id}")
    
    return vocab, merges_list
            
        
    
    
    
    
