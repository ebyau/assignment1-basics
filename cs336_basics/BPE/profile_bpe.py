from bpe import train_bpe

# This actually runs the BPE training
vocab, merges = train_bpe(
    input_path="../../tests/fixtures/corpus.en",
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)

print(f"Vocab size: {len(vocab)}, Merges: {len(merges)}")




# python -m cProfile -s cumtime profile_bpe.py