from bpe import train_bpe
from time import time
import json
import logging

logging.basicConfig(filename="bpe_training.log",
                    format='%(asctime)s %(levelname)s: %(message)s',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def serialize_to_disk(
    vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], dataset_name: str
):
    """Serialize a trained BPE vocabulary and merge rules to disk. This creates files for later loading or inspection of the learned tokenizer.

    Args:
        vocab: Mapping from token IDs to their corresponding byte representations.
        merges: Ordered list of byte-pair merges representing the learned BPE merge operations.
        dataset_name: Base name used to construct the output file names for the vocab and merges.
    """
    # save vocab
    vocab_json = {}
    for token_id, token_byte in vocab.items():
        # convert bytes to str, ensure key is string for JSON
        vocab_json[str(token_id)] = token_byte.decode("utf-8", errors="ignore")
    with open(f"./{dataset_name}_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2)

    # save merges
    with open(f"./{dataset_name}_merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            left_str = left.decode("utf-8", errors="ignore")
            right_str = right.decode("utf-8", errors="ignore")
            f.write(f"{left_str} {right_str}\n")



def get_longest_token(vocab: dict[int, bytes]):
    
    token_lengths = []
    for token_id, token_bytes in vocab.items():
        token_str = token_bytes.decode("utf-8", errors="ignore")
        token_lengths.append((len(token_bytes), len(token_str), token_str))
        
    # sort by byte length
    token_lengths.sort(reverse=True)
    return token_lengths
        
        
    

if __name__ == "__main__":
    datasets = {
        "tinystories": {
            "dataset_path": "../../data/TinyStoriesV2-GPT4-train.txt",
            "vocab_size": 10000,
            "special_tokens": ["<|endoftext|>"],
               "document_chunks": 10
        },
        "openwebtext": {
            "dataset_path": "../../data/owt_train.txt",
            "vocab_size": 32000,
            "special_tokens": ["<|endoftext|>"],
            "document_chunks": 60
        },
    }
    
    
    try:
        

        for dataset_name, config in datasets.items():
            
            logger.info(f"-------------Starting training for {dataset_name}--------------------------")

            # train bpe on dataset with all required parameters
            vocab, merges = train_bpe(
                input_path=config["dataset_path"],
                vocab_size=config["vocab_size"],
                special_tokens=config["special_tokens"],
                desired_num_chunks=config["document_chunks"]
            )
            
            # get the longest token
            longest_token = get_longest_token(vocab)
            logger.info("Top 10 longest tokens:")
            for i in range(10):
                token = longest_token[i]           
                logger.info(f"Token {10-i}: {token[2]} ({token[0]} bytes)")

            
            # serialize to disk
            serialize_to_disk(vocab, merges, dataset_name)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)