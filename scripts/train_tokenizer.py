#########################################################################################################################################
# The script in this file allows you to train a tokenizer from scratch on a text file in the local filesystem. 
# The tokenizer data files will be generated in the same directory as the input data. 
#########################################################################################################################################
from sentencepiece import SentencePieceTrainer
from pathlib import Path
import argparse

def main():
    # Accepted CLI input arguments
    parser = argparse.ArgumentParser(description="This script trains a tokenizer from scratch on raw text files using the SentencePiece implementation.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw text file containing training data")
    parser.add_argument("--vocab-size", type=int, required=True, help="The maximum vocabulary size of the tokenizer to be trained")
    parser.add_argument("--char-cov", type=float, default=1.0, help="The proportion of characters covered by the model; 1.0 (default) is fine for languages with small character sets")
    parser.add_argument("--backend", type=str, default="bpe", help="The token-encoding algorithm to be used. Accepted values: [bpe (default), char, word, unigram]. If word is used, input sentences must be pre-tokenized")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Assert that only one input path was given. While the tokenizer can accept multiple files, 
    # to prevent bad practices only the training dataset should be used to create the tokenizer.
    input_paths = args.input.split(',')
    input_path, *extras = input_paths
    assert not extras, "The tokenizer can only be trained on a single file!"

    # Store the other arguments' values
    vocab_size = args.vocab_size
    char_cov = args.char_cov
    backend = args.backend

    # Run the SentencePiece Trainer
    # LINK: https://github.com/google/sentencepiece#train-sentencepiece-model (may be useful if we ever want to add our own special tokens)
    # NOTE: Setting token_id=-1 disables that token.
    SentencePieceTrainer.train(input=input_path, 
                               model_prefix='tokenizer',
                               vocab_size=vocab_size, 
                               character_coverage=char_cov,
                               model_type=backend,
                               pad_id=0,
                               unk_id=1,
                               bos_id=-1,
                               eos_id=2)
    
    # Locate outputted tokenizer files from SentencePieceTrainer.train()
    # NOTE: These should match the 'model_prefix' keyword argument on L35
    cwd = Path().resolve()
    model_path = cwd / 'tokenizer.model'
    vocab_path = cwd / 'tokenizer.vocab'

    # Move the files to the same location as the input data
    model_dst = Path(input_path).parent / model_path.name
    vocab_dst = Path(input_path).parent / vocab_path.name
    model_path.rename(model_dst)
    vocab_path.rename(vocab_dst)

if __name__ == "__main__":
    main()
