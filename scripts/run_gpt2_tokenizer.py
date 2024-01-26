#########################################################################################################################################
# The script in this file allows you to run a trained tokenizer on all raw text files in a directory. 
#########################################################################################################################################
# TODO: I should probably raise UsageErrors instead of just AssertErrors
from transformers import GPT2TokenizerFast
from pathlib import Path
import argparse
import torch

from utils.token_packer import TokenPacker

def main():
    # Accepted CLI input arguments
    parser = argparse.ArgumentParser(description="This script runs a local GPT2TokenizerFast on all raw text files in a directory. It will generate .pt files representing examples from the raw text.")
    parser.add_argument("--tokenizer-dir", type=str, required=True, help="Path to directory containing vocab and merge files that describe a trained tokenizer.")
    parser.add_argument("--dir-to-tokenize", type=str, required=False, help="Directory with raw text files to tokenize. If omitted, the text files are assumed to be in the same directory as the tokenizer.")
    parser.add_argument("--pack-examples", action='store_true', help="Whether lines from the raw text file should be packed into the same example and separated with a </s> token. Must be supplied with the --example-length flag.")
    parser.add_argument("--example-length", type=int, default=0, help="The maximum length of packed examples. If --pack-examples is omitted, this argument is ignored.")
    parser.add_argument("--delete", action='store_true', help="Whether to delete the pre-tokenized files after they are converted.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Ensure that --tokenizer-dir exists and contains .model and .vocab files describing the tokenizer
    tokenizer_dir = Path(args.tokenizer_dir)
    assert tokenizer_dir.exists(), "The value of the --tokenizer-dir argument indicates a non-existent directory!"
    merge_file, *extras = list(tokenizer_dir.glob('./*merges.txt'))
    assert not extras, "The --tokenizer-dir argument has multiple merge files!"
    vocab_file, *extras = list(tokenizer_dir.glob('./*vocab.json'))
    assert not extras, "The --tokenizer-dir argument has multiple vocab files!"

    # Check if --dir-to-tokenize contains a valid directory path. List all files that are in the directory.
    dir_to_tokenize = args.dir_to_tokenize
    if not dir_to_tokenize:  # files are assumed to be in tokenizer dir
        files_to_tokenize = [file for file in tokenizer_dir.iterdir()]
        files_to_tokenize.remove(merge_file)
        files_to_tokenize.remove(vocab_file)
    else:
        dir_to_tokenize = Path(dir_to_tokenize)
        assert dir_to_tokenize != tokenizer_dir, "To tokenize files in the same directory as the tokenizer, omit the --dir-to-tokenize argument."
        assert dir_to_tokenize.exists(), "The value of the --dir-to-tokenize argument points to a non-existent directory!"
        files_to_tokenize = [file for file in dir_to_tokenize.iterdir()]

    # Ensure that if --pack-examples is passed, so is a positive value for --example-length
    pack_examples = args.pack_examples
    example_length = args.example_length
    assert pack_examples and example_length > 0, "If --pack-examples is passed, must supply a positive value for --example-length!"

    # Initialize the tokenizer and find its pad token and EOS token ids
    tokenizer = GPT2TokenizerFast(vocab_file=vocab_file, merges_file=merge_file)
    pad_token = -1 # if examples are to be packed to size --example-length, we will fill the last example with pad tokens
    # TODO: including the eos_token requires too much complexity to be worth it until I get this codebase working from start-to-finish
    eos_token = tokenizer.eos_token_id  # separate packed examples with </s> token 

    # Run the GPT2Tokenizer on the indicated files. Each line is treated as a separate example.
    tokenized_files = []
    for file in files_to_tokenize:
        if pack_examples:
            tokenized_files.append(tokenize_file_with_packing(tokenizer, file, example_length, pad_token))
        else:
            tokenized_files.append(tokenize_file_without_packing(tokenizer, file, pad_token))
    
    # Delete the files if --delete is passed
    if args.delete:
        for file in files_to_tokenize:
            file.unlink()


def tokenize_file_with_packing(tokenizer: GPT2TokenizerFast, file: Path, example_length: int, pad_token: int) -> Path:
    '''
    Tokenize the lines of a raw text file, treating each line in the input as a separate example.
    Examples in the raw text file will be packed into the same example in the tokenized output file until 
    they reach example_length. The last example will be padded with pad_token. Returns a path to the 
    tokenized file. 
    '''
    packer = TokenPacker(example_length, pad_token=pad_token)
    with file.open('r') as f:
        for line in f:
            line = line.strip()  # clear leading/trailing whitespace, e.g. "\n"
            if not line: continue  # ignore fully empty lines, which may appear in pretraining corpus
            # TODO: I need to add [EOS] Tokens myself later. Detect them via punctuation with a whitespace after them?
            tokens = tokenizer.encode(line)
            packer.pack(tokens)
    
    tokens_as_tensor = packer.to_tensor(dtype=torch.int32)
    tensor_path = file.parent / (file.name + '.tokenized.pt')
    torch.save(tokens_as_tensor, tensor_path)
    return tensor_path


def tokenize_file_without_packing(tokenizer: GPT2TokenizerFast, file: Path, pad_token: int) -> Path:
    # TODO: If we aren't packing examples, we need to make tensors as large as the largest encoded example. 
    # One decent way to do this is with a tempfile to which we write encoded lines. As we write these encoded
    # lines, we track the largest encoded line that we write. Then add the pad_token to lines as we process them
    # a second time. 
    raise NotImplementedError

if __name__ == "__main__":
    main()
