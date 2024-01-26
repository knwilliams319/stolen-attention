#########################################################################################################################################
# Grabs the GPT2Tokenizer from HuggingFace, adds special tokens (e.g. PAD and EOS), then saves the vocabulary file to disk for use by
# run_tokenizer.py
#########################################################################################################################################
from transformers import LlamaTokenizer
from pathlib import Path
import argparse

def main():
    # Accepted CLI input arguments
    parser = argparse.ArgumentParser(description="Grabs the GPT2Tokenizer from HuggingFace, adds special tokens (e.g. PAD and EOS), then saves the vocabulary file to disk.")
    parser.add_argument("--save-dir", type=str, required=False, help="Directory in which to save the tokenizer metadata. If omitted, they'll be placed in the same location as the input data.")

    # Parse the command-line arguments and store their values
    args = parser.parse_args()
    save_dir = Path(args.save_dir).resolve()

    # Ensure save directory exists
    assert save_dir.exists(), "The value of --save-dir does not point to an existing directory!"

    # Grab the GPT2Tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    ######################################################
    # NOTE: Don't even try it. HuggingFace doesn't edit the vocabulary file that is saved to disk, even when you add special tokens yourself.
    # tokenizer.add_special_tokens(
    #     {
    #         'bos_token': '[BOS]',
    #         'eos_token': '[EOS]',
    #         'unk_token': '[UNK]',
    #         'pad_token': '[PAD]'
    #     },
    #     replace_additional_special_tokens=True
    # )
    ######################################################
    
    # Save tokenizer files to the directory
    tokenizer.save_vocabulary(save_dir.as_posix(), filename_prefix='tokenizer')

if __name__ == "__main__":
    main()
