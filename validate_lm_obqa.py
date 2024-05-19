# SECTION: Necessary imports
import argparse
from pathlib import Path
# import pickle

import torch
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L
from sentencepiece import SentencePieceProcessor
import pandas as pd
import numpy as np
import scipy.spatial as sp
# from sklearn.metrics import confusion_matrix

from finetune_lm import OpenbookQADataset, OpenbookQAModel # TODO: may need to extend these models to collect extra statistics
#!SECTION
  
# SECTION: Paths and constants used for default arguments below
base = Path(__file__).parent
TOKENIZER_PATH = (base / "unigram-tokenizer/tokenizer.model").as_posix()    # NOTE: needs to be a string
EXPERIMENT_DIR = base / 'experiments/embed_dim_512/128_heads/base/finetune'
OBQA_VALID_PATH = base / 'floyd-finetune/data/obqa.valid.txt'
#!SECTION
        
# SECTION: Training loop
if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-batchsize', type=int, default=64)
    parser.add_argument('-tokenizer_path', type=str, default=TOKENIZER_PATH)
    parser.add_argument('-use_last', action='store_true') # default = use best weights
    parser.add_argument('-experiment_dir', type=Path, default=EXPERIMENT_DIR)
    parser.add_argument('-valid_path', type=Path, default=OBQA_VALID_PATH)
    opt = parser.parse_args()

    # Post-process some of the CLI arguments
    pattern = 'backup-state*.ckpt' if opt.use_last else 'best-weights*.ckpt'
    model_paths = opt.experiment_dir.glob(pattern)
    opt.model_path, *extras = model_paths
    if extras:
        raise ValueError(f"More than one checkpoint matched {pattern} in {opt.experiment_dir}!")
    opt.device = "cpu" if opt.no_cuda else "gpu"
    if opt.device == "gpu":
        assert torch.cuda.is_available()

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=opt.tokenizer_path)

    # Create Dataloader
    val_dataset = OpenbookQADataset(opt.valid_path, tokenizer)
    val_loader = data.DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=3)

    # Set up for training. Set random seeds and initialize Trainer. 
    L.seed_everything(7, workers=True)
    trainer = L.Trainer(
        benchmark=True,            # NOTE: can't be used with deterministic=True (but we don't care because val_loaders aren't shuffling)
        deterministic=False,
        default_root_dir=None,     # We aren't saving any stats to disk
        enable_progress_bar=True,
        accelerator=opt.device,          
        strategy="ddp",
        devices=1,                  
        precision="16-mixed",
        logger=False,              # Turns off creation of 'lightning_logs' directory
        limit_test_batches=None    # Might need to use this for higher dimensional models
    )

    # Instantiate the finetuned model, and validate it
    model = OpenbookQAModel.load_from_checkpoint(opt.model_path)
    trainer.validate(model, dataloaders=val_loader, verbose=True)
#!SECTION