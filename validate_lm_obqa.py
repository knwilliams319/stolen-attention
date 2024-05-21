# SECTION: Necessary imports
import argparse
from pathlib import Path
import shutil
# import pickle

import torch
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L
from sentencepiece import SentencePieceProcessor
import pandas as pd
import numpy as np
import scipy.spatial as sp
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

from finetune_lm import OpenbookQADataset, OpenbookQAModel
#!SECTION

# SECTION: Model definition to extend OpenbookQAModel to capture and save attention statistics
class CaptureStatsOpenbookQAModel(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = OpenbookQAModel.load_from_checkpoint(opt.model_path)
        assert opt.stats_last_k_layers <= self.model.hparams.num_layers, "Attempted to save data from more layers than the model has!"
        self.save_after_k = self.model.hparams.num_layers - opt.stats_last_k_layers
        self.num_correct = 0
        self.num_seen = 0
        self.batch_offset = 0

    def forward(self, x, mask=None):
        return self.model(x, mask=mask, save_after_k=self.save_after_k)

    def validation_step(self, batch, batch_idx):
        _, num_correct = self.model._calculate_loss(batch, save_after_k=self.save_after_k)
        self.num_correct += num_correct

        bsz, max_len = batch[0].size()
        q_lens = batch[2].sum(dim=-1)

        for l, layer in enumerate(self.model.transformer.layers):
            if l >= self.save_after_k:
                # Grab and condense stats for the layer
                attn_module = layer.self_attn
                query_norms = torch.norm(attn_module.query_points, dim=-1).mT        # [bsz, n_heads] --> [n_heads, bsz]
                k_norms = torch.norm(attn_module.k_embed, dim=-1).transpose(-2, 0)   # [bsz, n_heads, max_len] --> [n_heads, bsz, max_len]
                weights = attn_module.attn_weights.transpose(-2, 0)                  # [bsz, n_heads, max_len] --> [n_heads, bsz, max_len]

                # Individually log stats for each head in the layer
                for h in range(self.model.hparams.num_heads):
                    # Build batch-level statistics into the dictionary and log them repeatedly for the layer/head combination
                    for b_idx in range(bsz):
                        stats_dict = {"layer": float(l), "head": float(h)}  # identify the layer/head for which we're logging
                        stats_dict['question_len'] = float(q_lens[b_idx])
                        stats_dict['query_norm'] = query_norms[h][b_idx]

                        for s_idx in range(max_len):
                            stats_dict[f'k_norm_{s_idx}'] = k_norms[h][b_idx][s_idx]
                            stats_dict[f'k_weight_{s_idx}'] = weights[h][b_idx][s_idx]

                        # NOTE: Using self.logger directly is NECESSARY. Else, we will only log once per step, which
                        #       means that we'll only capture statistics for the last batch of the head (instead of all head/batch combos)
                        self.logger.log_metrics(stats_dict, step=b_idx+self.batch_offset) # step is equivalent to a question_id column since shuffle=False

        # Update batch offset to properly build the question_id field beyond the first validation batch
        self.batch_offset += bsz
        self.num_seen += bsz
# !SECTION: End of extended OpenbookQAModel
    
# SECTION: Custom logger to use Lightning's built-in logging framework to log statistics for each head of each layer
class NestedCSVLogger(Logger):
    def __init__(self, opt):
        super().__init__()
        model = OpenbookQAModel.load_from_checkpoint(opt.model_path)
        layers = []
        for i in range(model.hparams.num_layers):
            if i >= model.hparams.num_layers - opt.stats_last_k_layers:
                layers.append(i)
        self.num_heads = model.hparams.num_heads

        self.base_save_dir = opt.save_dir

        self.loggers = {}
        for layer in layers:
            self.loggers[layer] = {}
            for h in range(self.num_heads):
                self.loggers[layer][h] = CSVLogger(
                    self.base_save_dir / f'layer_{layer}',
                    name='',
                    version=f'head_{h}'
                )

    @property
    def name(self):
        return ""
    
    @property
    def version(self):
        return self.base_save_dir.name
    
    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass
    
    def get_logger(self, layer_idx, head_idx):
        return self.loggers[layer_idx][head_idx]

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # Use the 'layer' and 'head' keys to decide which logger to choose
        assert 'layer' in metrics
        assert 'head' in metrics
        logger = self.get_logger(metrics['layer'], metrics['head'])

        # Then log the remaining metrics using that logger's log function
        del metrics['layer']
        del metrics['head']
        logger.log_metrics(metrics, step)

    @rank_zero_only
    def save(self):
        for layer in self.loggers:
            for head in range(self.num_heads):
                self.loggers[layer][head].save()
# !SECTION: Custom logger
  
# SECTION: Paths and constants used for default arguments below
base = Path(__file__).parent
TOKENIZER_PATH = (base / "unigram-tokenizer/tokenizer.model").as_posix()        # NOTE: needs to be a string
EXPERIMENT_DIR = base / 'experiments/embed_dim_512/8_heads/base/finetune_bsz_1'
OBQA_VALID_PATH = base / 'floyd-finetune/data/obqa.valid.txt'
#!SECTION
        
# SECTION: Training loop
if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--batchsize', type=int, default=64) # TODO: change this back to 64
    parser.add_argument('--tokenizer-path', type=str, default=TOKENIZER_PATH)
    parser.add_argument('--use-last', action='store_true')  # default = use best weights
    parser.add_argument('--experiment-dir', type=Path, default=EXPERIMENT_DIR)
    parser.add_argument('--valid-path', type=Path, default=OBQA_VALID_PATH)
    parser.add_argument('--save-stats', action='store_true')
    parser.add_argument('--stats-last-k-layers', type=int, default=0)
    opt = parser.parse_args()

    # Post-process some of the CLI arguments
    pattern = 'backup-state*.ckpt' if opt.use_last else 'best-weights*.ckpt'
    model_paths = opt.experiment_dir.glob(pattern)
    opt.model_path, *extras = model_paths
    assert not extras, f"More than one checkpoint matched {pattern} in {opt.experiment_dir}!"
    opt.device = "cpu" if opt.no_cuda else "gpu"
    if opt.device == "gpu":
        assert torch.cuda.is_available()
    if opt.save_stats:
        assert opt.stats_last_k_layers > 0, "--save-stats was passed, so --stats-last-k-layers should be >0!"
        opt.save_dir = opt.model_path.parent / 'attention_stats'
    else:
        assert opt.stats_last_k_layers == 0, "--stats-last-k-layers was >0, but --save-stats was not passed!"

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=opt.tokenizer_path)

    # Create Dataloader
    val_dataset = OpenbookQADataset(opt.valid_path, tokenizer)
    val_loader = data.DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=3)

    # Set up for training. Set random seeds and initialize Trainer. 
    L.seed_everything(7, workers=True)
    trainer = L.Trainer(
        benchmark=True,
        deterministic=False,
        default_root_dir=None,
        enable_progress_bar=True,
        accelerator=opt.device,          
        strategy="ddp",
        devices=1,                  
        precision="16-mixed",
        logger=NestedCSVLogger(opt) if opt.save_stats else False,  # False => no logger
        log_every_n_steps=1,
        limit_val_batches=None,  # Might need to use this for higher dimensional models
    )

    # Instantiate the finetuned model, and validate it
    model = CaptureStatsOpenbookQAModel(opt)
    trainer.validate(model, dataloaders=val_loader, verbose=True)
    print(f"Accuracy: {model.num_correct / model.num_seen}")

    # If we were saving stats, post-process the logs
    # This loop will compress the directory structure and delete all hparams.yaml files (which are saved by default and I can't turn off)
    if opt.save_stats:
        process_dir = opt.save_dir.parent / 'attn_stats'
        for layer_dir in opt.save_dir.glob('*'):
            (process_dir / layer_dir.name).mkdir(exist_ok=False, parents=True)

            for csv_path in layer_dir.glob('*/metrics.csv'):
                # Use the CSV path to create a new path at which to save a processed copy
                subfolder = f'{csv_path.parts[-3]}' # layer_j
                name = f'{csv_path.parts[-2]}.csv'  # head_i.csv
                new_path = process_dir / subfolder / name

                # Use pandas to order the columns alphanumerically
                df = pd.read_csv(csv_path)
                df = df[df.columns.sort_values()]

                # Save the processed dataframe at the new location
                df.to_csv(new_path, index=False)

    shutil.rmtree(opt.save_dir) # Remove the original directory now that everything is processed
#!SECTION