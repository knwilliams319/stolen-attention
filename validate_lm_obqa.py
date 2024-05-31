# SECTION: Necessary imports
import argparse
from pathlib import Path
import shutil
# import pickle

import torch
# import torch.nn.functional as F
import torch.utils.data as data
import lightning as L
from sentencepiece import SentencePieceProcessor
import pandas as pd
# import numpy as np
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
        self.calculate_hull = opt.calculate_hull

    def forward(self, x, mask=None):
        return self.model(x, mask=mask, save_after_k=self.save_after_k)
    
    def _calculate_angle(self, query, key):
        # Compute the dot product
        dot_product = torch.dot(query, key)
        
        # Compute the magnitudes (norms) of the vectors
        norm_a = torch.norm(query)
        norm_b = torch.norm(key)
        
        # Compute the cosine of the angle
        cos_theta = dot_product / (norm_a * norm_b)
        
        # Clip the cosine value to the range [-1, 1] to avoid numerical issues with arccos
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # Compute the angle in radians
        angle_rad = torch.acos(cos_theta)
        
        return angle_rad.item(), norm_b.item()

    def validation_step(self, batch, batch_idx):
        _, num_correct = self.model._calculate_loss(batch, save_after_k=self.save_after_k)
        self.num_correct += num_correct

        bsz, max_len = batch[0].size()       # batch size and number of maximum possible positions
        n_pad_tokens = batch[2].sum(dim=-1)  # find number of pad positions

        for l, layer in enumerate(self.model.transformer.layers):
            if l >= self.save_after_k:
                # Grab and condense stats for the layer
                attn_module = layer.self_attn

                queries = attn_module.query_points.cpu().float().transpose(0, 1) # [bsz, n_heads, head_dim] --> [n_heads, bsz, head_dim]
                k_embed = attn_module.k_embed.cpu().float().transpose(0, 1)      # [bsz, n_heads, seq_len, head_dim] --> [n_heads, bsz, seq_len, head_dim]
                weights = attn_module.attn_weights.float().transpose(0, 1)       # [bsz, n_heads, max_len] --> [n_heads, bsz, max_len]

                # Individually log stats for each head in the layer
                for h in range(self.model.hparams.num_heads):
                    # Build batch-level statistics into the dictionary and log them repeatedly for the layer/head combination
                    for b_idx in range(bsz):
                        stats_dict = {"layer": float(l), "head": float(h)}  # identify the layer/head for which we're logging
                        stats_dict['question_len'] = float(max_len - n_pad_tokens[b_idx])
                        if self.calculate_hull:
                            k_hull = sp.ConvexHull(k_embed[h][b_idx][n_pad_tokens[b_idx]:])         # only count non-padded positions
                            vertices = k_hull.vertices + n_pad_tokens[b_idx].item()

                        for s_idx in range(max_len):
                            if s_idx >= n_pad_tokens[b_idx].item():
                                stats_dict[f'angle_{s_idx}'], stats_dict[f'norm_{s_idx}'] = self._calculate_angle(queries[h][b_idx], k_embed[h][b_idx][s_idx])
                                stats_dict[f'weight_{s_idx}'] = weights[h][b_idx][s_idx].item()
                                stats_dict[f'token_{s_idx}'] = batch[0][b_idx][s_idx]
                                if self.calculate_hull:
                                    stats_dict[f'k_vertex_{s_idx}'] = s_idx in vertices
                            else:
                                stats_dict[f'q_angle_{s_idx}'] = pd.NA
                                stats_dict[f'norm_{s_idx}'] = pd.NA
                                stats_dict[f'weight_{s_idx}'] = pd.NA
                                stats_dict[f'token_{s_idx}'] = 0
                                if self.calculate_hull:
                                    stats_dict[f'k_vertex_{s_idx}'] = pd.NA

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
EXPERIMENT_DIR = base / 'experiments/embed_dim_512/8_heads/base/finetune'
OBQA_TRAIN_PATH = base / 'data/openbookqa/obqa.train.txt'
OBQA_VALID_PATH = base / 'data/openbookqa/obqa.valid.txt'
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
    parser.add_argument('--train-path', type=Path, default=OBQA_TRAIN_PATH)
    parser.add_argument('--use-train', action='store_true') # default = use validation set
    parser.add_argument('--save-stats', action='store_true')
    parser.add_argument('--stats-last-k-layers', type=int, default=0)
    parser.add_argument('--calculate-hull', action='store_true')
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
        assert opt.stats_last_k_layers == 0, "--stats-last-k-layers was >0, but --save-stats was not set!"
        assert opt.calculate_hull == False, "--calculate-hull was passed, but --save-stats was not set!"
    opt.data_path = opt.train_path if opt.use_train else opt.valid_path

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=opt.tokenizer_path)

    # Create Dataloader
    val_dataset = OpenbookQADataset(opt.data_path, tokenizer)
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