# SECTION: Necessary imports
import torch
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.profilers import AdvancedProfiler
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import numpy as np
import pickle
# from transformers import GPT2TokenizerFast

from modules import CausalTransformer
#!SECTION

# SECTION: Dataloaders and LightningModules
class Wikitext103Dataset(data.Dataset):
    def __init__(self, tokens_path: str, pad_id: int, vocab_size: int):
        super().__init__()
        self.data = torch.load(tokens_path)
        self.pad_id = pad_id
        self.vocab_size = vocab_size

    @property
    def context_length(self):
        return self.data.size(1)

    def __len__(self):
        # Skip last batch, which is the only incomplete one (due to packing)
        # More importantly, we need label for all tokens of the input, but the last batch can't look ahead
        return self.data.size(0) - 1 

    def __getitem__(self, idx):
        tokens = self.data[idx]
        padding_mask = torch.zeros(self.context_length)
        labels = torch.cat([
            tokens[1:],
            torch.tensor([self.data[idx+1][0]], dtype=tokens.dtype)
        ])

        return tokens, labels, padding_mask
    
class FlattenedWikitext103Dataset(data.Dataset):
    def __init__(self, tokens_path: str, pad_id: int, vocab_size: int):
        super().__init__()
        # Load packed tokens and store context length before flattening them
        self.data = torch.load(tokens_path)
        self.context_length = self.data.size(1)
        self.data = torch.flatten(self.data)
        self.pad_id = pad_id
        self.vocab_size = vocab_size

        # Find last index at which tokens exist, as there may be padding tokens in the last packed batch
        self.num_tokens = 0
        for i in range(self.data.size(0)):
            if self.data[i] == self.pad_id:
                break
        self.num_tokens = i

    def __len__(self):
        return self.num_tokens - self.context_length

    def __getitem__(self, idx):
        tokens = self.data[idx:idx+self.context_length]
        padding_mask = torch.zeros(self.context_length)
        labels = self.data[idx+1:idx+self.context_length+1]
        return tokens, labels, padding_mask
# !SECTION

# SECTION: Model definitions
global sliding_mode # if True (default False), only the predictions on the last token of the batch will influence loss

class Wikitext103Model(CausalTransformer):
    def __init__(self, **model_kwargs):
        # Initialize model as per usual, but add extra state to track token statistics
        super().__init__(**model_kwargs)
        self.losses = {}      # KEY: token_id; VALUE: list of loss values for the prediction over that token
        # self.norms = {}       # KEY: (layer_idx, head_idx, token_id); VALUE: list of embedding norms for that token in that head of that layer
        # self.is_vertex = {}   # KEY: (layer_idx, head_idx, token_id); VALUE: list of vertex membership for that token in that head of that layer
        for t in range(self.hparams.num_classes):
            self.losses[t] = []
            # for l in range(self.hparams.num_layers):
            #     for h in range(self.hparams.num_heads):
            #         self.norms[(l, h, t)] = []
            #         self.is_vertex[(l, h, t)] = []

    def _calculate_loss(self, batch):
        data, labels, mask = batch
        data = data.int()
        preds = self.forward(data, pad_mask=mask) # shape = [bsz, context_len, vocab_size]

        if not sliding_mode:
            # Get predictions over all tokens in all batches
            preds = preds.view(-1, preds.size(-1))
            labels = labels.view(-1).long()
            loss = F.cross_entropy(preds, labels)  
            return loss
        else:
            # Grab predictions on the last element of the context for all batches
            # NOTE: Using -1 will get predictions on the 513th element given the previous 512, which is what we 
            #       generally want to measure with sliding window inference. But, we can't measure the norm of the
            #       embedding of elements that don't get crunched by the model's attention heads.
            preds = preds[:, -2]
            labels = labels[:, -2].long()

            # Track statistics
            loss = F.cross_entropy(preds, labels, reduction='none')
            labels_cpu = labels.cpu().numpy()
            loss_cpu = loss.detach().cpu()
            for batch_idx in range(loss_cpu.size(0)):
                token_id = labels_cpu[batch_idx]
                self.losses[token_id].append(loss_cpu[batch_idx].item())
            return torch.mean(loss)  # return mean so that logging doesn't error

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log(
            "train_loss",
            loss, 
            sync_dist=True,        # this doesn't seem to impact training time, likely because we have only 3 devices
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,  # this seems to slightly speed up training
            prog_bar=True
        )

        # calculate norms for total update and layers' updates
        total_norm = 0.0
        layer_grad_norms = [0.0] * self.hparams.num_layers
        for name, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2).item() ** 2
                total_norm += param_norm
                for i in range(self.hparams.num_layers):
                    if name.startswith(f'transformer.layers.{i}'):
                        layer_grad_norms[i] += param_norm
                        break
        for norm in layer_grad_norms:
            norm = norm ** (1. / 2)
        total_norm = total_norm ** (1. / 2)
        self.log(
            "grad_norm",
            total_norm,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
            prog_bar=True
        )
        for i, norm in enumerate(layer_grad_norms):
            self.log(
            f"layer_norm_{i}",
            norm,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
            prog_bar=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log(
            "val_loss",
            loss, 
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False
        )

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log(
            "test_loss", 
            loss, 
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False
        )
#!SECTION
  
# SECTION: Training parameters
# TODO: make these CLI arguments instead of constants 
CHECKPOINT_BASE = "./experiments/embed_dim_64/n_heads_8"
EXPERIMENT = "man-positional"
CHECKPOINT_DIR = CHECKPOINT_BASE + '/' + EXPERIMENT
VALID_PATH = "./data/wikitext-103/unigram.wiki.valid.tokens.tokenized.pt"
TOKENIZER_PATH = "./unigram-tokenizer/tokenizer.model"
#!SECTION
        
# SECTION: Training loop
if __name__ == "__main__":
    # Set up for training. Set random seeds and initialize Trainer. 
    L.seed_everything(7, workers=True)
    trainer = L.Trainer(
        deterministic=False,      # Doesn't matter since val_loaders aren't shuffling 
        default_root_dir=None,
        enable_progress_bar=True,
        accelerator="gpu",          
        strategy="ddp",
        devices=[1],                  
        precision="16-mixed",     # NOTE: Might need to be 32-true depending on the checkpoint
        benchmark=True,
        logger=False,             # Turns off creation of 'lightning_logs' directory
        limit_test_batches=None   # Might need to use this for higher dimensional models
    )

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=TOKENIZER_PATH)

    # Create dataloaders
    BATCH_SIZE = 448
    val_dataset = Wikitext103Dataset(VALID_PATH, tokenizer.pad_id(), len(tokenizer))
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4, persistent_workers=False)

    # NOTE: `val_dataset_flat` should ideally be a FlattenedWikitext103Dataset object. However, there are so many batches in this setting,
    #       that the total inference time would be 1.5 hours on my Mac. The normal validation set has 264 batches, which is a large enough
    #       number to get some rough estimates for now.
    val_dataset_flat = FlattenedWikitext103Dataset(VALID_PATH, tokenizer.pad_id(), len(tokenizer))
    val_loader_flat = data.DataLoader(val_dataset_flat, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=3, persistent_workers=True)

    # Load pretrained model
    checkpoint_dir = Path(CHECKPOINT_DIR)
    pretrained_file_path = list(checkpoint_dir.glob('epoch=24-step=*.ckpt'))[0]
    if pretrained_file_path.exists() and pretrained_file_path.is_file():
        print("Found pretrained model, loading...")
        model = Wikitext103Model.load_from_checkpoint(pretrained_file_path)
    else:
        raise FileNotFoundError(f'No checkpoint exists at {pretrained_file_path}!')
    
    # Validate model
    # print("Testing Normal Inference on Validation Set")
    # sliding_mode = False
    # trainer.test(model, dataloaders=val_loader, verbose=True)
    # with open(f'q-hull-props-{EXPERIMENT}.pkl', 'wb') as q_hull_file:
    #     pickle.dump(model.q_hull_props,q_hull_file)
    # with open(f'k-hull-props-{EXPERIMENT}.pkl', 'wb') as k_hull_file:
    #     pickle.dump(model.k_hull_props,k_hull_file)

    print("Testing Sliding Window Inference on Validation Set")
    sliding_mode = True
    trainer.test(model, dataloaders=val_loader_flat, verbose=True)
    avg_losses = [pd.NA]*16000
    variances = [pd.NA]*16000
    counts = [pd.NA]*16000
    for token_id, losses_array in model.losses.items():
        if losses_array:
            avg_losses[token_id] = np.mean(losses_array)
            variances[token_id] = np.var(losses_array)
            counts[token_id] = len(losses_array)
    statistics = pd.DataFrame({
        'token_id': np.arange(16000),
        'avg_loss': avg_losses,
        'loss_variance': variances,
        'val_freq': counts
    })
    statistics_path = checkpoint_dir / f'val-loss-stats.csv'
    statistics.to_csv(statistics_path, index=False)
    print("Done")

#!SECTION