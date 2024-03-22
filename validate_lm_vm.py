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
from sklearn.metrics import confusion_matrix
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
        # self.losses = {}    # KEY: token_id; VALUE: list of loss values for the prediction over that token
        self.norms = {}       # KEY: (layer_idx, token_id, q/k); VALUE: list of embedding norms for that token in the Q/K matrix of first head of that layer
        self.is_vertex = {}   # KEY: (layer_idx, token_id, q/k); VALUE: list of vertex membership for that token in Q/K matrix of the first head of that layer
        for t in range(self.hparams.num_classes):
            # self.losses[t] = []
            for l in range(self.hparams.num_layers):
                self.norms[(l, t)] = {}
                self.is_vertex[(l, t)] = {}
                for m in ['q', 'k']:
                    self.norms[(l, t)][m] = []
                    self.is_vertex[(l, t)][m] = []

    def _calculate_loss(self, batch):
        data, labels, mask = batch
        data = data.int()
        preds = self.forward(data, pad_mask=mask) # shape = [bsz, context_len, vocab_size]

        if not sliding_mode:
            # Get predictions over all tokens in all batches
            raise ValueError('sliding_mode is set to False!')
            # preds = preds.view(-1, preds.size(-1))
            # labels = labels.view(-1).long()
            # loss = F.cross_entropy(preds, labels)  
            # return loss
        else:
            # Grab predictions on the last element of the context for all batches
            # NOTE: Using -1 will get predictions on the 513th element given the previous 512, which is what we 
            #       generally want to measure with sliding window inference. But, we can't measure the norm of the
            #       embedding of elements that don't get crunched by the model's attention heads.
            preds = preds[:, -2]
            labels = labels[:, -2].long()

            # Track statistics
            # loss = F.cross_entropy(preds, labels, reduction='none')
            # labels_cpu = labels.cpu().numpy()
            # loss_cpu = loss.detach().cpu()
            # for batch_idx in range(loss_cpu.size(0)):
            #     token_id = labels_cpu[batch_idx]
            #     self.losses[token_id].append(loss_cpu[batch_idx].item())
            # return torch.mean(loss)  # return mean so that logging doesn't error

            # Track Q/K Hull statistics
            # NOTE: When calculating Q/K Hull statistics, batch size should be 1
            token_id = data[0][-1].item()
            for i, layer in enumerate(self.transformer.layers):
                if layer.self_attn.last_token_q_norm: # will be None for a deficient layer
                    self.norms[(i, token_id)]['q'].append(layer.self_attn.last_token_q_norm)
                    self.is_vertex[(i, token_id)]['q'].append(layer.self_attn.last_token_q_vertex)
                if layer.self_attn.last_token_k_norm: # will be None for a deficient layer
                    self.norms[(i, token_id)]['k'].append(layer.self_attn.last_token_k_norm)
                    self.is_vertex[(i, token_id)]['k'].append(layer.self_attn.last_token_k_vertex)  

            # we're doing nothing special with loss for Q/K stats
            return F.cross_entropy(preds, labels) 

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
CHECKPOINT_BASE = "./experiments/embed_dim_64/n_heads_16"
EXPERIMENT = "base"
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
        devices=[2],                  
        precision="16-mixed",     # NOTE: Might need to be 32-true depending on the checkpoint
        benchmark=True,
        logger=False,             # Turns off creation of 'lightning_logs' directory
        limit_test_batches=None   # Might need to use this for higher dimensional models
    )

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=TOKENIZER_PATH)

    # Create dataloaders
    BATCH_SIZE = 1
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

    # avg_losses = [pd.NA]*16000
    # variances = [pd.NA]*16000
    # counts = [pd.NA]*16000
    # for token_id, losses_array in model.losses.items():
    #     if losses_array:
    #         avg_losses[token_id] = np.mean(losses_array)
    #         variances[token_id] = np.var(losses_array)
    #         counts[token_id] = len(losses_array)
    # statistics = pd.DataFrame({
    #     'token_id': np.arange(16000),
    #     'avg_loss': avg_losses,
    #     'loss_variance': variances,
    #     'val_freq': counts
    # })
    # statistics_path = checkpoint_dir / f'val-loss-stats.csv'
    # statistics.to_csv(statistics_path, index=False)

    q_hull_norms = {}
    q_hull_norms_vars = {}
    q_hull_props = {}
    k_hull_norms = {}
    k_hull_norms_vars = {}
    k_hull_props = {}
    q_hull_counts = {}
    k_hull_counts = {}
    q_and_k = {} 
    q_not_k = {}
    k_not_q = {}
    neither = {}
    for i in range(model.hparams.num_layers):
        q_hull_norms[f'layer_{i}_q_norm'] = [pd.NA]*16000
        q_hull_props[f'layer_{i}_q_prop'] = [pd.NA]*16000
        q_hull_norms_vars[f'layer_{i}_q_norm_var'] = [pd.NA]*16000
        k_hull_norms[f'layer_{i}_k_norm'] = [pd.NA]*16000
        k_hull_props[f'layer_{i}_k_prop'] = [pd.NA]*16000
        k_hull_norms_vars[f'layer_{i}_k_norm_var'] = [pd.NA]*16000
        q_and_k[f'layer_{i}_q_and_k'] = [pd.NA]*16000
        q_not_k[f'layer_{i}_q_not_k'] = [pd.NA]*16000
        k_not_q[f'layer_{i}_k_not_q'] = [pd.NA]*16000
        neither[f'layer_{i}_not_q_nor_k'] = [pd.NA]*16000
        q_hull_counts[f'layer_{i}_q_count'] = [pd.NA]*16000
        k_hull_counts[f'layer_{i}_k_count'] = [pd.NA]*16000
    for (layer, token), d in model.norms.items():
        q_norms, k_norms = d['q'], d['k']
        if len(q_norms) > 0:
            q_hull_norms[f'layer_{layer}_q_norm'][token] = np.mean(q_norms)
            q_hull_norms_vars[f'layer_{layer}_q_norm_var'][token] = np.var(q_norms)
            q_hull_counts[f'layer_{layer}_q_count'][token] = len(q_norms)
        if len(k_norms) > 0:
            k_hull_norms[f'layer_{layer}_k_norm'][token] = np.mean(k_norms)
            k_hull_norms_vars[f'layer_{layer}_k_norm_var'][token] = np.var(k_norms)
            k_hull_counts[f'layer_{layer}_k_count'][token] = len(k_norms)
    for (layer, token), d in model.is_vertex.items():
        q_is_vertex, k_is_vertex = d['q'], d['k']
        if do_q := (len(q_is_vertex) > 0):
            q_hull_props[f'layer_{layer}_q_prop'][token] = np.mean(q_is_vertex)
        if do_k := (len(k_is_vertex) > 0):
            k_hull_props[f'layer_{layer}_k_prop'][token] = np.mean(k_is_vertex)
        if do_q and do_k:
            try:
                bins = confusion_matrix(q_is_vertex, k_is_vertex)
                if len(bins) == 1: # occurs if q_is_vertex == k_is_vertex for all elements
                    if q_is_vertex[0]:
                        q_and_k[f'layer_{layer}_q_and_k'][token] = len(q_is_vertex)
                        q_not_k[f'layer_{layer}_q_not_k'][token] = 0
                        k_not_q[f'layer_{layer}_k_not_q'][token] = 0
                        neither[f'layer_{layer}_not_q_nor_k'][token] = 0
                    else:
                        q_and_k[f'layer_{layer}_q_and_k'][token] = 0
                        q_not_k[f'layer_{layer}_q_not_k'][token] = 0
                        k_not_q[f'layer_{layer}_k_not_q'][token] = 0
                        neither[f'layer_{layer}_not_q_nor_k'][token] = len(q_is_vertex)
                else:
                    q_and_k[f'layer_{layer}_q_and_k'][token] = bins[1][1]
                    q_not_k[f'layer_{layer}_q_not_k'][token] = bins[1][0]
                    k_not_q[f'layer_{layer}_k_not_q'][token] = bins[0][1]
                    neither[f'layer_{layer}_not_q_nor_k'][token] = bins[0][0]
            except ValueError: # occurs if q, k have unequal numbers of elements
                pass
    statistics_dict = {}
    for d in [q_hull_norms, q_hull_norms_vars, q_hull_props, 
              k_hull_norms, k_hull_norms_vars, k_hull_props,
              q_and_k, q_not_k, k_not_q, neither, 
              q_hull_counts, k_hull_counts]:
        for key, array in d.items():
            statistics_dict[key] = array
    stats_df = pd.DataFrame(statistics_dict)
    stats_df['token_id'] = np.arange(16000)
    statistics_path = checkpoint_dir / f'hull-stats.csv'
    stats_df.to_csv(statistics_path, index=False)
    print("Done")

#!SECTION