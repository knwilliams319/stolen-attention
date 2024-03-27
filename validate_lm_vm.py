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
    def __init__(self, tokens_path: str, pad_id: int, vocab_size: int, stride: int=1, window_length=None):
        super().__init__()
        # Load packed tokens and store other constructor arguments
        self.data = torch.load(tokens_path)
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.stride = stride

        # If not provided, default window length is packed tokens' original context length
        self.window_length = window_length if window_length else self.data.size(1) 

        # Flatten packed tokens 
        self.data = torch.flatten(self.data)

        # Find last index at which tokens exist, as there may be padding tokens in the last packed batch
        self.num_tokens = 0
        for i in range(self.data.size(0)):
            if self.data[i] == self.pad_id:
                break
        self.num_tokens = i

    def __len__(self):
        num_windows = self.num_tokens - self.window_length
        divisor, remainder = divmod(num_windows, self.stride) 
        if remainder == 0: 
            return divisor
        else: # if remainder is nonzero, // rounds down to ignore an extra batch that's still within range for labels
            return divisor + 1

    def __getitem__(self, idx):
        strided_idx = idx * self.stride
        tokens = self.data[strided_idx:strided_idx+self.window_length]
        padding_mask = torch.zeros(self.window_length)
        labels = self.data[strided_idx+1:strided_idx+self.window_length+1]
        return tokens, labels, padding_mask
# !SECTION

# SECTION: Model definitions
global sliding_mode # if True (default False), only the predictions on the last token of the batch will influence loss

class Wikitext103Model(CausalTransformer):
    def __init__(self, **model_kwargs):
        # Initialize model as per usual, but add extra state to track token statistics
        super().__init__(**model_kwargs)
        self.norms = {}        # KEY: (layer_idx, token_id); VALUE: list of embedding norms for that token in the K matrix of first head of that layer
        self.is_vertex = {}    # KEY: (layer_idx, token_id); VALUE: list of vertex membership for that token in K Hull of the first head of that layer
        self.attn_weights = {} # KEY: (layer_idx, token_id); VALUE: list of attention weights assigned to that token in the first head of that layer
        for t in range(self.hparams.num_classes):
            for l in range(self.hparams.num_layers):
                self.norms[(l, t)] = []
                self.is_vertex[(l, t)] = []
                self.attn_weights[(l, t)] = []

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
            preds = preds[:, -1]
            labels = labels[:, -1].long()

            # Track statistics
            # loss = F.cross_entropy(preds, labels, reduction='none')
            # labels_cpu = labels.cpu().numpy()
            # loss_cpu = loss.detach().cpu()
            # for batch_idx in range(loss_cpu.size(0)):
            #     token_id = labels_cpu[batch_idx]
            #     self.losses[token_id].append(loss_cpu[batch_idx].item())
            # return torch.mean(loss)  # return mean so that logging doesn't error
            def stat_generator(tokens, vertices, embeddings, attn_weights):
                # This generator is the most optimal way to check for vertex membership, as we can
                # avoid using "in vertices" statements that take O(n) time. 
                internal_i = 0
                vertex_i = 0
                while vertex_i < len(vertices):
                    if vertices[vertex_i] == internal_i:
                        yield tokens[internal_i].item(), True, embeddings[internal_i].item(), attn_weights[internal_i].item()
                        internal_i += 1
                        vertex_i += 1
                    else:
                        yield tokens[internal_i].item(), False, embeddings[internal_i].item(), attn_weights[internal_i].item()
                        internal_i += 1

            # Track K Hull statistics
            # NOTE: When calculating K Hull statistics, batch size should be 1
            token_ids = data[0].cpu()
            for l, layer in enumerate(self.transformer.layers):
                if layer.self_attn.k_hull is not None: # will be None for a deficient layer
                    vertices = sorted(layer.self_attn.k_hull.vertices)
                    embedding_norms = torch.norm(layer.self_attn.k_matrix, dim=-1)
                    weights = layer.self_attn.attn_weights
                    for t, is_vertex, norm, weight in stat_generator(token_ids, vertices, embedding_norms, weights):
                        self.norms[(l, t)].append(norm)
                        self.is_vertex[(l, t)].append(is_vertex)
                        self.attn_weights[(l, t)].append(weight)

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
CHECKPOINT_BASE = "./experiments/embed_dim_64/n_heads_8"
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
        devices=[0],                  
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

    WINDOW_LENGTH = 512
    val_dataset_flat = FlattenedWikitext103Dataset(VALID_PATH, tokenizer.pad_id(), len(tokenizer), stride=300, window_length=WINDOW_LENGTH)
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

    k_norm_mean = {}
    k_norm_var = {}
    k_norm_min = {}
    k_norm_max = {}
    k_hull_prop = {}
    weight_mean = {}
    weight_min = {}
    weight_max = {}
    weight_var = {}

    for i in range(model.hparams.num_layers):
        k_norm_mean[f'layer_{i}_norm_mean'] = [pd.NA]*16000
        k_norm_var[f'layer_{i}_norm_var'] = [pd.NA]*16000
        k_norm_min[f'layer_{i}_norm_min'] = [pd.NA]*16000
        k_norm_max[f'layer_{i}_norm_max'] = [pd.NA]*16000
        k_hull_prop[f'layer_{i}_vertex_prop'] = [pd.NA]*16000
        weight_mean[f'layer_{i}_weight_mean'] = [pd.NA]*16000
        weight_min[f'layer_{i}_weight_min'] = [pd.NA]*16000
        weight_max[f'layer_{i}_weight_max'] = [pd.NA]*16000
        weight_var[f'layer_{i}_weight_var'] = [pd.NA]*16000

    for (l, t), k_norms in model.norms.items():
        if len(k_norms) > 0:
            k_norm_mean[f'layer_{l}_norm_mean'][t] = np.mean(k_norms)
            k_norm_var[f'layer_{l}_norm_var'][t] = np.var(k_norms)
            k_norm_min[f'layer_{l}_norm_min'][t] = np.min(k_norms)
            k_norm_max[f'layer_{l}_norm_max'][t] = np.max(k_norms)
    for (l, t), is_vertex in model.is_vertex.items():
        if len(is_vertex) > 0:
            k_hull_prop[f'layer_{l}_vertex_prop'][t] = np.mean(is_vertex)
    for (l, t), weights in model.attn_weights.items():
        if len(weights) > 0:
            weight_mean[f'layer_{l}_weight_mean'][t] = np.mean(weights)
            weight_min[f'layer_{l}_weight_min'][t] = np.min(weights)
            weight_max[f'layer_{l}_weight_max'][t] = np.max(weights)
            weight_var[f'layer_{l}_weight_var'][t] = np.var(weights)
       
    statistics_dict = {}
    for d in [k_norm_mean, k_norm_min, k_norm_max, k_norm_var,
              k_hull_prop,
              weight_mean, weight_min, weight_max, weight_var]:
        for key, array in d.items():
            statistics_dict[key] = array
    stats_df = pd.DataFrame(statistics_dict)
    stats_df['token_id'] = np.arange(16000)
    statistics_path = checkpoint_dir / f'khull-stats-window={WINDOW_LENGTH}.csv'
    stats_df.to_csv(statistics_path, index=False)
    print("Done")

#!SECTION