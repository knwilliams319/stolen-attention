# SECTION: Necessary imports
import torch
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L
from pathlib import Path
from sentencepiece import SentencePieceProcessor
import pandas as pd
import numpy as np
import scipy.spatial as sp
# import pickle
# from sklearn.metrics import confusion_matrix

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
        last_label = self.data[idx+1][0]
        return tokens, last_label # NOTE: due to token packing, pretraining batches will never have padding and we don't need to return a mask
        # labels = torch.cat([
        #     tokens[1:],
        #     torch.tensor([self.data[idx+1][0]], dtype=tokens.dtype)
        # ])
        # return tokens, labels, padding_mask

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
        last_label = self.data[strided_idx+self.window_length]
        return tokens, last_label # NOTE: due to token packing, pretraining batches will never have padding and we don't need to return a mask
        # labels = self.data[strided_idx+1:strided_idx+self.window_length+1]
        # return tokens, labels, padding_mask
# !SECTION

# SECTION: Model definitions

class Wikitext103Model(CausalTransformer):
    def __init__(self, **model_kwargs):
        # Initialize model as per usual, but add extra state to track token statistics
        super().__init__(**model_kwargs)
        # self.all_attn_weights = {}
        # self.all_attn_norms = {}
        # self.all_query_norms = {}
        # #self.batch_no = []
        # self.all_token_ids = []
        
        # for i in range(self.hparams.num_heads):
        #     self.all_attn_norms[i] = []
        #     self.all_attn_weights[i] = []
        #     self.all_query_norms[i] = []

        # self.batch_no = []
        # self.n_vertices = {}
        # self.n_tokens = []
        # self.n_interior = {}
        # self.query_norm = {}
        # self.query_inside = {}
        # self.max_weight_in = {}
        # self.max_norm_in = {}
        # self.max_weight_out = {}
        # self.max_norm_out = {}
        # self.max_inside = {}
        # self.avg_weight_all = {}
        # self.avg_norm_all = {}
        # self.avg_weight_in = {}
        # self.avg_norm_in = {}
        # self.avg_weight_out = {}
        # self.avg_norm_out = {}
        # self.avg_weight_all_top5 = {}
        # self.avg_norm_all_top5 = {}
        # self.avg_weight_in_top5 = {}
        # self.avg_norm_in_top5 = {}
        # self.avg_weight_out_top5 = {}
        # self.avg_norm_out_top5 = {}
        # self.n_inside_top5 = {}

    def _calculate_loss(self, batch, sliding=False):
        data, last_labels = batch
        data = data.int()
        preds = self(data) # shape = [bsz, context_len, vocab_size]
        if sliding:
            preds = preds[:, -1]
            last_labels = last_labels.long()
            return F.cross_entropy(preds, last_labels)
        else:
            labels = torch.cat([data[:, 1:], last_labels.unsqueeze(1)], dim=1)
            return F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1).long())

    # TODO: I want to use this script to test my flow in train_lm.py
    # def _calculate_loss(self, batch):
    #     data, labels, mask, batch_idx = batch
    #     data = data.int()
    #     preds = self.forward(data, pad_mask=mask) # shape = [bsz, context_len, vocab_size]

    #     if not sliding_mode:
    #         # Get predictions over all tokens in all batches
    #         raise ValueError('sliding_mode is set to False!')
    #         # preds = preds.view(-1, preds.size(-1))
    #         # labels = labels.view(-1).long()
    #         # loss = F.cross_entropy(preds, labels)  
    #         # return loss
    #     else:
    #         # Grab predictions on the last element of the context for all batches
    #         # NOTE: Using -1 will get predictions on the 513th element given the previous 512, which is what we 
    #         #       generally want to measure with sliding window inference.
    #         preds = preds[:, -1]
    #         labels = labels[:, -1].long()

    #         # Track statistics (deprecated)
    #         # loss = F.cross_entropy(preds, labels, reduction='none')
    #         # labels_cpu = labels.cpu().numpy()
    #         # loss_cpu = loss.detach().cpu()
    #         # for batch_idx in range(loss_cpu.size(0)):
    #         #     token_id = labels_cpu[batch_idx]
    #         #     self.losses[token_id].append(loss_cpu[batch_idx].item())
    #         # return torch.mean(loss)  # return mean so that logging doesn't error

    #         # Capture statistics that are layer-agnostic
    #         # token_ids = data[0].cpu().numpy()
    #         #self.batch_no.append(batch_idx.item())
    #         # self.all_token_ids.append(token_ids)
    #         # self.n_tokens.append(len(token_ids))

    #         # for l, layer in enumerate(self.transformer.layers):
    #         #     # self.all_attn_weights.append(layer.self_attn.attn_weights.numpy())
                
    #         #     for h in range(self.hparams.num_heads):
    #         #         # vertices = sorted(layer.self_attn.k_hull[h].vertices)
    #         #         k_norms = torch.norm(layer.self_attn.k_matrix[h], dim=-1).numpy()
    #         #         weights = layer.self_attn.attn_weights[h].numpy()
    #         #         query_norm = torch.norm(layer.self_attn.query_point[h], dim=-1).numpy()
    #         #         # k_hull = layer.self_attn.k_hull[h]

    #         #         # Append statistics to internal state
    #         #         self.all_query_norms[h].append(query_norm)
    #         #         self.all_attn_norms[h].append(k_norms)
    #         #         self.all_attn_weights[h].append(weights)

    #                 # # Initialize list of statistics for this head
    #                 # if h not in self.n_vertices:
    #                 #     self.n_vertices[h] = []
    #                 #     self.n_interior[h] = []
    #                 #     self.query_norm[h] = []
    #                 #     self.query_inside[h] = []
    #                 #     self.max_weight_in[h] = []
    #                 #     self.max_norm_in[h] = []
    #                 #     self.max_weight_out[h] = []
    #                 #     self.max_norm_out[h] = []
    #                 #     self.max_inside[h] = []
    #                 #     self.avg_weight_all[h] = []
    #                 #     self.avg_norm_all[h] = []
    #                 #     self.avg_weight_in[h] = []
    #                 #     self.avg_norm_in[h] = []
    #                 #     self.avg_weight_out[h] = []
    #                 #     self.avg_norm_out[h] = []
    #                 #     self.avg_weight_all_top5[h] = []
    #                 #     self.avg_norm_all_top5[h] = []
    #                 #     self.avg_weight_in_top5[h] = []
    #                 #     self.avg_norm_in_top5[h] = []
    #                 #     self.avg_weight_out_top5[h] = []
    #                 #     self.avg_norm_out_top5[h] = []
    #                 #     self.n_inside_top5[h] = []

    #                 # # Capture statistics for this head
    #                 # self.n_vertices[h].append(len(vertices))
    #                 # self.n_interior[h].append(len(token_ids) - len(vertices))

    #                 # vertex_weights = weights[vertices]
    #                 # vertex_norms = k_norms[vertices]
    #                 # interior_weights = [weight for i, weight in enumerate(weights) if i not in vertices]
    #                 # interior_norms = [norm for i, norm in enumerate(k_norms) if i not in vertices]

    #                 # self.avg_weight_all[h].append(torch.mean(weights).item())
    #                 # self.avg_norm_all[h].append(torch.mean(k_norms).item())
    #                 # indexed_weights = list(enumerate(weights))
    #                 # sorted_weights = sorted(indexed_weights, key=lambda x: x[1], reverse=True)
    #                 # self.avg_weight_all_top5[h].append(np.mean([weight for _, weight in sorted_weights[0:5]]))
    #                 # self.avg_norm_all_top5[h].append(np.mean([k_norms[idx] for idx, _ in sorted_weights[0:5]]))
    #                 # top_inside = [not idx in vertices for idx, _ in sorted_weights[0:5]]
    #                 # self.n_inside_top5[h].append(sum(top_inside))

    #                 # self.avg_weight_out[h].append(torch.mean(vertex_weights).item())
    #                 # self.avg_norm_out[h].append(torch.mean(vertex_norms).item())
    #                 # indexed_weights_out = list(enumerate(vertex_weights))
    #                 # sorted_weights_out = sorted(indexed_weights_out, key=lambda x: x[1], reverse=True)
    #                 # self.avg_weight_out_top5[h].append(np.mean([weight for _, weight in sorted_weights_out[0:5]]))
    #                 # self.avg_norm_out_top5[h].append(np.mean([vertex_norms[idx] for idx, _ in sorted_weights_out[0:5]]))
    #                 # self.max_weight_out[h].append(sorted_weights_out[0][1].item())
    #                 # self.max_norm_out[h].append(vertex_norms[sorted_weights_out[0][0]].item())

    #                 # # sometimes, all keys are vertices of the convex hull
    #                 # if len(interior_weights) > 0:
    #                 #     self.avg_weight_in[h].append(np.mean(interior_weights))
    #                 #     self.avg_norm_in[h].append(np.mean(interior_norms))
    #                 #     indexed_weights_in = list(enumerate(interior_weights))
    #                 #     sorted_weights_in = sorted(indexed_weights_in, key=lambda x: x[1], reverse=True)
    #                 #     self.avg_weight_in_top5[h].append(np.mean([weight for _, weight in sorted_weights_in[0:5]]))
    #                 #     self.avg_norm_in_top5[h].append(np.mean([interior_norms[idx] for idx, _ in sorted_weights_in[0:5]]))
    #                 #     self.max_weight_in[h].append(sorted_weights_in[0][1].item())
    #                 #     self.max_norm_in[h].append(interior_norms[sorted_weights_in[0][0]].item())
    #                 #     self.max_inside[h].append(sorted_weights_out[0][1].item() < sorted_weights_in[0][1].item())
    #                 # else:
    #                 #     self.avg_weight_in[h].append(pd.NA)
    #                 #     self.avg_norm_in[h].append(pd.NA)
    #                 #     self.avg_weight_in_top5[h].append(pd.NA)
    #                 #     self.avg_norm_in_top5[h].append(pd.NA)
    #                 #     self.max_weight_in[h].append(pd.NA)
    #                 #     self.max_norm_in[h].append(pd.NA)
    #                 #     self.max_inside[h].append(False)

    #                 # # This must be last, as doing add_points may change the vertex list
    #                 # try:
    #                 #     k_hull.add_points(query.unsqueeze(0))
    #                 #     self.query_inside[h].append(not len(token_ids) in k_hull.vertices)
    #                 # except sp.QhullError:
    #                 #     self.query_inside[h].append(pd.NA)
    #                 # self.query_norm[h].append(torch.norm(query).item())
                    
    #         # we're doing nothing special with loss for Q/K stats
    #         return F.cross_entropy(preds, labels) 

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, sliding=True)
        self.log(
            "val_loss",
            loss, 
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False
        )

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, sliding=True)
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
CHECKPOINT_BASE = "./experiments/embed_dim_512/64_heads"
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
    BATCH_SIZE = 64
    val_dataset = Wikitext103Dataset(VALID_PATH, tokenizer.pad_id(), len(tokenizer))
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4, persistent_workers=False)

    WINDOW_LENGTH = 512
    STRIDE = 16
    val_dataset_flat = FlattenedWikitext103Dataset(VALID_PATH, tokenizer.pad_id(), len(tokenizer), stride=STRIDE, window_length=WINDOW_LENGTH)
    val_loader_flat = data.DataLoader(val_dataset_flat, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=3, persistent_workers=True)

    # Load pretrained model
    checkpoint_dir = Path(CHECKPOINT_DIR)
    pretrained_file_path = list(checkpoint_dir.glob('backup-state*.ckpt')) # Should grab the best checkpoint
    # pretrained_file_path = list(checkpoint_dir.glob('best-weights-epoch=*.ckpt'))
    pretrained_file_path, *extras = pretrained_file_path
    if extras:
        raise ValueError('Too many checkpoints were globbed in this directory!')
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
    trainer.validate(model, dataloaders=val_loader_flat, verbose=True)

    # for i in range(8):
    #     weights = np.array(model.all_attn_weights[i])
    #     weights_path = checkpoint_dir / f'attn_weights-wdw={WINDOW_LENGTH}-stride={STRIDE}-head={i}.npy'
    #     np.save(weights_path, weights)

    #     norms = np.array(model.all_attn_norms[i])
    #     norms_path = checkpoint_dir / f'attn_norms-wdw={WINDOW_LENGTH}-stride={STRIDE}-head={i}.npy'
    #     np.save(norms_path, norms)

    #     q_norms = np.array(model.all_query_norms[i])
    #     q_norms_path = checkpoint_dir / f'attn_query_norms-wdw={WINDOW_LENGTH}-stride={STRIDE}-head={i}.npy'
    #     np.save(q_norms_path, q_norms)
    
    # token_ids = np.array(model.all_token_ids)
    # token_ids_path = checkpoint_dir / f'token_ids-wdw={WINDOW_LENGTH}-stride={STRIDE}.npy'
    # np.save(token_ids_path, token_ids)

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

    # k_norm_mean = {}
    # k_norm_var = {}
    # k_norm_min = {}
    # k_norm_max = {}
    # k_hull_prop = {}
    # weight_mean = {}
    # weight_min = {}
    # weight_max = {}
    # weight_var = {}

    # for i in range(model.hparams.num_layers):
    #     k_norm_mean[f'layer_{i}_norm_mean'] = [pd.NA]*16000
    #     k_norm_var[f'layer_{i}_norm_var'] = [pd.NA]*16000
    #     k_norm_min[f'layer_{i}_norm_min'] = [pd.NA]*16000
    #     k_norm_max[f'layer_{i}_norm_max'] = [pd.NA]*16000
    #     k_hull_prop[f'layer_{i}_vertex_prop'] = [pd.NA]*16000
    #     weight_mean[f'layer_{i}_weight_mean'] = [pd.NA]*16000
    #     weight_min[f'layer_{i}_weight_min'] = [pd.NA]*16000
    #     weight_max[f'layer_{i}_weight_max'] = [pd.NA]*16000
    #     weight_var[f'layer_{i}_weight_var'] = [pd.NA]*16000

    # for (l, t), k_norms in model.norms.items():
    #     if len(k_norms) > 0:
    #         k_norm_mean[f'layer_{l}_norm_mean'][t] = np.mean(k_norms)
    #         k_norm_var[f'layer_{l}_norm_var'][t] = np.var(k_norms)
    #         k_norm_min[f'layer_{l}_norm_min'][t] = np.min(k_norms)
    #         k_norm_max[f'layer_{l}_norm_max'][t] = np.max(k_norms)
    # for (l, t), is_vertex in model.is_vertex.items():
    #     if len(is_vertex) > 0:
    #         k_hull_prop[f'layer_{l}_vertex_prop'][t] = np.mean(is_vertex)
    # for (l, t), weights in model.attn_weights.items():
    #     if len(weights) > 0:
    #         weight_mean[f'layer_{l}_weight_mean'][t] = np.mean(weights)
    #         weight_min[f'layer_{l}_weight_min'][t] = np.min(weights)
    #         weight_max[f'layer_{l}_weight_max'][t] = np.max(weights)
    #         weight_var[f'layer_{l}_weight_var'][t] = np.var(weights)
       
    # statistics_dict = {}
    # for d in [k_norm_mean, k_norm_min, k_norm_max, k_norm_var,
    #           k_hull_prop,
    #           weight_mean, weight_min, weight_max, weight_var]:
    #     for key, array in d.items():
    #         statistics_dict[key] = array
    # stats_df = pd.DataFrame(statistics_dict)
    # stats_df['token_id'] = np.arange(16000)
    # statistics_path = checkpoint_dir / f'khull-stats-window={WINDOW_LENGTH}.csv'
    # stats_df.to_csv(statistics_path, index=False)

    # try:
    #     attn_type = model.hparams.attention_norm
    # except KeyError:
    #     if model.hparams.use_euclidean_attention:
    #         attn_type = 2
    #     else:
    #         attn_type = 0

    # for head in model.n_vertices:
    #     stats_df = pd.DataFrame({
    #         'batch_no': model.batch_no,
    #         'attn_norm': str(attn_type),
    #         'n_heads': str(model.hparams.num_heads),
    #         'head_id': str(head),
    #         'n_vertices': model.n_vertices[head],
    #         'n_interior': model.n_interior[head],
    #         'query_norm': model.query_norm[head],
    #         'query_inside': model.query_inside[head],
    #         'max_weight_in': model.max_weight_in[head],
    #         'max_norm_in': model.max_norm_in[head],
    #         'max_weight_out': model.max_weight_out[head],
    #         'max_norm_out': model.max_norm_out[head],
    #         'max_inside': model.max_inside[head],
    #         'avg_weight_all': model.avg_weight_all[head],
    #         'avg_norm_all': model.avg_norm_all[head],
    #         'avg_weight_in': model.avg_weight_in[head],
    #         'avg_norm_in': model.avg_norm_in[head],
    #         'avg_weight_out': model.avg_weight_out[head],
    #         'avg_norm_out': model.avg_norm_out[head],
    #         'avg_weight_all_top5': model.avg_weight_all_top5[head],
    #         'avg_norm_all_top5': model.avg_norm_all_top5[head],
    #         'avg_weight_in_top5': model.avg_weight_in_top5[head],
    #         'avg_norm_in_top5': model.avg_norm_in_top5[head],
    #         'avg_weight_out_top5': model.avg_weight_out_top5[head],
    #         'avg_norm_out_top5': model.avg_norm_out_top5[head],
    #         'n_inside_top5': model.n_inside_top5[head]
    #     })
    #     statistics_path = checkpoint_dir / f'khull-batch-stats-head{head}.csv'
    #     stats_df.to_csv(statistics_path, index=False)
    
    print("Done")

#!SECTION