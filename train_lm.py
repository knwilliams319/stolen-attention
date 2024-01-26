# SECTION: Necessary imports
import torch
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.profilers import AdvancedProfiler
from pathlib import Path
from sentencepiece import SentencePieceProcessor
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
        return self.data.size(0)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        labels = tokens[1:]
        labels = torch.cat([  # insert random token for last label
            tokens[1:], 
            torch.randint(0, self.vocab_size, (1,), dtype=labels.dtype)
        ])

        padding_mask = torch.zeros(self.context_length)
        if idx < len(self.data) - 1:
            padding_mask += float('-inf') * self.data[idx].eq(self.pad_id)
            padding_mask = torch.nan_to_num(padding_mask)

        return tokens, labels, padding_mask

class Wikitext103Model(CausalTransformer):
    def _calculate_loss(self, batch, mode):
        data, labels, mask = batch
        data = data.int() # F.one_hot(data.long(), self.hparams.num_classes).to(dtype=self.dtype)
        preds = self.forward(data, pad_mask=mask) # shape = [bsz, context_len, vocab_size]
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1).long())
        self.log("%s_loss" % mode, loss, sync_dist=True)  # TODO: a warning says we might want to use this, but it could cause communication overhead
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
#!SECTION
  
# SECTION: Training parameters
# TODO: make these CLI arguments instead of constants 
CHECKPOINT_PATH = "./"
TRAIN_PATH = "./data/wikitext-103/wiki.train.tokens.tokenized.pt"
VALID_PATH = "./data/wikitext-103/wiki.valid.tokens.tokenized.pt"
TEST_PATH = "./data/wikitext-103/wiki.train.tokens.tokenized.pt"
TOKENIZER_PATH = "./bpe-tokenizer/tokenizer.model"
# TOKENIZER_VOCAB = "./data/wikitext-103/tokenizer-vocab.json"
# TOKENIZER_MERGES = "./data/wikitext-103/tokenizer-merges.txt"
#!SECTION
        
# SECTION: Training loop
if __name__ == "__main__":
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = Path(CHECKPOINT_PATH)
    root_dir.mkdir(exist_ok=True)
    trainer = L.Trainer(
        deterministic=False, 
        default_root_dir=root_dir,
        enable_progress_bar=True,
        callbacks=[ModelSummary(),
                   ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                   LearningRateMonitor(logging_interval='step')],
        accelerator="gpu",
        devices=3,
        strategy=DDPStrategy(static_graph=True),
        precision="16-mixed",
        max_epochs=100,
        gradient_clip_val=0.1,
        profiler=None # AdvancedProfiler(dirpath='./', filename='profile.log')
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=TOKENIZER_PATH)
    # tokenizer = GPT2TokenizerFast(vocab_file=TOKENIZER_VOCAB, merges_file=TOKENIZER_MERGES)

    # Create dataloaders
    train_dataset = Wikitext103Dataset(TRAIN_PATH, tokenizer.pad_id(), len(tokenizer))
    val_dataset = Wikitext103Dataset(VALID_PATH, tokenizer.pad_id(), len(tokenizer))
    test_dataset = Wikitext103Dataset(TEST_PATH, tokenizer.pad_id(), len(tokenizer))

    train_loader = data.DataLoader(
        train_dataset, batch_size=67, shuffle=True, drop_last=True, num_workers=4, pin_memory=True # TODO: Crank up the batch size
    )
    val_loader = data.DataLoader(val_dataset, batch_size=67, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=67, shuffle=False, drop_last=False, num_workers=4)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = Path(CHECKPOINT_PATH, "Wikitext103Model.ckpt")
    if pretrained_filename.exists() and pretrained_filename.is_file():
        print("Found pretrained model, loading...")
        model = Wikitext103Model.load_from_checkpoint(pretrained_filename)
    else:
        model = Wikitext103Model(
            num_classes=len(tokenizer),
            max_context_len=1024,
            model_dim=128,
            use_euclidean_attention=False,
            learn_temperatures=False,
            positional_temperatures=False,
            num_heads=8,
            num_layers=16,
            dropout=0.3,
            attn_dropout=0.1,
            activation_dropout=0.1,
            ffn_dim=4096,
            use_pos_encoding=True,
            warmup_updates=16000,
            lr_period_updates=270000,
            t_mult=2,
            warmup_end_lr=1.0,
            warmup_init_lr=1e-07,
            min_lr=0.0001,
        )
        trainer.fit(model, train_loader, val_loader)
        # model = Wikitext103Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    # train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    # val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # result = {
    #     "test_acc": test_result[0]["test_acc"],
    #     "val_acc": val_result[0]["test_acc"],
    #     "train_acc": train_result[0]["test_acc"],
    # }
#!SECTION