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
# from transformers import GPT2TokenizerFast

from modules import CausalTransformer
#!SECTION

# SECTION: Dataloaders and LightningModules
class OpenbookQADataset(data.Dataset):
    def __init__(self, data_dir: str, dataset: str, sliding=False):
        super().__init__()
        data_dir = Path(data_dir)
        questions_path = data_dir / f'{dataset}-questions.pt'
        answers_path = data_dir / f'{dataset}-answers.pt'
        self.questions = torch.load(questions_path)
        self.answers = torch.load(answers_path)
        self.sliding = sliding

    @property
    def context_length(self):
        return self.questions.size(1)

    def __len__(self):
        return self.questions.size(0)

    def __getitem__(self, idx):
        # TODO: should I include a padding mask? the model might just learn to ignore padded tokens anyways
        question = self.questions[idx]
        padding_mask = torch.zeros(self.context_length)
        if self.sliding:
            label = self.answers[idx]
        else:
            label = torch.cat([question[1:], self.answers[idx]])
        return question, label, padding_mask
# !SECTION

# SECTION: Model Definition
class OpenbookQAModel(CausalTransformer):
    def _calculate_loss(self, batch, sliding=False):
        data, labels, mask = batch
        data = data.int()
        preds = self.forward(data, pad_mask=mask) # shape = [bsz, context_len, vocab_size]
        if sliding:
            preds = preds[:, -1]
            labels = labels[:, -1].long()
            return F.cross_entropy(preds, labels)
        else:
            return F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1).long())

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, sliding=True)
        self.log(
            "train_loss",
            loss, 
            sync_dist=True,       # this doesn't seem to impact training time, likely because we have only 3 devices
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
CHECKPOINT_BASE = "./experiments/1_layer_8_heads"
EXPERIMENT = "oracle"
CHECKPOINT_DIR = CHECKPOINT_BASE + '/' + EXPERIMENT
DATASET_BASE = "./data/openbookqa/processed"
FINETUNE_DIR = CHECKPOINT_DIR + '/' + 'finetune'
TOKENIZER_PATH = "./unigram-tokenizer/tokenizer.model"
#!SECTION
        
# SECTION: Finetuning loop
if __name__ == "__main__":
    # Create checkpoint directory
    finetune_path = Path(FINETUNE_DIR)
    finetune_path.mkdir(parents=True, exist_ok=True)

    # Set up for fine-tuning. Seed random seeds and initialize Trainer. 
    L.seed_everything(7, workers=True)
    trainer = L.Trainer(
        deterministic=True, 
        default_root_dir=FINETUNE_DIR,
        enable_progress_bar=True,
        logger=CSVLogger(
            CHECKPOINT_DIR,
            name='',
            version='finetune'
        ),
        callbacks=[
            ModelSummary(),
            ModelCheckpoint(
                save_weights_only=True, 
                mode="min", 
                monitor="val_loss",
                dirpath=FINETUNE_DIR
            ),
            ModelCheckpoint(
                save_weights_only=False,
                every_n_train_steps=1000,
                dirpath=FINETUNE_DIR,
                filename='last-{epoch:02d}-{step:02d}'
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        accelerator="gpu",
        devices=3,                 # TODO: Change this back to 3 (David was running an experiment on GPU0)
        strategy="ddp",
        precision="16-mixed",      
        max_epochs=3,
        gradient_clip_val=1.0,     
        benchmark=False,           # this can't be used when deterministic=True
        profiler=None,             # AdvancedProfiler(dirpath='./', filename='profile.log'),
        limit_train_batches=None,  # TODO: change this back to None
        limit_val_batches=None,    # TODO: change this back to None
        log_every_n_steps=50       # TODO: change this back to 50
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=TOKENIZER_PATH)

    # Create dataloaders
    train_dataset = OpenbookQADataset(DATASET_BASE, 'train', sliding=True)
    val_dataset = OpenbookQADataset(DATASET_BASE, 'dev', sliding=True)
    #test_dataset = OpenbookQADataset(DATASET_BASE, 'test')

    BATCH_SIZE = 16
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=3, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=3)
    #test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=3)

    # Load pretrained model. Take the checkpoint with the best validation loss. 
    # NOTE: We override the learning rate and num_steps parameters to fit the fine-tuning workload and avoid overfitting
    checkpoints = Path(CHECKPOINT_DIR).glob("epoch=*.ckpt")
    pretrained_model, *extras = checkpoints
    if extras:
        raise ValueError(f"The directory {CHECKPOINT_DIR} has multiple pretrained model checkpoints inside!")
    
    model = OpenbookQAModel.load_from_checkpoint(
        pretrained_model, 
        lr=1e-2, 
        num_steps=trainer.max_epochs*len(train_loader)/trainer.num_devices,
        dropout=0.8,
        attn_dropout=0.8,
        activation_dropout=0.8
    )
    trainer.fit(model, train_loader, val_loader)
#!SECTION