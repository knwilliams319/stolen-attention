# SECTION: Necessary imports
import torch
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from lightning.pytorch.loggers import CSVLogger
from modules.transformer import FinetuneHead
from pandas import read_csv
#!SECTION

# SECTION: Dataloaders and LightningModules
class OpenbookQADataset(data.Dataset):
    def __init__(self, data_dir: str, dataset: str, difficulty: str):
        super().__init__()
        data_dir = Path(data_dir)
        questions_path = data_dir / f'{dataset}-{difficulty}-questions.pt'
        answers_path = data_dir / f'{dataset}-{difficulty}-answers.pt'
        support_path = data_dir / f'{dataset}-{difficulty}-support.csv'
        self.questions = torch.load(questions_path)
        self.answers = torch.load(answers_path)
        self.question_lengths = read_csv(support_path, sep=';')['length']

    @property
    def context_length(self):
        return self.questions.size(1)

    def __len__(self):
        return self.questions.size(0)

    def __getitem__(self, idx):
        # grab question and correct answer choice
        question = self.questions[idx]
        label = self.answers[idx]

        # create mask to ignore padded positions of the sequence (questions are left-padded)
        padding_mask = torch.ones(self.context_length)
        padding_mask[-self.question_lengths[idx]:] = 0
        padding_mask *= float('-inf')
        padding_mask = torch.nan_to_num(padding_mask)
        return question, label, padding_mask
# !SECTION

# SECTION: Model Definition
class OpenbookQAModel(FinetuneHead):
    def __init__(self, pretrained_path, num_classes, num_steps, lr=1e-3, dropout=0.0, attn_dropout=0.0, activation_dropout=0.0):
        super().__init__(pretrained_path, num_classes, num_steps, lr=lr, dropout=dropout, attn_dropout=attn_dropout, activation_dropout=activation_dropout)

    def _calculate_loss(self, batch):
        data, labels, mask = batch
        data = data.int()
        preds = self.forward(data, pad_mask=mask) # shape = [bsz, context_len, vocab_size]
        return F.cross_entropy(preds, labels.long())

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
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
        for _, p in self.named_parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
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
        devices=1,                 # TODO: Change this back to 3 (David was running an experiment on GPU0)
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",      
        max_epochs=3,
        gradient_clip_val=5.0,     
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
    train_dataset = OpenbookQADataset(DATASET_BASE, 'train', difficulty='easy')
    val_dataset = OpenbookQADataset(DATASET_BASE, 'dev', difficulty='easy')
    #test_dataset = OpenbookQADataset(DATASET_BASE, 'test')

    BATCH_SIZE = 16
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=3, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=3)
    #test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=3)

    # Find the pretrained checkpoint with the best validation loss. 
    checkpoints = Path(CHECKPOINT_DIR).glob("epoch=*.ckpt")
    pretrained_model, *extras = checkpoints
    if extras:
        raise ValueError(f"The directory {CHECKPOINT_DIR} has multiple pretrained model checkpoints inside!")
    
    # Create Finetuned Model
    model = OpenbookQAModel(
        pretrained_model,
        4,
        trainer.max_epochs*len(train_loader)/trainer.num_devices,
        lr=1e-3,
        dropout=0.0,
        attn_dropout=0.0,
        activation_dropout=0.0
    )
    trainer.fit(model, train_loader, val_loader)
#!SECTION