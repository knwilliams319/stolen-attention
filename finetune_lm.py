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

        # if label.item() == 51:
        #     label = torch.tensor(0)
        # elif label.item() == 124:
        #     label = torch.tensor(1)
        # elif label.item() == 146:
        #     label = torch.tensor(2)
        # else:
        #     label = torch.tensor(3)

        # create mask to ignore padded positions of the sequence (questions are left-padded)
        padding_mask = torch.ones(self.context_length)
        padding_mask[-self.question_lengths[idx]:] = 0
        padding_mask *= float('-inf')
        padding_mask = torch.nan_to_num(padding_mask)  # NOTE: these are being ignored right now! check transformer.py!
        return question, label, padding_mask
# !SECTION

# SECTION: Model Definition
class OpenbookQAModel(FinetuneHead):
    def __init__(self, pretrained_path, num_classes, num_steps, lr=1e-3, dropout=0.0, attn_dropout=0.0, activation_dropout=0.0):
        super().__init__(pretrained_path, num_classes, num_steps, lr=lr, dropout=dropout, attn_dropout=attn_dropout, activation_dropout=activation_dropout)

    # def _calculate_loss(self, batch):
    #     data, labels, mask = batch
    #     data = data.int()
    #     preds = self.forward(data, pad_mask=mask) # shape = [bsz, context_len, vocab_size]
        # if sliding:
        #     preds = preds[:, -1]
        #     labels = labels[:, -1].long()
        #     loss = F.cross_entropy(preds, labels)

        #     answers = []
        #     prob_A = preds[:, 51]
        #     prob_B = preds[:, 124]
        #     prob_C = preds[:, 146]
        #     prob_D = preds[:, 163]
        #     for A, B, C, D in zip(prob_A, prob_B, prob_C, prob_D):
        #         largest = torch.argmax(torch.tensor([A, B, C, D]))
        #         if largest == 0:
        #             answers.append(51)
        #         elif largest == 1:
        #             answers.append(124)
        #         elif largest == 2:
        #             answers.append(146)
        #         else:
        #             answers.append(163)
        #     num_correct = 0
        #     for answer, actual in zip(answers, labels):
        #         if answer == actual.item():
        #             num_correct += 1
        #     return loss, num_correct
        # else:
        #     return F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1).long())
    
    def _calculate_loss(self, batch):
        data, labels, mask = batch
        data = data.int()
        preds = self.forward(data, pad_mask=mask) # shape = [bsz, context_len, vocab_size]
        labels = labels.long().squeeze(1)
        # loss = F.cross_entropy(preds, labels)

        # Try David's custom-built F.cross_entropy that's only over the answer space despite technically allowing the model to output any token
        batch_size, vocab_size = preds.shape
        denominators = torch.exp(preds[:, 51]) + torch.exp(preds[:, 124]) + torch.exp(preds[:, 146]) + torch.exp(preds[:, 163])
        numerators = torch.exp(preds[torch.arange(batch_size), labels])  # exponentiate score assigned to correct answer choice for each quesiton
        losses = -torch.log(numerators / denominators)
        loss = losses.sum() / batch_size

        # Also find the number of correct answers. The model's chosen answer is the one with the highest logit. 
        answer_mask = torch.ones(vocab_size, dtype=torch.bool, device=preds.device)
        answer_mask[[51, 124, 146, 163]] = False
        preds = preds.masked_fill(answer_mask, -float('inf'))
        choices = preds.argmax(dim=1)
        num_correct = torch.sum(choices == labels)
        return loss, num_correct

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch)
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
        for _, p in self.named_parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        total_norm = total_norm ** (0.5)
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
        loss, num_correct = self._calculate_loss(batch)
        self.log(
            "val_loss",
            loss, 
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False
        )
        self.log(
            "val_accuracy",
            num_correct,         # NOTE: 100% accuracy is 31*16 + 4 = 500
            on_step=False,
            on_epoch=True,
            reduce_fx="sum"
        )

    def test_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch)
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
CHECKPOINT_BASE = "./experiments/12_layers_12_heads"
EXPERIMENT = "base"
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

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=TOKENIZER_PATH)

    # Create dataloaders
    train_dataset = OpenbookQADataset(DATASET_BASE, 'train', difficulty='easy')
    val_dataset = OpenbookQADataset(DATASET_BASE, 'dev', difficulty='easy')
    #test_dataset = OpenbookQADataset(DATASET_BASE, 'test')

    BATCH_SIZE = 8
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=3, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=3)
    #test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=3)

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
                mode="max", 
                monitor="val_accuracy",
                dirpath=FINETUNE_DIR
            ),
            ModelCheckpoint(
                save_weights_only=False,
                every_n_train_steps=len(train_loader),     # Save state of model at the end of every epoch
                dirpath=FINETUNE_DIR,
                filename='last-{epoch:02d}-{step:02d}'
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        accelerator="gpu",
        devices=1,                 # TODO: Change this back to 3 (David was running an experiment on GPU0)
        strategy="ddp",            # NOTE: use "ddp_find_unused_parameters_true" if I'm freezing any params
        precision="16-mixed",
        max_epochs=5,
        gradient_clip_val=1.0,
        benchmark=False,           # this can't be used when deterministic=True
        profiler=None,             # AdvancedProfiler(dirpath='./', filename='profile.log'),
        limit_train_batches=None,  # TODO: change this back to None
        limit_val_batches=None,    # TODO: change this back to None
        log_every_n_steps=50       # TODO: change this back to 50
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Find the pretrained checkpoint with the best validation loss. 
    checkpoints = Path(CHECKPOINT_DIR).glob("epoch=*.ckpt")
    pretrained_model, *extras = checkpoints
    if extras:
        raise ValueError(f"The directory {CHECKPOINT_DIR} has multiple pretrained model checkpoints inside!")
    
    # Create Finetuned Model
    # NOTE: My OpenbookQAModel is currently ignoring the dropout arguments!
    model = OpenbookQAModel(
        pretrained_model,
        16000,
        trainer.max_epochs*len(train_loader)/trainer.num_devices,
        lr=1e-4,
        dropout=0.10,
        attn_dropout=0.10,
        activation_dropout=0.10
    )
    trainer.fit(model, train_loader, val_loader)
#!SECTION