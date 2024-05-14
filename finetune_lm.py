# SECTION: Necessary imports
import argparse
from pathlib import Path

import torch
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from sentencepiece import SentencePieceProcessor
from lightning.pytorch.loggers import CSVLogger

from modules import CausalTransformer, REXScheduler
#!SECTION

# SECTION: Datasets
class OpenbookQADataset(data.Dataset):
    def __init__(self, dataset_path: Path, tokenizer):
        super().__init__()
        # Read raw text data and separate the important blocks of the prompt
        self.data = []
        with dataset_path.open('rt') as f:
            for line in f:
                line = line.replace('\n','')
                tokens = line.split('|')
                d = {}
                d['fact'] = tokens[0]
                d['stem'] = tokens[1]
                d['A'] = tokens[2]
                d['B'] = tokens[3]
                d['C'] = tokens[4]
                d['D'] = tokens[5]
                d['Answer'] = tokens[6]   
                self.data.append(d)
        
        # Make a prompt and tokenize the examples 
        self.lengths = []
        for i, example in enumerate(self.data):
            text = 'Fact: %s Stem: %s A: %s B: %s C: %s D: %s Answer: %s' % (example['fact'],
                                                                             example['stem'],
                                                                             example['A'],
                                                                             example['B'],
                                                                             example['C'],
                                                                             example['D'],
                                                                             example['Answer'])
            tokens = tokenizer.encode(text)
            self.data[i] = tokens
            self.lengths.append(len(tokens))
        
        # Left-pad the questions to the same length
        # NOTE: Uncomment this block if using a batch size that is greater than 1
        def pad_to_longest_question(questions, lengths):
            max_length = max(self.lengths)
            for i, question in enumerate(questions):
                padding = [0] * (max_length - lengths[i])  # NOTE: I can't see where David uses padding... could this be a cause for concern?
                yield padding + question
        self.data = torch.tensor(list(pad_to_longest_question(self.data, self.lengths)), dtype=torch.int32)

    @property
    def context_length(self):
        return len(self.data[0]) - 1 # exclude answer choice token from length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab question and padding mask (correct answer choice is last token)
        tokens = self.data[idx][:-1]
        answer = self.data[idx][-1]

        # create mask to ignore padded positions of the sequence (questions are left-padded)
        padding_mask = torch.zeros(self.context_length)
        padding_mask[-self.lengths[idx]:] = 1
        return tokens, answer, padding_mask
# !SECTION

# SECTION: Model Definition
# TODO: Try to use the FinetuneHead to train a new classification head on top of the model after I get this working
class OpenbookQAModel(CausalTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        # So far, Adam with no scheduler has performed better than using a scheduler or another optimizer + a scheduler. 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9)
        # scheduler = REXScheduler(optimizer, num_steps=self.hparams.num_steps)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "frequency": 1
        #     }
        # }
        return optimizer
    
    def _calculate_loss(self, batch):
        data, labels, mask = batch  # TODO: apply the padding mask for each question?
        data = data
        preds = self(data, pad_mask=mask)[:, -1] # only take last hidden state
        labels = labels.long()

        # Try David's custom-built F.cross_entropy that's only over the answer space despite technically allowing the model to output any token
        # NOTE: My UnigramTokenizer has this mapping: A-->51, B-->124, C-->146, D-->163
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
            on_step=True,
            on_epoch=True,
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
            on_step=True,
            on_epoch=False,
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
            num_correct.float(),  # NOTE: 100% accuracy is 500 (and without the float conversion, I get a warning)
            on_step=False,
            on_epoch=True,
            reduce_fx="sum"
        )

def get_model(opt):
    assert opt.pretrained_path, f"The options object has no stored pretrained_path!"
    model = OpenbookQAModel.load_from_checkpoint(
        opt.pretrained_path, 
        lr=opt.lr,
        num_steps=opt.num_steps
    )
    return model    
#!SECTION
  
# SECTION: Paths and constants used for default arguments below
base = Path(__file__).parent
EXPERIMENT_DIR = base / 'experiments/embed_dim_64/n_heads_8/base'
TRAIN_PATH = base / 'floyd-finetune/data/obqa.train.txt'
VAL_PATH = base / 'floyd-finetune/data/obqa.valid.txt'
TOKENIZER_PATH = "./unigram-tokenizer/tokenizer.model"
#!SECTION
        
# SECTION: Finetuning loop
if __name__ == "__main__":
    # Parse CLI Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-clip_norm', type=float, default=2.0)
    parser.add_argument('-batchsize', type=int, default=3)
    parser.add_argument('-log_every', type=int, default=100)
    parser.add_argument('-train_path', type=Path, default=TRAIN_PATH)
    parser.add_argument('-val_path', type=Path, default=VAL_PATH)
    parser.add_argument('-tokenizer_path', type=str, default=TOKENIZER_PATH)
    parser.add_argument('-pretrained_dir', type=Path, default=EXPERIMENT_DIR)
    parser.add_argument('-save_dir', type=str, default='finetune')
    # TODO: add arguments to apply different dropout rates, e.g. parser.add_argument('-attn_dropout', type=int, default=0.1)
    opt = parser.parse_args()

    # Post-process some of the CLI arguments
    pretrained_paths = opt.pretrained_dir.glob('epoch=*.ckpt')  # the best pretrained checkpoint will follow this naming pattern
    opt.pretrained_path, *extras = pretrained_paths
    if extras:
        raise ValueError('The passed-in pretrained_dir argument holds more than one pretrained model checkpoint inside!')
    opt.device = "cpu" if opt.no_cuda else "gpu"
    if opt.device == "gpu":
        assert torch.cuda.is_available()
    opt.save_dir = opt.pretrained_dir / opt.save_dir

    # Initialize tokenizer
    tokenizer = SentencePieceProcessor(model_file=opt.tokenizer_path)

    # Create DataLoaders
    train_dataset = OpenbookQADataset(opt.train_path, tokenizer)
    val_dataset = OpenbookQADataset(opt.val_path, tokenizer)
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=3)
    val_loader = data.DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=3)

    # Seed everything and initialize the Trainer
    L.seed_everything(10, workers=True)
    trainer = L.Trainer(
        deterministic=True,
        default_root_dir=opt.save_dir,
        enable_progress_bar=True,
        logger=CSVLogger(
            opt.pretrained_dir,
            name='',
            version=opt.save_dir.name
        ),
        callbacks=[
            ModelSummary(),
            ModelCheckpoint(
                save_weights_only=True, 
                mode="max", 
                monitor="val_accuracy",
                dirpath=opt.save_dir
            ),
            ModelCheckpoint(
                save_weights_only=False,
                every_n_train_steps=len(train_loader),     # Save state of model at the end of every epoch
                dirpath=opt.save_dir,
                filename='last-{epoch:02d}-{step:02d}'
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        accelerator=opt.device,
        devices=1,
        strategy="ddp",
        precision="16-mixed",
        max_epochs=opt.epochs,
        gradient_clip_val=opt.clip_norm,
        benchmark=False,                       # this can't be used when deterministic=True
        profiler=None,                         # AdvancedProfiler(dirpath='./', filename='profile.log'),
        limit_train_batches=None,              # TODO: change this back to None
        limit_val_batches=None,                # TODO: change this back to None
        log_every_n_steps=opt.log_every        # TODO: change this back to 50
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Instantiate the pretrained model, create checkpoint directory, and finetune it
    opt.num_steps = trainer.max_epochs*len(train_loader)/trainer.num_devices
    model = get_model(opt)
    opt.save_dir.mkdir(parents=True, exist_ok=True)    
    trainer.fit(model, train_loader, val_loader)
#!SECTION