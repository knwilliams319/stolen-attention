import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from transformer import TransformerGPT
from lr_scheduler import REXScheduler

# SECTION: Datasets
class OpenbookQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path, tokenizer):
        super().__init__()
        # Read raw text data and separate the important blocks of the prompt
        data = []
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
                d['answer'] = tokens[6]   
                data.append(d)
        
        # Make a prompt and tokenize the examples 
        self.prompts = []
        lengths = []
        self.answers = []
        for example in data:
            if example['answer'] == 'A':
                self.answers.append(0)
            elif example['answer'] == 'B':
                self.answers.append(1)
            elif example['answer'] == 'C':
                self.answers.append(2)
            elif example['answer'] == 'D':
                self.answers.append(3)
            else:
                raise ValueError('Encountered invalid answer choice!')
            for choice in ['A', 'B', 'C', 'D']:
                text = '[CLS] %s %s %s [END]' % (example['fact'], example['stem'], example[choice])
                tokens = tokenizer(text)['input_ids']
                self.prompts.append(tokens)
                lengths.append(len(tokens))
        del data
        
        # Left-pad the questions to the same length, and create padding masks to ignore those positions when calcuating attention
        max_length = max(lengths)
        self.pad_masks = torch.zeros((len(self.prompts), max_length))
        for i, prompt in enumerate(self.prompts):
            self.pad_masks[i][-lengths[i]:] = 1
            padding = [0] * (max_length - lengths[i])
            self.prompts[i] = padding + prompt
        self.prompts = torch.tensor(self.prompts)
        del lengths

        # Save vocab size to generate logit masks at runtime (pre-creating them requires more memory than we have)
        self.vocab_size = len(tokenizer)

    @property
    def context_length(self):
        return len(self.prompts[0])

    def __len__(self):
        return len(self.prompts) // 4 # guaranteed to be divisible by 4

    def __getitem__(self, idx):
        # grab question and padding mask (correct answer choice is last token)
        tokens = self.prompts[idx:idx+4]
        answer = self.answers[idx]
        padding_mask = self.pad_masks[idx:idx+4]  # NOTE: if I run out of memory, I should make the padding masks at runtime, too

        # NOTE: [CLS] token is encoded as [58, 5097, 50, 60]

        # Create the numer_mask and denom_mask to compute model's likelihood of the given sequence
        numer_mask = torch.zeros((len(tokens), self.context_length, len(tokenizer)))  # TODO: this and the pad_mask should prolly be boolTensors
        denom_mask = torch.zeros((len(tokens), self.context_length, len(tokenizer)))
        for i, sequence in enumerate(tokens):
            for pos, token in enumerate(sequence):
                if token != 0: # skip pad tokens
                    numer_mask[i][pos][token] = 1
                    denom_mask[i][pos][:] = 1
        return tokens, answer, padding_mask, numer_mask, denom_mask
# !SECTION
    
# SECTION: Model Definition
class OpenbookQAModel(TransformerGPT):
    def __init__(self, vocab_size, d_model, N, heads, dropout, opt):
        super().__init__(vocab_size, d_model, N, heads, dropout, opt)

    def configure_optimizers(self):
        # So far, Adam with no scheduler has performed better than using a scheduler or another optimizer + a scheduler. 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.lr, betas=(0.9, 0.98), eps=1e-9)
        # scheduler = REXScheduler(optimizer, num_steps=opt.num_steps)
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
        data, answer, pad_mask, numer_mask, denom_mask = batch
        data, pad_mask, numer_mask, denom_mask = data.squeeze(0), pad_mask.squeeze(0), numer_mask.squeeze(0), denom_mask.squeeze(0)
        logits = torch.exp(self(data, trg_mask=pad_mask))

        # Try David's custom-built F.cross_entropy that's only over the answer space despite technically allowing the model to output any token
        # NOTE: GPT2Tokenizer has this mapping: A-->317, B-->347, C-->327, D-->360
        denominators = torch.sum(torch.sum(logits * denom_mask, dim=2), dim=1) # TODO: support multiple batches, this will sum up everything...
        numerators = torch.sum(torch.sum(logits * numer_mask, dim=2), dim=1)  # exponentiate score assigned to correct answer choice for each quesiton
        prompt_lengths = pad_mask.sum(dim=1)
        probs = (numerators / denominators) / prompt_lengths
        loss = -torch.log(probs[answer] / probs.sum()) # TODO: when I support multiple batches, I'll need to sum losses for each group of 4 and divide by batch size

        # Also find the number of correct answers. The model's chosen answer is the one with the highest logit. 
        correct = 1 if probs.argmax() == answer else 0
        return loss, correct
    
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
            float(num_correct),  # NOTE: 100% accuracy is 31*16 + 4 = 500 (and without the float conversion, I get a warning)
            on_step=False,
            on_epoch=True,
            reduce_fx="sum"
        )

def get_model(opt, vocab_size):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1
    
    model = OpenbookQAModel(vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout, opt)
       
    if opt.pretrained_path is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.pretrained_path))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model
# !SECTION: Model Definition

# SECTION: Constants and Paths
parent = Path(__file__).parent
EXPERIMENT_DIR = parent / 'experiments'
FINETUNE_DIR = EXPERIMENT_DIR / 'finetune'
TRAIN_PATH = parent / 'data' / 'obqa.train.txt'
VAL_PATH = parent / 'data' / 'obqa.valid.txt'
# !SECTION: Constants and Paths

# SECTION: Main (Training Loop)
if __name__ == "__main__":
    # Parse CLI Arguments
    # TODO: Create an argument to toggle a LR Scheduler and implement the scheduler
    # TODO: Make CLI arguments out of the constants above
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    #parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-d_model', type=int, default=768)
    parser.add_argument('-n_layers', type=int, default=12)
    parser.add_argument('-heads', type=int, default=12)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1) # TODO: support higher batch sizes
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-norm', type=float, default=2.0)
    opt = parser.parse_args()

    # Post-process some of the CLI arguments
    opt.pretrained_path = EXPERIMENT_DIR / 'model_weights'
    opt.device = "cpu" if opt.no_cuda else "gpu"
    if opt.device == "gpu":
        assert torch.cuda.is_available()

    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") # len(tokenizer) == 50257

    # Create DataLoaders
    train_dataset = OpenbookQADataset(TRAIN_PATH, tokenizer)
    val_dataset = OpenbookQADataset(VAL_PATH, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=3)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=3)

    # Seed everything and initialize the Trainer
    L.seed_everything(10, workers=True)
    trainer = L.Trainer(
        deterministic=True,
        default_root_dir=FINETUNE_DIR,
        enable_progress_bar=True,
        logger=CSVLogger(
            EXPERIMENT_DIR,
            name='',
            version=FINETUNE_DIR.name
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
        accelerator=opt.device,
        devices=1,
        strategy="ddp",
        precision="32-true",
        max_epochs=opt.epochs,
        gradient_clip_val=opt.norm,
        benchmark=False,                       # this can't be used when deterministic=True
        profiler=None,                         # AdvancedProfiler(dirpath='./', filename='profile.log'),
        limit_train_batches=None,              # TODO: change this back to None
        limit_val_batches=None,                # TODO: change this back to None
        log_every_n_steps=opt.printevery       # TODO: change this back to 50
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Instantiate the pretrained model, create checkpoint directory, and finetune it
    opt.num_steps = trainer.max_epochs*len(train_loader)/trainer.num_devices
    model = get_model(opt, len(tokenizer))
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.fit(model, train_loader, val_loader)
# !SECTION: Main (Training Loop)