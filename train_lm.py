# SECTION: Necessary imports
import torch
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from pathlib import Path

from modules import CausalTransformer
#!SECTION

# SECTION: Dataloaders and LightningModules
class Wikitext103Dataset(data.Dataset):
    def __init__(self, tokens_path: Path):
        super().__init__()
        self.data = torch.load(tokens_path)

    def context_length(self):
        return self.data.size()[1]

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        # TODO: Should I preprocess differently to get the next token as the label instead of a pad id?
        # TODO: Include the tokenizer and use tokenizer.pad_id() instead of torch.zeros
        labels = torch.cat([self.data[idx][1:], torch.zeros(1)])
        return self.data[idx], labels
    

class Wikitext103Model(CausalTransformer):
    def _calculate_loss(self, batch, mode="train"):
        # TODO: I don't like all of these type conversions. How can I condense this?
        #       It seems like F.one_hot() and F.cross_entropy() require inputs to be long. But this is super inefficient
        #       for data storage. Plus, we need the data to be .float() to be sent through the transformer.
        data, labels = batch

        # NOTE: Data must be stored as longs for F.one_hot to work
        # TODO: Should I pre-encode the tensors before I save them? Or is F.one_hot negligible work?
        data = F.one_hot(data.long(), num_classes=self.hparams.num_classes).float()  # TODO: inside of forward, the data has dtype torch.float32... is AMP even working if we do this?

        preds = self.forward(data) # shape = [bsz, context_len, vocab_size]
        labels = labels.long()

        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        self.log("%s_loss" % mode, loss)
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
CHECKPOINT_PATH = "./checkpoints"
TRAIN_PATH = "./data/wikitext-103/wiki.train.tokens.tokenized.pt"
VALID_PATH = "./data/wikitext-103/wiki.valid.tokens.tokenized.pt"
TEST_PATH = "./data/wikitext-103/wiki.train.tokens.tokenized.pt"
#!SECTION
        
# SECTION: Training loop
if __name__ == "__main__":
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = Path(CHECKPOINT_PATH)
    root_dir.mkdir(exist_ok=True)
    trainer = L.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        accelerator="gpu",
        devices=3,
        strategy=DDPStrategy(static_graph=True),
        precision="16-mixed",
        max_epochs=100,
        gradient_clip_val=0.1,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create dataloaders
    train_dataset = Wikitext103Dataset(TRAIN_PATH)
    val_dataset = Wikitext103Dataset(VALID_PATH)
    test_dataset = Wikitext103Dataset(TEST_PATH)

    train_loader = data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=3, pin_memory=True # TODO: Crank up the batch size
    )
    val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = Path(CHECKPOINT_PATH, "Wikitext103Model.ckpt")
    if pretrained_filename.is_file():
        print("Found pretrained model, loading...")
        model = Wikitext103Model.load_from_checkpoint(pretrained_filename)
    else:
        model = Wikitext103Model(input_dim=32000,  # I think this is just context_length... do we need to one-hot encode?
                                 model_dim=128,
                                 num_classes=32000,  # TODO: use the tokenizer to find this value
                                 lr=0.01,
                                 max_iters=trainer.max_epochs * len(train_loader),
                                 warmup=1000,
                                 max_context_len=1024)
        trainer.fit(model, train_loader, val_loader)
        model = Wikitext103Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {
        "test_acc": test_result[0]["test_acc"],
        "val_acc": val_result[0]["test_acc"],
        "train_acc": train_result[0]["test_acc"],
    }
#!SECTION