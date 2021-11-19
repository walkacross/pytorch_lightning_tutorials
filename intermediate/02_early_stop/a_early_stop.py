#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

#PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
PATH_DATASETS = "/data/user/yujiang/"

BATCH_SIZE = 32

#AVAIL_GPUS = min(1, torch.cuda.device_count())
#BATCH_SIZE = 256 if AVAIL_GPUS else 64

# %% [markdown]
# ## Simplest example
#
# Here's the simplest most minimal example with just a training loop (no validation, no testing).
#
# **Keep in Mind** - A `LightningModule` *is* a PyTorch `nn.Module` - it just has a few more helpful features.


# %%
class MNISTModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.my_custom_learningin_rate = 100
        
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# %% [markdown]
# By using the `Trainer` you automatically get:
# 1. Tensorboard logging
# 2. Model checkpointing
# 3. Training and validation loop
# 4. early-stopping

# %%
# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_full = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST(PATH_DATASETS, train=False, transform=transforms.ToTensor())

mnist_train, mnist_val = random_split(train_full, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE)
val_loader = DataLoader(mnist_val, batch_size=BATCH_SIZE)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE)


# Initialize a trainer and set

#automatically saves a checkpoint for you in your current working directory, with the state of your last training epoch. This makes sure you can resume training in case it was interrupted.
# saves checkpoints to '/your/path/to/save/checkpoints' at every epoch end
"""
A Lightning checkpoint has everything needed to restore a training session including:

16-bit scaling factor (apex)

Current epoch

Global step

Model state_dict

State of all optimizers

State of all learningRate schedulers

State of all callbacks

The hyperparameters used for that model if passed in as hparams (Argparse.Namespace)
"""
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_to_save_dir = "/data/user/yujiang/tmp/pl"
# Init ModelCheckpoint callback, monitoring 'val_loss'
#checkpoint_callback = ModelCheckpoint(monitor="val_loss")
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_to_save_dir,
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
    save_top_k=2,
    mode="min",
)


# add early_stop callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=True, mode="min")


trainer = Trainer(
    max_epochs=10,
    progress_bar_refresh_rate=20,
    
    #!
   callbacks=[checkpoint_callback, early_stop_callback]
)


# Train the model ⚡
trainer.fit(mnist_model, train_dataloader=train_loader, val_dataloaders=val_loader)


# run test set
result = trainer.test(model=mnist_model, test_dataloaders=test_loader)
print(result)

# %%
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs


# %%

#checkpoint_path = "/data/user/yujiang/tmp/pl/lightning_logs/version_0/checkpoints/epoch=8-step=15470.ckpt"
#checkpoint loading
#model = MNISTModel.load_from_checkpoint(checkpoint_path)

#print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

#model.eval()
#y_hat = model(x)




