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
PATH_DATASETS = "/home/yujiang/Documents/data"

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


# Initialize a trainer
trainer = Trainer(
    max_epochs=100,
    progress_bar_refresh_rate=20,
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

