from torch.nn import BCELoss, CrossEntropyLoss
from pytorch_lightning import LightningModule
import torch
import numpy as np

# "." allows to import from the same directory
from .net1d import Net1D
from .heads import RegressionHead
from .losses import WeightedFocalLoss
import torch.nn.functional as F


class ResNet(LightningModule):
    def __init__(
        self, lr, n_channels=37, Focal_loss=False, online_hard_mine=None, **kwargs
    ):
        super().__init__()
        self.lr = lr
        self.Focal_loss = Focal_loss
        self.online_hard_mine = online_hard_mine
        self.model = Net1D(
            in_channels=n_channels,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=True,
            return_softmax=False,  # False
            n_classes=1,
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # flatten label
        y = y.view(-1, 1).float()
        logits = self.forward(x)
        loss_function = BCELoss()
        if self.Focal_loss:
            loss_function = WeightedFocalLoss()
        loss = loss_function(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # flatten label
        y = y.view(-1, 1).float()
        logits = self.forward(x)
        loss_function = BCELoss()
        if self.Focal_loss:
            loss_function = WeightedFocalLoss()
        loss = loss_function(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signals, labels = batch
        # flatten label
        labels = labels.view(-1, 1).float()
        # generate predictions
        out = self.forward(signals)
        # pred = self.forward(signals)

        # compute and log loss
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
