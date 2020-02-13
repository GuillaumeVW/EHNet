"""
Example template for defining a system
"""
import logging as log
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST

import pytorch_lightning as pl


class EHNetModel(pl.LightningModule):
    """
    Sample model to show how to define a template
    Input size: (batch_size, 1, frequency_bins, time)
    """

    def __init__(self, batch_size, spectograms_size, n_kernels, kernel_size, stride, padding,
                 n_lstm_layers, n_lstm_units, lstm_dropout):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(EHNetModel, self).__init__()

        self.batch_size = batch_size
        self.spectograms_size = spectograms_size
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_lstm_layers = n_lstm_layers
        self.n_lstm_units = n_lstm_units
        self.lstm_dropout = lstm_dropout

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.n_kernels,
                              kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding)
        n_features = self.n_kernels * (((self.spectograms_size[0] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=self.n_lstm_units, num_layers=self.n_lstm_layers,
                            batch_first=True, dropout=self.lstm_dropout, bidirectional=True)
        self.dense = nn.Linear(in_features=2 * self.n_lstm_units, out_features=self.spectograms_size[0])

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = x[:, None, :, :]  # (batch_size, 1, frequency_bins, time)
        x = nn.ReLU(self.conv(x))  # (batch_size, n_kernels, n_features, time)
        x = x.permute(0, 3, 1, 2)  # (batch_size, time, n_kernels, n_features)
        x = x.view(x.shape[0], x.shape[1], -1)  # (batch_size, time, n_kernels * n_features)
        x = self.lstm(x)  # (batch_size, time, 2 * n_lstm_units)
        x = nn.ReLU(self.dense(x))  # (batch_size, time, frequency_bins)
        x = x.permute(0, 2, 1)  # (batch_size, frequency_bins, time)

        return x

    def loss(self, target, prediction):
        loss = F.mse_loss(prediction, target)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch

        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch

        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adadelta(self.parameters(), lr=1.0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        # init data generators
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(root=self.hparams.data_root, train=train,
                        transform=transform, download=True)

        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        batch_size = self.hparams.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(train=True)

    @pl.data_loader
    def val_dataloader(self):
        log.info('Validation data loader called.')
        return self.__dataloader(train=False)

    @pl.data_loader
    def test_dataloader(self):
        log.info('Test data loader called.')
        return self.__dataloader(train=False)
