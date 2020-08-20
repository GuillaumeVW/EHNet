import logging as log
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram

from dataloader.wav_dataset import WAVDataset


class EHNetModel(pl.LightningModule):
    def __init__(self, hparams=Namespace(**{'train_dir': Path(), 'val_dir': Path(), 'batch_size': 4, 'n_frequency_bins': 256, 'n_kernels': 256,
                                            'kernel_size_f': 32, 'kernel_size_t': 11, 'n_lstm_layers': 2, 'n_lstm_units': 1024, 'lstm_dropout': 0})):
        super(EHNetModel, self).__init__()

        self.hparams = hparams
        self.train_dir = Path(self.hparams.train_dir)
        self.val_dir = Path(self.hparams.val_dir)
        self.batch_size = self.hparams.batch_size
        self.n_frequency_bins = self.hparams.n_frequency_bins
        self.n_kernels = self.hparams.n_kernels
        self.kernel_size = (self.hparams.kernel_size_f, self.hparams.kernel_size_t)
        self.stride = (self.kernel_size[0] // 2, 1)
        self.padding = (0, self.kernel_size[1] // 2)
        self.n_lstm_layers = self.hparams.n_lstm_layers
        self.n_lstm_units = self.hparams.n_lstm_units
        self.lstm_dropout = self.hparams.lstm_dropout

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.n_kernels,
                              kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding)
        n_features = int(self.n_kernels * (((self.n_frequency_bins - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]) + 1))
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=self.n_lstm_units, num_layers=self.n_lstm_layers,
                            batch_first=True, dropout=self.lstm_dropout, bidirectional=True)
        self.dense = nn.Linear(in_features=2 * self.n_lstm_units, out_features=self.n_frequency_bins)
        self.flatten = nn.Flatten(start_dim=2)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        x.requires_grad = True
        x = torch.unsqueeze(x, 1)  # (batch_size, 1, frequency_bins, time)
        x = F.relu(self.conv(x))  # (batch_size, n_kernels, n_features, time)
        x = x.permute(0, 3, 1, 2)  # (batch_size, time, n_kernels, n_features)
        x = self.flatten(x)  # (batch_size, time, n_kernels * n_features)
        x, _ = self.lstm(x)  # (batch_size, time, 2 * n_lstm_units)
        x = F.relu(self.dense(x))  # (batch_size, time, frequency_bins)
        x = x.permute(0, 2, 1)  # (batch_size, frequency_bins, time)

        return x

    def loss(self, target, prediction):
        loss = F.mse_loss(prediction, target)
        return loss

    def training_step(self, batch, batch_idx):
        # forward pass
        x, y = batch

        x_ms = x.pow(2).sum(-1).sqrt()
        y_ms = y.pow(2).sum(-1).sqrt()

        y_hat = self.forward(x_ms)

        loss_val = self.loss(y_ms, y_hat)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x_ms = x.pow(2).sum(-1).sqrt()
        y_ms = y.pow(2).sum(-1).sqrt()

        y_hat = self.forward(x_ms)

        loss_val = self.loss(y_ms, y_hat)

        output = OrderedDict({
            'val_loss': loss_val,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=1.0, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        # init data generators
        transform = Spectrogram(n_fft=(self.n_frequency_bins - 1) * 2, power=None)

        if train:
            dataset = WAVDataset(self.train_dir, transform=transform)
        else:
            dataset = WAVDataset(self.val_dir, transform=transform)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        return loader

    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return self.__dataloader(train=False)
