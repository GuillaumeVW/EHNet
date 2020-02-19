"""
Example template for defining a system
"""
import logging as log
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchaudio.transforms import Spectrogram
from torchaudio.functional import angle, istft
from dataloader.wav_dataset import WAVDataset


import pytorch_lightning as pl


class EHNetModel(pl.LightningModule):
    """
    Sample model to show how to define a template
    Input size: (batch_size, frequency_bins, time)
    """

    def __init__(self, train_dir, val_dir, hparams={'batch_size': 4, 'n_frequency_bins': 256, 'n_kernels': 256, 'kernel_size': (32, 11),
                 'n_lstm_layers': 2, 'n_lstm_units': 1024, 'lstm_dropout': 0}):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(EHNetModel, self).__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = hparams['batch_size']
        self.n_frequency_bins = hparams['n_frequency_bins']
        self.n_kernels = hparams['n_kernels']
        self.kernel_size = hparams['kernel_size']
        self.stride = (self.kernel_size[0] // 2, 1)
        self.padding = (self.kernel_size[1] // 2, self.kernel_size[1] // 2)
        self.n_lstm_layers = hparams['n_lstm_layers']
        self.n_lstm_units = hparams['n_lstm_units']
        self.lstm_dropout = hparams['lstm_dropout']

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
        n_features = int(self.n_kernels * (((self.n_frequency_bins - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]) + 1))
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=self.n_lstm_units, num_layers=self.n_lstm_layers,
                            batch_first=True, dropout=self.lstm_dropout, bidirectional=True)
        self.dense = nn.Linear(in_features=2 * self.n_lstm_units, out_features=self.n_frequency_bins)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = torch.unsqueeze(x, 1)  # (batch_size, 1, frequency_bins, time)
        x = nn.ReLU()(self.conv(x))  # (batch_size, n_kernels, n_features, time)
        x = x.permute(0, 3, 1, 2)  # (batch_size, time, n_kernels, n_features)
        x = nn.Flatten(start_dim=2)(x)  # (batch_size, time, n_kernels * n_features)
        x, _ = self.lstm(x)  # (batch_size, time, 2 * n_lstm_units)
        x = nn.ReLU()(self.dense(x))  # (batch_size, time, frequency_bins)
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
        n_fft = (self.n_frequency_bins - 1) * 2

        x_waveform, y_waveform = batch
        
        window = torch.hann_window(n_fft)
        if x_waveform.is_cuda:
            window = window.cuda()

        x_stft = torch.stft(x_waveform, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, window=window)
        y_stft = torch.stft(y_waveform, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, window=window)

        x_spectrogram, y_spectrogram = (x_stft.pow(2).sum(-1), y_stft.pow(2).sum(-1))

        y_spectrogram_hat = self.forward(x_spectrogram)

        y_stft_hat = torch.stack([y_spectrogram_hat.sqrt() * torch.cos(angle(x_stft)),
                                  y_spectrogram_hat.sqrt() * torch.sin(angle(x_stft))], dim=-1)

        y_waveform_hat = istft(y_stft_hat, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, window=window, length=y_waveform.shape[-1])

        loss_val = self.loss(y_spectrogram, y_spectrogram_hat)

        timedomain_MSE_val = self.loss(y_waveform, y_waveform_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_timedomain_MSE': timedomain_MSE_val
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
        val_timedomain_MSE_mean = 0
        for output in outputs:
            val_loss = output['val_loss']
            val_timedomain_MSE = output['val_timedomain_MSE']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
                val_timedomain_MSE = torch.mean(val_timedomain_MSE)
            val_loss_mean += val_loss
            val_timedomain_MSE_mean += val_timedomain_MSE

        val_loss_mean /= len(outputs)
        val_timedomain_MSE_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_timedomain_MSE': val_timedomain_MSE_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean, 'val_timedomain_MSE': val_timedomain_MSE_mean}
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

        transform = Spectrogram(n_fft=(self.n_frequency_bins - 1) * 2)

        if train:
            dataset = WAVDataset(self.train_dir, transform=transform)
        else:
            dataset = WAVDataset(self.val_dir, transform=None)

        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        batch_size = self.batch_size

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
