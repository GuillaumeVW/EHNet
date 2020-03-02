"""
Example template for defining a system
"""
import logging as log
from collections import OrderedDict
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchaudio.transforms import Spectrogram
from torchaudio.functional import angle, istft
from dataloader.wav_dataset import WAVDataset
from utils.metrics import SNR, PESQ, STOI


import pytorch_lightning as pl


class EHNetModel(pl.LightningModule):
    """
    Sample model to show how to define a template
    Input size: (batch_size, frequency_bins, time)
    """

    def __init__(self, hparams=Namespace(**{'train_dir': None, 'val_dir': None, 'test_dir': None, 'batch_size': 4, 'n_frequency_bins': 256, 'n_kernels': 256,
                                            'kernel_size_f': 32, 'kernel_size_t': 11, 'n_lstm_layers': 2, 'n_lstm_units': 1024, 'lstm_dropout': 0})):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(EHNetModel, self).__init__()

        self.hparams = hparams
        self.train_dir = Path(self.hparams.train_dir)
        self.val_dir = Path(self.hparams.val_dir)
        self.test_dir = Path(self.hparams.test_dir)
        self.batch_size = self.hparams.batch_size
        self.n_frequency_bins = self.hparams.n_frequency_bins
        self.n_kernels = self.hparams.n_kernels
        self.kernel_size = (self.hparams.kernel_size_f, self.hparams.kernel_size_t)
        self.stride = (self.kernel_size[0] // 2, 1)
        self.padding = (self.kernel_size[1] // 2, self.kernel_size[1] // 2)
        self.n_lstm_layers = self.hparams.n_lstm_layers
        self.n_lstm_units = self.hparams.n_lstm_units
        self.lstm_dropout = self.hparams.lstm_dropout

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

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the testing loop
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

        loss_test = self.loss(y_spectrogram, y_spectrogram_hat)

        # calculate average SNR, PESQ and STOI for the batch
        noisy_waveforms = torch.unbind(x_waveform.cpu())
        clean_waveforms = torch.unbind(y_waveform.cpu())
        denoised_waveforms = torch.unbind(y_waveform_hat.cpu())

        orig_SNR = 0
        denoised_SNR = 0

        orig_PESQ = 0
        denoised_PESQ = 0

        orig_STOI = 0
        denoised_STOI = 0
        
        for noisy_waveform, clean_waveform, denoised_waveform in zip(noisy_waveforms, clean_waveforms, denoised_waveforms):
            noisy_waveform = noisy_waveform.numpy()
            clean_waveform = clean_waveform.numpy()
            denoised_waveform = denoised_waveform.numpy()

            orig_SNR += SNR(noisy_waveform, clean_waveform)
            denoised_SNR += SNR(denoised_waveform, clean_waveform)

            orig_PESQ += PESQ(noisy_waveform, clean_waveform)
            denoised_PESQ += PESQ(denoised_waveform, clean_waveform)

            orig_STOI += STOI(noisy_waveform, clean_waveform)
            denoised_STOI += STOI(denoised_waveform, clean_waveform)

        orig_SNR /= self.batch_size
        denoised_SNR /= self.batch_size

        orig_PESQ /= self.batch_size
        denoised_PESQ /= self.batch_size

        orig_STOI /= self.batch_size
        denoised_STOI /= self.batch_size

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_test = loss_test.unsqueeze(0)

        output = OrderedDict({
            'test_loss': loss_test,
            'test_orig_SNR': orig_SNR,
            'test_denoised_SNR': denoised_SNR,
            'test_orig_PESQ': orig_PESQ,
            'test_denoised_PESQ': denoised_PESQ,
            'test_orig_STOI': orig_STOI,
            'test_denoised_STOI': denoised_STOI
        })

        return output

    def test_end(self, outputs):
        """
        Called at the end of testing to aggregate outputs
        :param outputs: list of individual outputs of each test step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        test_loss_mean = 0
        test_orig_SNR_mean = 0
        test_denoised_SNR_mean = 0
        test_orig_PESQ_mean = 0
        test_denoised_PESQ_mean = 0
        test_orig_STOI_mean = 0
        test_denoised_STOI_mean = 0
        for output in outputs:
            test_loss = output['test_loss']
            test_orig_SNR = output['test_orig_SNR']
            test_denoised_SNR = output['test_denoised_SNR']
            test_orig_PESQ = output['test_orig_PESQ']
            test_denoised_PESQ = output['test_denoised_PESQ']
            test_orig_STOI = output['test_orig_STOI']
            test_denoised_STOI = output['test_denoised_STOI']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                test_loss = torch.mean(test_loss)
            
            test_loss_mean += test_loss
            test_orig_SNR_mean += test_orig_SNR
            test_denoised_SNR_mean += test_denoised_SNR
            test_orig_PESQ_mean += test_orig_PESQ
            test_denoised_PESQ_mean += test_denoised_PESQ
            test_orig_STOI_mean += test_orig_STOI
            test_denoised_STOI_mean += test_denoised_STOI

        test_loss_mean /= len(outputs)
        test_orig_SNR_mean /= len(outputs)
        test_denoised_SNR_mean /= len(outputs)
        test_orig_PESQ_mean /= len(outputs)
        test_denoised_PESQ_mean /= len(outputs)
        test_orig_STOI_mean /= len(outputs)
        test_denoised_STOI_mean /= len(outputs)
        
        tqdm_dict = {'test_loss': test_loss_mean,
                     'test_orig_SNR': test_orig_SNR_mean,
                     'test_denoised_SNR': test_denoised_SNR_mean,
                     'test_orig_PESQ': test_orig_PESQ_mean,
                     'test_denoised_PESQ': test_denoised_PESQ_mean,
                     'test_orig_STOI': test_orig_STOI_mean,
                     'test_denoised_STOI': test_denoised_STOI_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_loss': test_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adadelta(self.parameters(), lr=1.0, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        # init data generators

        transform = Spectrogram(n_fft=(self.n_frequency_bins - 1) * 2, normal=True)

        if train:
            dataset = WAVDataset(self.train_dir, transform=transform)
        else:
            dataset = WAVDataset(self.val_dir, transform=transform)

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

    def __test_dataloader(self):
        dataset = WAVDataset(self.test_dir, transform=None)

        batch_size = self.batch_size

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
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
        return self.__test_dataloader()
