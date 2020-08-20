from pytorch_lightning import Trainer
from model.ehnet_model import EHNetModel
from argparse import Namespace
import os

train_dir = './WAVs/dataset/training'
val_dir = './WAVs/dataset/validation'

hparams = {'train_dir': train_dir,
           'val_dir': val_dir,
           'batch_size': 64,
           'n_frequency_bins': 256,
           'n_kernels': 256,
           'kernel_size_f': 32,
           'kernel_size_t': 11,
           'n_lstm_layers': 2,
           'n_lstm_units': 1024,
           'lstm_dropout': 0.2}

model = EHNetModel(hparams=Namespace(**hparams))

trainer = Trainer(gpus=1, min_epochs=200)
trainer.fit(model)
