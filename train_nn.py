from pytorch_lightning import Trainer
from model.ehnet_model import EHNetModel
from pathlib import Path

train_dir = Path('./WAVs/dataset/training')
val_dir = Path('./WAVs/dataset/validation')


model = EHNetModel(train_dir=train_dir, val_dir=val_dir, hparams={'batch_size': 4,
                                                                  'n_frequency_bins': 256,
                                                                  'n_kernels': 256,
                                                                  'kernel_size': (32, 11),
                                                                  'n_lstm_layers': 2,
                                                                  'n_lstm_units': 1024,
                                                                  'lstm_dropout': 0})

trainer = Trainer()
trainer.fit(model)
