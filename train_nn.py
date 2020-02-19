from pytorch_lightning import Trainer
from model.ehnet_model import EHNetModel
from pathlib import Path
from torch import tensor

train_dir = Path('./WAVs/dataset/training')
val_dir = Path('./WAVs/dataset/validation')
test_dir = Path('./WAVs/dataset/testing_seen_noise')

model = EHNetModel(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, hparams={'batch_size': 32,
                                                                                     'n_frequency_bins': 256,
                                                                                     'n_kernels': 256,
                                                                                     'kernel_size': tensor([32, 11]),
                                                                                     'n_lstm_layers': 2,
                                                                                     'n_lstm_units': 1024,
                                                                                     'lstm_dropout': 0})

trainer = Trainer(gpus=1)
trainer.fit(model)
