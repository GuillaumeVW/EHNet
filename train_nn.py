from pytorch_lightning import Trainer
from model.ehnet_model import EHNetModel
from argparse import Namespace

train_dir = './WAVs/dataset/training'
val_dir = './WAVs/dataset/validation'
test_dir = './WAVs/dataset/testing_seen_noise'

model = EHNetModel(hparams=Namespace(**{'train_dir': train_dir,
                                        'val_dir': val_dir,
                                        'test_dir': test_dir,
                                        'batch_size': 32,
                                        'n_frequency_bins': 256,
                                        'n_kernels': 256,
                                        'kernel_size_f': 32,
                                        'kernel_size_t': 11,
                                        'n_lstm_layers': 2,
                                        'n_lstm_units': 1024,
                                        'lstm_dropout': 0}))

trainer = Trainer(gpus=1, max_epochs=200, min_epochs=120)
trainer.fit(model)
