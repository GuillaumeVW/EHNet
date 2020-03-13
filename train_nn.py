from pytorch_lightning import Trainer
from pytorch_lightning.logging import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model.ehnet_model import EHNetModel
from argparse import Namespace
import os

train_dir = './WAVs/dataset/training'
val_dir = './WAVs/dataset/validation'

model = EHNetModel(hparams=Namespace(**{'train_dir': train_dir,
                                        'val_dir': val_dir,
                                        'batch_size': 16,
                                        'n_frequency_bins': 256,
                                        'n_kernels': 256,
                                        'kernel_size_f': 32,
                                        'kernel_size_t': 11,
                                        'n_lstm_layers': 2,
                                        'n_lstm_units': 1024,
                                        'lstm_dropout': 0.2}))

logger = CometLogger(api_key="4iSR8MNOiJzKARKC4OTTa7kg8", project_name="ehnet", workspace="guillaumevw")
checkpoint_path = os.path.join('lightning_logs', str(logger.version))
os.makedirs(checkpoint_path)
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_top_k=1, mode='min')
trainer = Trainer(gpus=1, min_epochs=200, logger=logger, checkpoint_callback=checkpoint_callback)
trainer.fit(model)
