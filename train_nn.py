from pytorch_lightning import Trainer
from model.ehnet_model import EHNetModel
from pathlib import Path

train_dir = Path('./WAVs/dataset/training')
val_dir = Path('./WAVs/dataset/validation')


model = EHNetModel(train_dir=train_dir, val_dir=val_dir, batch_size=4)

trainer = Trainer()
trainer.fit(model)
