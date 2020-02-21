from pytorch_lightning import Trainer
from model.ehnet_model import EHNetModel
from pathlib import Path

checkpoint = Path('./lightning_logs/version_16/checkpoints/_ckpt_epoch_43.ckpt')

model = EHNetModel.load_from_checkpoint(checkpoint_path=checkpoint)

trainer = Trainer(gpus=1)
trainer.test(model)
