from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torchaudio.functional import angle, istft
from torchaudio.transforms import Spectrogram

from dataloader.wav_dataset import WAVDataset
from model.ehnet_model import EHNetModel

model = EHNetModel.load_from_checkpoint(Path('/home/guillaume/Downloads/epoch=30(1).ckpt'))

testing_dir = Path('./WAVs/MS-SNSD-test/testing_seen_noise')
dataset = WAVDataset(dir=testing_dir)
dataloader = DataLoader(dataset, batch_size=16, drop_last=False, shuffle=True)
noisy_batch, clean_batch = next(iter(dataloader))

#  enable eval mode
model.zero_grad()
model.eval()
model.freeze()

# disable gradients to save memory
torch.set_grad_enabled(False)

n_fft = (model.n_frequency_bins - 1) * 2

x_waveform = noisy_batch

transform = Spectrogram(n_fft=n_fft, power=None)

x_stft = transform(x_waveform)
y_stft = transform(clean_batch)
x_ms = x_stft.pow(2).sum(-1).sqrt()
y_ms = y_stft.pow(2).sum(-1).sqrt()

y_ms_hat = model(x_ms)

y_stft_hat = torch.stack([y_ms_hat * torch.cos(angle(x_stft)),
                          y_ms_hat * torch.sin(angle(x_stft))], dim=-1)

window = torch.hann_window(n_fft)
y_waveform_hat = istft(y_stft_hat, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, window=window, length=x_waveform.shape[-1])
for i, waveform in enumerate(y_waveform_hat.numpy()):
    sf.write('denoised' + str(i) + '.wav', waveform, 16000)
