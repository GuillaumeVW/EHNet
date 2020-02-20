from dataloader.wav_dataset import WAVDataset
from pathlib import Path
from torchaudio.transforms import Spectrogram
import torch


dataset_dir = Path('./WAVs/dataset/training')

n_frequency_bins = 256
n_fft = (n_frequency_bins - 1) * 2
transform = Spectrogram(n_fft=n_fft)

wav_dataset_spectrogram = WAVDataset(dataset_dir, transform=transform)
wav_dataset_waveform = WAVDataset(dataset_dir, transform=None)

x_spectrogram, _ = wav_dataset_spectrogram.__getitem__(1)
x_waveform, _ = wav_dataset_waveform.__getitem__(1)

window = torch.hann_window(n_fft)
x_stft = torch.stft(x_waveform, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, window=window)

x_spectrogram_calculated = x_stft.pow(2).sum(-1)

print(x_spectrogram.shape)
print(x_spectrogram_calculated.shape)
print(torch.equal(x_spectrogram, x_spectrogram_calculated))
