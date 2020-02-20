from dataloader.wav_dataset import WAVDataset
from pathlib import Path
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram


dataset_dir = Path('./WAVs/dataset/training')

n_frequency_bins = 256
transform = Spectrogram(n_fft=(n_frequency_bins - 1) * 2)

wav_dataset = WAVDataset(dataset_dir, transform=transform)

dataloader = DataLoader(wav_dataset, batch_size=4, shuffle=True)

i = 0
for x, y in dataloader:
    print('Batch:', i, '- Shape x:', x.shape, '- Shape y:', y.shape)
    i += 1
