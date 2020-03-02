import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

torchaudio.set_audio_backend("soundfile")  # default backend (SoX) has bugs when loading WAVs


class WAVDataset(Dataset):
    """
    Create a PyTorch Dataset object from a directory containing clean and noisy WAV files
    """
    def __init__(self, dir: Path, transform=None):
        self.clean_dir = dir.joinpath('clean')
        self.noisy_dir = dir.joinpath('noisy')
        self.transform = transform

        assert os.path.exists(self.clean_dir), 'No clean WAV file folder found!'
        assert os.path.exists(self.noisy_dir), 'No noisy WAV file folder found!'

        self.clean_WAVs = {}
        for i, filename in enumerate(sorted(os.listdir(self.clean_dir))):
            self.clean_WAVs[i] = self.clean_dir.joinpath(filename)

        self.noisy_WAVs = {}
        for i, filename in enumerate(sorted(os.listdir(self.noisy_dir))):
            self.noisy_WAVs[i] = self.noisy_dir.joinpath(filename)

    def __len__(self):
        return len(self.noisy_WAVs)

    def __getitem__(self, idx):
        noisy_path = self.noisy_WAVs[idx]
        clean_path = self.clean_dir.joinpath(noisy_path.name.split('+')[0] + '.wav')  # get the filename of the clean WAV from the filename of the noisy WAV
        clean_waveform, _ = torchaudio.load(clean_path, normalization=lambda x: torch.abs(x).max())
        noisy_waveform, _ = torchaudio.load(noisy_path, normalization=lambda x: torch.abs(x).max())

        assert clean_waveform.shape[0] == 1 and noisy_waveform.shape[0] == 1, 'WAV file is not single channel!'

        if self.transform:
            return self.transform(noisy_waveform.view(-1)), self.transform(clean_waveform.view(-1))
        else:
            return noisy_waveform.view(-1), clean_waveform.view(-1)
