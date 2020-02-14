import os
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset


class WAVDataset(Dataset):
    """
    Create a PyTorch Dataset object from a directory containing clean and noisy WAV files
    """
    def __init__(self, dir: Path):
        self.clean_dir = dir.joinpath('clean')
        self.noisy_dir = dir.joinpath('noisy')

        assert os.path.exists(self.clean_dir), 'No clean WAV file folder found!'
        assert os.path.exists(self.noisy_dir), 'No noisy WAV file folder found!'

        self.clean_WAVs = {}
        for i, filename in enumerate(sorted(os.listdir(self.clean_dir))):
            self.clean_WAVs[i] = self.clean_dir.joinpath(filename)

        self.noisy_WAVs = {}
        for i, filename in enumerate(sorted(os.listdir(self.noisy_dir))):
            self.noisy_WAVs[i] = self.noisy_dir.joinpath(filename)

    def __len__(self):
        return len([name for name in os.listdir(self.clean_dir) if os.path.isfile(os.path.join(self.clean_dir, name))])

    def __getitem__(self, idx):
        noisy_path = self.noisy_WAVs[idx]
        clean_path = self.clean_dir.joinpath(noisy_path.name.split('_')[0] + '.wav')  # get the filename of the clean WAV from the filename of the noisy WAV
        clean_waveform, _ = torchaudio.load_wav(clean_path)
        noisy_waveform, _ = torchaudio.load_wav(noisy_path)

        assert clean_waveform.shape[0] == 1 and noisy_waveform.shape[0] == 1, 'WAV file is not single channel!'

        return noisy_waveform.view(-1), clean_waveform.view(-1)
