import os
from torch.utils.data import Dataset


class WAVDataset(Dataset):
    """
    Create a PyTorch Dataset object from a directory containing clean and noisy WAV files
    """
    def __init__(self, dir):
        self.clean_dir = os.path.join(dir, 'clean')
        self.noisy_dir = os.path.join(dir, 'noisy')

        assert os.path.exists(self.clean_dir), 'No clean WAV file folder found!'
        assert os.path.exists(self.noisy_dir), 'No noisy WAV file folder found!'

    def __len__(self):
        return len([name for name in os.listdir(self.clean_dir) if os.path.isfile(os.path.join(self.clean_dir, name))])
