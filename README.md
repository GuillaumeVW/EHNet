# EHNet
This in an implementation of EHNet [1] in PyTorch and PyTorch Lightning.
EHNet is a convolutional-recurrent neural network for single channel speech enhancement.

## Prerequisites
* torch 1.4
* pytorch_lightning 0.7.6
* torchaudio 1.4
* soundfile 0.10.3.post1

## How to train
A dataset containg both clean speech and corresponding noisy speech (i.e. clean speech with noise added) is required.
3 notebooks are included to generate this dataset from a dataset consisting of clean speech recordings and noise recordings.

Running _train_nn.py_ starts the training.

The _train_dir_ variable should contain the path to a folder containing a _clean_ and a _noisy_ folder, containing the clean WAV files and the noisy WAV files respectively. The filename of a noisy WAV file must be the same as the corresponding clean WAV file, with optionally a suffix added delimited by _+_,
e.g. clean01.wav &rarr; clean01+noise.wav

The _val_dir_ follows the same convention, but this folder is used for validation.

## How to test
Running the _test_nn.py_ file results in the output (denoised) WAV files.

_testing_dir_ should point to a folder with the same structure as _train_dir_ and _val_dir_.

## Acknowledgements
[1] H. Zhao, S. Zarar, I. Tashev, and C.-H. Lee, "Convolutional-Recurrent Neural Networks for Speech Enhancement," arXiv:1805.00579 [cs, eess], May 2018.