import numpy as np
from pypesq import pesq
from pystoi.stoi import stoi
import math


def power(x):
    return np.sum(np.square(x))


def SNR(noisy_waveform, clean_waveform):
    noise_waveform = noisy_waveform - clean_waveform
    return 10 * math.log10(power(clean_waveform) / power(noise_waveform))


def PESQ(noisy_waveform, clean_waveform):
    return pesq(clean_waveform, noisy_waveform, 16000)


def STOI(noisy_waveform, clean_waveform):
    return stoi(clean_waveform, noisy_waveform, 16000)
