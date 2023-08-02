import torch
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import librosa
import librosa.display as display
librosa.util.example_audio_file = lambda: 'sox'
import soundfile as sf
import cv2
import glob
import torchaudio.functional as F
import torchaudio.transforms as T
import math
import requests
import random
import librosa.feature
from scipy.signal import resample
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.io.wavfile import write
import scienceplots
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../de_noising"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../fine_tuning"))
from utils import *
from LateSupressionUnet import *
from DeNoiserUnet import *
from FineTuneModel import *

def generate_spec_exp(audio_sequence, rate, n_fft=4096, hop_length=512):
    """
    Generate spectrogram using librosa
    audio_sequence: list representing waveform
    rate: sampling rate (16000 for all LibriSpeech audios)
    nfft and hop_length: stft parameters
    """
    S = librosa.feature.melspectrogram(y=audio_sequence, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmin=20,
                                       fmax=8000)
    log_spectra = librosa.power_to_db(S, ref=np.mean, top_db=80)
    return log_spectra

def reconstruct_wave_exp(spec, rate=16000, normalize_data=False):
    """
    Reconstruct waveform
    spec: spectrogram generated using Librosa
    rate: sampling rate
    """
    power = librosa.db_to_power(spec, ref=5.0)
    audio = librosa.feature.inverse.mel_to_audio(power, sr=rate, n_fft=4096, hop_length=512)
    out_audio = audio / np.max(audio) if normalize_data else audio
    return out_audio


def graph_spec_exp(spec, rate=16000, title="Log-Power spectrogram", save_path="./spec"):
    """
    plot spectrogram
    spec: spectrogram generated using Librosa
    rate: sampling rate
    """
    plt.figure()
    display.specshow(spec, sr=rate, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path + ".png")

def plot_time_wave_exp(audio, title, rate=16000):
    """
    plot waveform given speech audio
    audio: array containing waveform
    rate: sampling rate

    """
    time = np.linspace(0, len(audio)/rate, len(audio), endpoint=False)
    plt.figure()
    plt.plot(time, audio)
    plt.xlabel("Time (secs)")
    plt.ylabel("Power")
    plt.savefig(f"{title}.jpg")

#######################################################################################################

def check_librosa_wav_to_spec():
    """ """
    # Load audio file
    dry_example =       "/home/may.tiger/AIProject/librosa_experiment/f0001_us_f0001_00008_dry.wav"
    revereb_example =   "/home/may.tiger/AIProject/librosa_experiment/f0001_us_f0001_00008_wet.wav" 
    
    # extract data
    dry_example_data, dry_example_time, dry_example_rate = extract_audio(dry_example)
    revereb_example_data, revereb_example_time, revereb_example_rate = extract_audio(revereb_example)
    
    # generate spectrograms
    dry_spec =      generate_spec_exp(dry_example_data,     dry_example_rate)
    reverb_spec =   generate_spec_exp(revereb_example_data, revereb_example_rate)
    
    # graph time waves
    plot_time_wave_exp(dry_example_data, "Dry_Wave_before")
    plot_time_wave_exp(revereb_example_data, "Wet_Wave_before")
    
    # graph spectrograms
    graph_spec_exp(dry_spec,    dry_example_rate,       "Dry_Spec_before", "Dry_Spec_before")
    graph_spec_exp(reverb_spec, revereb_example_rate,   "Wet_Spec_before", "Wet_Spec_before")
    
    # reconstruct wav files
    reconstruct_dry = reconstruct_wave_exp(dry_spec, dry_example_rate)
    reconstruct_wet = reconstruct_wave_exp(reverb_spec, revereb_example_rate)
    
    # graph time waves afetr reconstruction
    plot_time_wave_exp(reconstruct_dry, "Dry_Wave_after")
    plot_time_wave_exp(reconstruct_wet, "Wet_Wave_after")
    
    # save wav files
    sf.write("reconstruct_dry.wav", reconstruct_dry, dry_example_rate)
    sf.write("reconstruct_wet.wav", reconstruct_wet, revereb_example_rate)
    
check_librosa_wav_to_spec()