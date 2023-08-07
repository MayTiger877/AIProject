
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as display
import librosa.feature
from scipy.signal import resample
import os
import os.path
import sys
import torch
import torchaudio
import cv2
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


def test_data(clean_audio_dir, noisy_audio_dir, lower_bound, upper_bound, checkpoints):
    """
    read test data generating reverberant spectrograms and waveforms
    
    clean_audio_dir: directory containing speech audio files
    noisy_audio_dir: directory
    lower_bound: initial example to be considered
    upper_bound: final example to be considered
    checkpoints: list containing directories for save rev spectrogram, target spectrogram
                rev waveforms and target waveforms respectively
    noise: add noise with random snr in [noise[0], noise[1]]
    """
    
    checkpointX = checkpoints[0]
    checkpointY = checkpoints[1]
    checkpoint_wavenoisy = checkpoints[2]
    checkpoint_wavetarget = checkpoints[3]
    
    sys.path.append(clean_audio_dir)
    sys.path.append(noisy_audio_dir)

    clean_audio_file_names = []
    for subdir, dirs, files in os.walk(clean_audio_dir):
        for file in files:
            if (".wav" in file):
                clean_audio_file_names.append(os.path.join(subdir,file))
                
    
    print ("Audio files found: " + str(len(clean_audio_file_names)))
    
    noisy_audio_file_names = []
    for subdir, dirs, files in os.walk(noisy_audio_dir):
        for file in files:
            if (".wav" in file):
                noisy_audio_file_names.append(os.path.join(subdir,file))
    
    print ("Audio files found: " + str(len(noisy_audio_file_names)))
    
    time_size = 340
    frequency_size = 128
    X = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))
    y = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))

    wave_targets = []
    wave_noisy = []

    for i in range(lower_bound, upper_bound):
        clean_speech_audio, clean_speech_time, clean_speech_rate = extract_audio(clean_audio_file_names[i])
        wave_targets.append(clean_speech_audio)
        clean_speech_spec = generate_spec(clean_speech_audio, clean_speech_rate)
        
        noisy_speech_audio, noisy_speech_time, noisy_speech_rate = extract_audio(noisy_audio_file_names[i])
        wave_targets.append(noisy_speech_audio)
        noisy_speech_spec = generate_spec(noisy_speech_audio, noisy_speech_rate)
        
        clean_speech_spec = cv2.resize(clean_speech_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
        noisy_speech_spec = cv2.resize(noisy_speech_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)

        print("Proccesing audio file n°: " + str(i+1), flush = True)
        X[i-lower_bound, 0, :, :] = torch.tensor(noisy_speech_spec)
        y[i-lower_bound, 0, :, :] = torch.tensor(clean_speech_spec)
            
        if ((i+1)%500 == 0):    
            torch.save(X, checkpointX)
            torch.save(y, checkpointY)
            torch.save(wave_noisy, checkpoint_wavenoisy)
            torch.save(wave_targets, checkpoint_wavetarget)
            print('Saved data', flush = True)
          
        if ((i+1)==(len(noisy_audio_file_names))):
            torch.save(X, checkpointX)
            torch.save(y, checkpointY)
            torch.save(wave_noisy, checkpoint_wavenoisy)
            torch.save(wave_targets, checkpoint_wavetarget)
            print('Saved data', flush = True)
            return X, y

    return X, y

def DENoise_gen_spec():
    dry_audio_rootdir = '/home/may.tiger/AIProject/big_data_set/denoise_data/clean_16Hz'
    wet_audio_rootdir = '/home/may.tiger/AIProject/big_data_set/denoise_data/noisyreverb_16Hz'

    checkpointX =           '/home/may.tiger/AIProject/de_noising/generateSpecs/noisyspecs.pth'
    checkpointY =           '/home/may.tiger/AIProject/de_noising/generateSpecs/cleanspecs.pth'
    checkpoint_wavenoisy =  '/home/may.tiger/AIProject/de_noising/generateSpecs/wavenoisy.pth'
    checkpoint_wavetarget = '/home/may.tiger/AIProject/de_noising/generateSpecs/waveclean.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_wavenoisy, checkpoint_wavetarget]

    print("starting generating\n")
    dir_len = len([entry for entry in os.listdir(dry_audio_rootdir) if os.path.isfile(os.path.join(dry_audio_rootdir, entry))])
    print(dir_len)
    X, y = test_data(dry_audio_rootdir, wet_audio_rootdir, 0, dir_len, checkpoints)

    print("finished generating\n")
    
def DENoise_gen_spec_extra():
    dry_audio_rootdir = '/home/may.tiger/AIProject/big_data_set/denoise_extra/dry'
    wet_audio_rootdir = '/home/may.tiger/AIProject/big_data_set/denoise_extra/wet'

    checkpointX =           '/home/may.tiger/AIProject/de_noising/generateSpecs_extra/noisyspecs.pth'
    checkpointY =           '/home/may.tiger/AIProject/de_noising/generateSpecs_extra/cleanspecs.pth'
    checkpoint_wavenoisy =  '/home/may.tiger/AIProject/de_noising/generateSpecs_extra/wavenoisy.pth'
    checkpoint_wavetarget = '/home/may.tiger/AIProject/de_noising/generateSpecs_extra/waveclean.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_wavenoisy, checkpoint_wavetarget]

    print("starting generating\n")
    dir_len = len([entry for entry in os.listdir(dry_audio_rootdir) if os.path.isfile(os.path.join(dry_audio_rootdir, entry))])
    print(dir_len)
    X, y = test_data(dry_audio_rootdir, wet_audio_rootdir, 0, dir_len, checkpoints)

    print("finished generating\n")
    
    
DENoise_gen_spec_extra()