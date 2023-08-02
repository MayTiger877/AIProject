
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


def train_data(audio_dir, rir_dir, lower_bound, upper_bound, checkpointX, checkpointY):
    """
    Read training data generating reverberant waveforms and spectrograms
   
    audio_dir: directory containing the speech audio files
    rir_dir: directory containing RIRs audio files
    lower_bound: initial example to be considered
    upper_bound: final example to be considered
    checkpointX: directory + filename to save reverberant data
    checkpointY: directory + filename to save target data 
    """

    sys.path.append(audio_dir)
    sys.path.append(rir_dir)
    
    rir_file_names = []
    for subdir, dirs, files in os.walk(rir_dir):
        for file in files:
            if (".wav" in file):
                rir_file_names.append(os.path.join(subdir,file))

    audio_file_names = []
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            if (".wav" in file):
                audio_file_names.append(os.path.join(subdir,file))
    
    print ("RIRs found: " + str(len(rir_file_names)))
    print ("Audio files found: " + str(len(audio_file_names)))
    time_size = 340
    frequency_size = 128
    X = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))
    y = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))
    
    for i in range(lower_bound, upper_bound):
        rir_index = random.sample(range(len(rir_file_names)), 1)[0]
        ir_audio, ir_time, ir_rate = extract_audio(rir_file_names[rir_index])

        speech_audio, speech_time, speech_rate = extract_audio(audio_file_names[i])
        speech_spec = generate_spec(speech_audio, speech_rate)
        
        random_snr = random.sample(range(15, 36), 1)[0]
        speech_rev = discrete_conv(speech_audio, ir_audio, 16000, 96000, snr = random_snr)
        speech_rev = speech_rev[0:len(speech_audio)]
        rev_spec = generate_spec(speech_rev, speech_rate)
        
        speech_spec = cv2.resize(speech_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
        rev_spec = cv2.resize(rev_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)

        print("Proccesing audio file n°: " + str(i+1))
        X[i-lower_bound, 0, :, :] = torch.tensor(rev_spec)
        y[i-lower_bound, 0, :, :] = torch.tensor(speech_spec)

        if ((i+1)%500 == 0):
          torch.save(X, checkpointX)
          torch.save(y, checkpointY)
          print('Saved data')

    return X, y

def test_data(audio_dir, rir_dir, lower_bound, upper_bound, checkpoints, noise = [15, 35]):

    """
    read test data generating reverberant spectrograms and waveforms
    
    audio_dir: directory containing speech audio files
    rir_dir: directory
    lower_bound: initial example to be considered
    upper_bound: final example to be considered
    checkpoints: list containing directories for save rev spectrogram, target spectrogram
                rev waveforms and target waveforms respectively
    noise: add noise with random snr in [noise[0], noise[1]]
    """
    
    checkpointX = checkpoints[0]
    checkpointY = checkpoints[1]
    checkpoint_waverev = checkpoints[2]
    checkpoint_wavetarget = checkpoints[3]
    
    sys.path.append(audio_dir)
    sys.path.append(rir_dir)
    
    rir_file_names = []
    for subdir, dirs, files in os.walk(rir_dir):
        for file in files:
            if (".wav" in file):
                rir_file_names.append(os.path.join(subdir,file))

    audio_file_names = []
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            if (".wav" in file):
                audio_file_names.append(os.path.join(subdir,file))
    
    print ("RIRs found: " + str(len(rir_file_names)))
    print ("Audio files found: " + str(len(audio_file_names)))
    time_size = 340
    frequency_size = 128
    X = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))
    y = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))

    wave_data = []
    wave_targets = []


    for i in range(lower_bound, upper_bound):
        rir_index = random.sample(range(len(rir_file_names)), 1)[0]
        ir_audio, ir_time, ir_rate = extract_audio(rir_file_names[rir_index])

        speech_audio, speech_time, speech_rate = extract_audio(audio_file_names[i])
        wave_targets.append(speech_audio)
        speech_spec = generate_spec(speech_audio, speech_rate)
        
        random_snr = random.sample(range(noise[0], noise[1]), 1)[0]
        speech_rev = discrete_conv(speech_audio, ir_audio, 16000, 96000, snr = random_snr)
        speech_rev = speech_rev[0:len(speech_audio)]
        wave_data.append(speech_rev)
        rev_spec = generate_spec(speech_rev, speech_rate)
        
        speech_spec = cv2.resize(speech_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
        rev_spec = cv2.resize(rev_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)

        print("Proccesing audio file n°: " + str(i+1))
        X[i-lower_bound, 0, :, :] = torch.tensor(rev_spec)
        y[i-lower_bound, 0, :, :] = torch.tensor(speech_spec)

        if ((i+1)%500 == 0):    
          torch.save(X, checkpointX)
          torch.save(y, checkpointY)
          torch.save(wave_data, checkpoint_waverev)
          torch.save(wave_targets, checkpoint_wavetarget)
          print('Saved data')

    return X, y

def mardy_test_data(audio_dir, rir_dir, lower_bound, upper_bound, checkpoints, snr = 30, distance = 'far'):

    """
    read test data generating reverberant spectrograms and waveforms
    
    audio_dir: directory containing speech audio files
    rir_dir: directory contaning MARDY RIRs
    lower_bound: initial example to be considered
    upper_bound: final example to be considered
    checkpoints: list containing directories for save rev spectrogram, target spectrogram
                rev waveforms and target waveforms respectively
    snr: add awgn with snr
    """
    
    checkpointX = checkpoints[0]
    checkpointY = checkpoints[1]
    checkpoint_waverev = checkpoints[2]
    checkpoint_wavetarget = checkpoints[3]
    
    sys.path.append(audio_dir)
    sys.path.append(rir_dir)

    num_distance = '3' if distance == 'far' else '1'
    print('Distance Microphones ' + num_distance)
    
    rir_file_names = []
    for subdir, dirs, files in os.walk(rir_dir):
        for file in files:
            if (".wav" in file and file[3]==num_distance):
                rir_file_names.append(os.path.join(subdir,file))

    audio_file_names = []
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            if (".flac" in file):
                audio_file_names.append(os.path.join(subdir,file))
    
    print ("RIRs found: " + str(len(rir_file_names)))
    print ("Audio files found: " + str(len(audio_file_names)))
    time_size = 340
    frequency_size = 128
    X = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))
    y = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))

    wave_data = []
    wave_targets = []


    for i in range(lower_bound, upper_bound):
        rir_index = random.sample(range(len(rir_file_names)), 1)[0]
        ir_audio, ir_time, ir_rate = extract_audio(rir_file_names[rir_index])

        speech_audio, speech_time, speech_rate = extract_audio(audio_file_names[i])
        wave_targets.append(speech_audio)
        speech_spec = generate_spec(speech_audio, speech_rate)
        speech_rev = discrete_conv(speech_audio, ir_audio, 16000, 48000, aug_factor = 10, snr = snr)
        speech_rev = speech_rev[0:len(speech_audio)]
        wave_data.append(speech_rev)
        rev_spec = generate_spec(speech_rev, speech_rate)
        
        speech_spec = cv2.resize(speech_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
        rev_spec = cv2.resize(rev_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)

        print("Proccesing audio file n°: " + str(i+1))
        X[i-lower_bound, 0, :, :] = torch.tensor(rev_spec)
        y[i-lower_bound, 0, :, :] = torch.tensor(speech_spec)

        if ((i+1)%500 == 0):
          torch.save(X, checkpointX)
          torch.save(y, checkpointY)
          torch.save(wave_data, checkpoint_waverev)
          torch.save(wave_targets, checkpoint_wavetarget)
          print('Saved data')

    return X, y

def realdata_from_dir(audio_dir, lower_bound, upper_bound, checkpointX, checkpoint_wave):
    """
    read real test data with reverberant spectrograms and waveforms
    
    audio_dir: directory containing speech audio files
    lower_bound: initial example to be considered
    upper_bound: final example to be considered
    checkpointX: directory + filename to save reverberant spectrograms
    checkpointX: directory + filename to save reverberant waveforms
    """
    
    sys.path.append(audio_dir)

    audio_file_names = []
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            if (".wav" in file):
                audio_file_names.append(os.path.join(subdir,file))

    print ("Archivos de audio encontrados: " + str(len(audio_file_names)))
    time_size = 340
    frequency_size = 128

    X = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))
    y = torch.zeros((upper_bound-lower_bound, 1, frequency_size, time_size))
    
    waves_rev = []
    for i in range(lower_bound, upper_bound):
        print(i)
        speech_rev, speech_time, speech_rate = extract_audio(audio_file_names[i])
        waves_rev.append(speech_rev)
        rev_spec = generate_spec(speech_rev, speech_rate)
        rev_spec = cv2.resize(rev_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)

        print("Procesado archivo de audio n°: " + str(i+1))
        X[i-lower_bound, 0, :, :] = torch.tensor(rev_spec)
  
        if ((i+1)%50 == 0):
          torch.save(X, checkpointX)
          torch.save(waves_rev, checkpoint_wave)
          print('Saved data')

    return X

def LSUnet_gen_specs():
    rir_rootdir =           '/home/may.tiger/AIProject/data/ClassroomOmni/'
    audio_rootdir =         '/home/may.tiger/AIProject/data/dry'
    checkpointX =           '/home/may.tiger/AIProject/data/our_data/X_test.pth'
    checkpointY =           '/home/may.tiger/AIProject/data/our_data/y_test.pth'
    checkpoint_waverev =    '/home/may.tiger/AIProject/data/our_data_2/waverev.pth'
    checkpoint_wavetarget = '/home/may.tiger/AIProject/data/our_data_2/wavetarget.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_waverev, checkpoint_wavetarget]

    print("starting test\n")
    dir_len = len([entry for entry in os.listdir("/home/may.tiger/AIProject/data/dry") if os.path.isfile(os.path.join("/home/may.tiger/AIProject/data/dry", entry))])
    X, y = test_data(audio_rootdir, rir_rootdir, 0, dir_len, checkpoints)

    # ---------------------------------------------------------------------------------------------

    rir_rootdir1 = 'GreatHallOmni/'
    audio_rootdir1 = 'LibriSpeechTrain/'
    checkpointX1 = 'data_audio/non_norm_data/X_train_1.pth'
    checkpointY1 = 'data_audio/non_norm_data/y_train_1.pth'
    _, _ = train_data(audio_rootdir1, rir_rootdir1, 0, 5000, checkpointX1, checkpointY1)

    rir_rootdir2 = 'OctagonOmni/'
    audio_rootdir2 = 'LibriSpeechTrain/'
    checkpointX2 = 'data_audio/non_norm_data/X_train_2.pth'
    checkpointY2 = 'data_audio/non_norm_data/y_train_2.pth'
    _, _ = train_data(audio_rootdir2, rir_rootdir2, 5000, 10000, checkpointX2, checkpointY2)

    rir_rootdir3 = 'GreatHallOmni/'
    audio_rootdir3 = 'LibriSpeechTrain/'
    checkpointX3 = 'data_audio/non_norm_data/X_train_3.pth'
    checkpointY3 = 'data_audio/non_norm_data/y_train_3.pth'
    _, _ = train_data(audio_rootdir3, rir_rootdir3, 10000, 15000, checkpointX3, checkpointY3)

    rir_rootdir4 = 'OctagonOmni/'
    audio_rootdir4 = 'LibriSpeechTrain/'
    checkpointX4 = 'data_audio/non_norm_data/X_train_4.pth'
    checkpointY4 = 'data_audio/non_norm_data/y_train_4.pth'
    _, _ = train_data(audio_rootdir4, rir_rootdir4, 15000, 20000, checkpointX4, checkpointY4)

    rir_rootdir = 'ClassroomOmni/'
    audio_rootdir = 'LibriSpeechTest/'
    checkpointX = 'data_audio/non_norm_data/X_test.pth'
    checkpointY = 'data_audio/non_norm_data/y_test.pth'
    checkpoint_waverev = 'data_audio/non_norm_data/waverev.pth'
    checkpoint_wavetarget = 'data_audio/non_norm_data/wavetarget.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_waverev, checkpoint_wavetarget]

    print("starting test\n")
    X, y = test_data(audio_rootdir, rir_rootdir, 0, 500, checkpoints)

    rir_rootdir = 'data_espec/MARDY'
    audio_rootdir = 'LibriSpeechTest/'
    checkpointX = 'data_audio/non_norm_data/X_test_2.pth'
    checkpointY = 'data_audio/non_norm_data/y_test_2.pth'
    checkpoint_waverev = 'data_audio/non_norm_data/waverev_2.pth'
    checkpoint_wavetarget = 'data_audio/non_norm_data/wavetarget_2.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_waverev, checkpoint_wavetarget]

    print("starting MARDY test\n")
    X, y = mardy_test_data(audio_rootdir, rir_rootdir, 500, 1000, checkpoints, snr = 30)

    rir_rootdir = 'data_espec/MARDY'
    audio_rootdir = 'LibriSpeechTest/'
    checkpointX = 'data_audio/non_norm_data/X_test_3.pth'
    checkpointY = 'data_audio/non_norm_data/y_test_3.pth'
    checkpoint_waverev = 'data_audio/non_norm_data/waverev_3.pth'
    checkpoint_wavetarget = 'data_audio/non_norm_data/wavetarget_3.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_waverev, checkpoint_wavetarget]

    X, y = mardy_test_data(audio_rootdir, rir_rootdir, 500, 1000, checkpoints, snr = 30, distance = 'near')

    rir_rootdir = 'ClassroomOmni/'
    audio_rootdir = 'LibriSpeechTest/'
    checkpointX = 'data_audio/non_norm_data/X_test_4.pth'
    checkpointY = 'data_audio/non_norm_data/y_test_4.pth'
    checkpoint_waverev = 'data_audio/non_norm_data/waverev_4.pth'
    checkpoint_wavetarget = 'data_audio/non_norm_data/wavetarget_4.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_waverev, checkpoint_wavetarget]

    X, y = test_data(audio_rootdir, rir_rootdir, 0, 500, checkpoints, noise = [35, 36])

    rir_rootdir = 'ClassroomOmni/'
    audio_rootdir = 'LibriSpeechTest/'
    checkpointX = 'data_audio/non_norm_data/X_test_5.pth'
    checkpointY = 'data_audio/non_norm_data/y_test_5.pth'
    checkpoint_waverev = 'data_audio/non_norm_data/waverev_5.pth'
    checkpoint_wavetarget = 'data_audio/non_norm_data/wavetarget_5.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_waverev, checkpoint_wavetarget]

    X, y = test_data(audio_rootdir, rir_rootdir, 0, 500, checkpoints, noise = [15, 16])

    # ---------------------------------------------------------------------------------------------

    audio_rootdir = 'masive_data/VUT_FIT_L207/MicID01/SpkID01_20171225_T/01/'
    checkpointX = 'real_data/X_test_real1.pth'
    checkpoint_wave = 'real_data/waves1.pth'

    X = realdata_from_dir(audio_rootdir, 0, 5, checkpointX, checkpoint_wave)

    audio_rootdir = 'masive_data/VUT_FIT_L207/MicID01/SpkID01_20171225_T/10/'
    checkpointX = 'real_data/X_test_real2.pth'
    checkpoint_wave = 'real_data/waves2.pth'

    X = realdata_from_dir(audio_rootdir, 0, 500, checkpointX, checkpoint_wave)
