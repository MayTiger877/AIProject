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


def check_librosa_wav_to_spec():
    """ """
    # Load audio file
    dry_example =       "/home/may.tiger/AIProject/librosa_experiment/f0001_us_f0001_00008_dry.wav"
    revereb_example =   "/home/may.tiger/AIProject/librosa_experiment/f0001_us_f0001_00008_wet.wav" 
    
    # extract data
    dry_example_data, dry_example_time, dry_example_rate = extract_audio(dry_example)
    revereb_example_data, revereb_example_time, revereb_example_rate = extract_audio(revereb_example)
    
    # generate spectrograms
    dry_spec = generate_spec(dry_example_data, dry_example_rate)
    reverb_spec = generate_spec(revereb_example_data, revereb_example_rate)
    
    # graph spectrograms
    graph_spec(dry_spec,    dry_example_rate,       "Dry_Spec_before")
    graph_spec(reverb_spec, revereb_example_rate,   "Wet_Spec_before")
    
    # reconstruct wav files
    reconstruct_dry = reconstruct_wave(dry_spec, dry_example_rate)
    reconstruct_wet = reconstruct_wave(reverb_spec, revereb_example_rate)
    
    # save wav files
    sf.write("reconstruct_dry.wav", reconstruct_dry, dry_example_rate)
    sf.write("reconstruct_wet.wav", reconstruct_wet, revereb_example_rate)
    
check_librosa_wav_to_spec()