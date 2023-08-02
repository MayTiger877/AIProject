import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display as display
import librosa.feature
from scipy.signal import resample
import os
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

#########################################################################################

def normalize_data(X_path, y_path, checkpoints):  
  checkpointX = checkpoints[0]
  checkpointY = checkpoints[1]
  X = torch.load(X_path)
  y = torch.load(y_path)

  for i in range(X.shape[0]):
    print("Processing specgram n°:" + str(i+1))
    spec_rev = X[i, 0, :, :].numpy()
    spec_target = y[i, 0, :, :].numpy()

    spec_rev_norm, _ = normalize(spec_rev)
    spec_target_norm, _ = normalize(spec_target)

    X[i, 0, :, :] = torch.tensor(spec_rev_norm)
    y[i, 0, :, :] = torch.tensor(spec_target_norm)
    
  torch.save(X, checkpointX)
  torch.save(y, checkpointY)
  print('Saved data')

def LSUnet_norm_specs():
  X_path = 'data_audio/non_norm_data/X_train_1.pth'
  y_path = 'data_audio/non_norm_data/y_train_1.pth'
  checkpoints = ['data_audio/normalized_data/X_train_n1.pth', 
                 'data_audio/normalized_data/y_train_n1.pth']
  normalize_data(X_path, y_path, checkpoints)
   
  X_path = 'data_audio/non_norm_data/X_train_2.pth'
  y_path = 'data_audio/non_norm_data/y_train_2.pth'
  checkpoints = ['data_audio/normalized_data/X_train_n2.pth', 
                 'data_audio/normalized_data/y_train_n2.pth']
  normalize_data(X_path, y_path, checkpoints)
   
  X_path = 'data_audio/non_norm_data/X_train_3.pth'
  y_path = 'data_audio/non_norm_data/y_train_3.pth'
  checkpoints = ['data_audio/normalized_data/X_train_n3.pth', 
                 'data_audio/normalized_data/y_train_n3.pth']
  normalize_data(X_path, y_path, checkpoints)
   
  X_path = 'data_audio/non_norm_data/X_train_4.pth'
  y_path = 'data_audio/non_norm_data/y_train_4.pth'
  checkpoints = ['data_audio/normalized_data/X_train_n4.pth', 
                 'data_audio/normalized_data/y_train_n4.pth']
  normalize_data(X_path, y_path, checkpoints)
   
  X_path = '/home/may.tiger/AIProject/data/our_data/X_test.pth'
  y_path = '/home/may.tiger/AIProject/data/our_data/y_test.pth'
  checkpoints = ['/home/may.tiger/AIProject/data/our_norm_data/X_train_norm.pth', 
                 '/home/may.tiger/AIProject/data/our_norm_data/y_train_norm.pth']
  normalize_data(X_path, y_path, checkpoints)