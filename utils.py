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
from pysepm import llr, cepstrum_distance, fwSNRseg 
from pystoi import stoi
from pypesq import pesq

'''
################################# LateSupUnet #################################
class LateSupUnet(nn.Module):
	"""
	Late Reverberation Supression U-net
	"""
	def __init__(self, n_channels=1, bilinear=True):
		super(LateSupUnet, self).__init__()
		self.n_channels = n_channels
		self.bilinear = bilinear
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.outc = OutConv(64, 1)

	def forward(self, input):
		x1 = self.inc(input)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		output = input - x
		return output

################################# DeNoiseUNet #################################
class DeNoiseUNet(nn.Module):
    """
    Late Reverberation Supression U-net
    """
    def __init__(self, n_channels, bilinear=True, confine = True):
        super(DeNoiseUNet, self).__init__()
        self.confine = confine
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        output = self.tanh(x) if self.confine else x
        return x

################################# FineTuningUNet #################################
class FineTuningUNet(nn.Module):
    """
    Noise + Late Reverberation Supression U-net
    """
    def __init__(self, denoise_unet, late_sup_unet):
        super(FineTuningUNet, self).__init__()
        self.denoise_unet = denoise_unet
        self.late_sup_unet = late_sup_unet

    def forward(self, input):
        output_1 = self.denoise_unet(input)
        combined_output = self.late_sup_unet(output_1)
        return combined_output

'''
#####################################################################################
#                                     helper functions                              #                     
#####################################################################################

def weights_init(m):
	"""
	Initialise weights of the model.
	"""
	if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif (type(m) == nn.BatchNorm2d):
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 64, (7, 7), 2, 1)

		self.conv2 = nn.Conv2d(64, 128, (7, 7), 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)

		self.conv3 = nn.Conv2d(128, 256, (7, 7), 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(256)

		self.conv4 = nn.Conv2d(256, 512, (7, 7), 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(512)

		self.conv5 = nn.Conv2d(512, 1024, (7, 7), 2, 1, bias=False)
		self.bn5 = nn.BatchNorm2d(1024)

		self.fc9 = nn.Linear(7168, 1)
		# self.sig = nn.Sigmoid()

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), 0.1)
		x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
		x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
		x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
		x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1)

		x = torch.reshape(x, (x.shape[0], -1))
		x = F.leaky_relu(self.fc9(x), 0.1)
		# x = self.sig(x)
		return x

class DNN(nn.Module):
    """
    Adaptive implementation of MLP for speech dereverberation
    """

    def __init__(self, neural_layers, act_fun):
        """
        neural_layers: list with number of units in each layer
        act_fun: list with activation functions

        Example: neural network with 128 input units, 1000 hidden units and 1 output unit,
                 ReLU activation function and Identity output activation function

                 net = DNN([128, 1000, 1], [nn.ReLU(), nn.Identity()])
        """
        super(DNN, self).__init__()
        self.layers = nn.Sequential()

        if len(neural_layers) < 2:
            print('len(neural_layes) must be higher than 2')
            return
        for i in range(len(neural_layers) - 1):
            self.layers.add_module('layer_{}'.format(i + 1),
                                   nn.Linear(neural_layers[i],
                                             neural_layers[i + 1])
                                   )
            self.layers.add_module('act_fun_{}'.format(i + 1),
                                   act_fun[i])

    def forward(self, x):
        x = self.layers(x)
        return x

class LSTMDNN(nn.Module):
    """
    LSTM based dereverberation
    """

    def __init__(self):
        super(LSTMDNN, self).__init__()
        self.lstm1 = nn.LSTM(128 * 11, 512)
        self.lstm2 = nn.LSTM(512, 512)
        self.linear = nn.Linear(512, 128)

    def forward(self, x):
        x, _ = self.lstm1(x.view(x.shape[0], 1, x.shape[1]))
        x, _ = self.lstm2(x)
        x = self.linear(x[:, 0, :])
        return x

class SupLSTM(nn.Module):
    """
    Late supression dereverberation using LSTM
    """

    def __init__(self):
        super(SupLSTM, self).__init__()
        self.lstm1 = nn.LSTM(128, 512)
        self.lstm2 = nn.LSTM(512, 512)
        self.linear = nn.Linear(512, 128)

    def forward(self, input):
        x, _ = self.lstm1(input.view(input.shape[0], 1, input.shape[1]))
        x, _ = self.lstm2(x)
        x = self.linear(x[:, 0, :])
        output = input - x
        return output

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # adding residual connection
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class Up_NoResiduals(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = torch.zeros_like(x2)   # adding residual connection set to zero
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ReverbDataset(Dataset):
    """
    Reverberation dataset
    """

    def __init__(self, X, y):
        """
        X: (# examples, 1, 128, 340) tensor containing reverberant spectrograms
        y: (# examples, 1, 128, 340) tensor containing target spectrograms
        """

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def extract_audio(filename):
    """
    Extract audio given the filename (.wav, .flac, etc format)
    """

    audiotensor, rate = torchaudio.load(filename)
    audio = audiotensor.numpy()
    audio = np.reshape(audio, (1, -1))
    audio = audio[0]
    time = np.linspace(0, len(audio)/rate, len(audio), endpoint=False)
    return audio, time, rate

def generate_spec(audio_sequence, rate, n_fft=2048, hop_length=512):
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

def reconstruct_wave(spec, rate=16000, normalize_data=False):
    """
    Reconstruct waveform
    spec: spectrogram generated using Librosa
    rate: sampling rate
    """
    power = librosa.db_to_power(spec, ref=5.0)
    audio = librosa.feature.inverse.mel_to_audio(power, sr=rate, n_fft=2048, hop_length=512)
    out_audio = audio / np.max(audio) if normalize_data else audio
    return out_audio

def normalize(spec, eps=1e-6):
    """
    Normalize spectrogram with zero mean and unitary variance
    spec: spectrogram generated using Librosa
    """

    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    return spec_norm, (mean, std)

def minmax_scaler(spec):
    """
    min max scaler over spectrogram
    """
    spec_max = np.max(spec)
    spec_min = np.min(spec)

    return (spec-spec_min)/(spec_max - spec_min), (spec_max, spec_min)

def linear_scaler(spec):
    """
    linear scaler over spectrogram
    min value -> -1 and max value -> 1
    """
    spec_max = np.max(spec)
    spec_min = np.min(spec)
    m = 2/(spec_max-spec_min)
    n = (spec_max + spec_min)/(spec_min-spec_max)

    return m*spec + n, (m, n)

def split_specgram(example, clean_example, frames = 11):
    """
    Split specgram in groups of frames, the purpose is prepare data for the LSTM model input

    example: reverberant spectrogram
    clean_example: clean or target spectrogram

    return data input to the LSTM model and targets
    """
    clean_spec = clean_example[0, :, :]
    rev_spec = example[0, :, :]

    n, m = clean_spec.shape

    targets = torch.zeros((m-frames+1, n))
    data = torch.zeros((m-frames+1, n*frames))
  
    idx_target = frames//2
    for i in range(m-frames+1):
    	try:
    		targets[i, :] = clean_spec[:, idx_target]
    		data[i, :] = torch.reshape(rev_spec[:, i:i+frames], (1, -1))[0, :]
    		idx_target += 1
    	except (IndexError):
    		pass
    return data, targets

def split_realdata(example, frames = 11):
    
    """
    Split 1 specgram in groups of frames, the purpose is prepare data for the LSTM and MLP model input

    example: reverberant ''real'' (not simulated) spectrogram

    return data input to the LSTM or MLP model 
    """
  
    rev_spec = example[0, :, :]
    n, m = rev_spec.shape
    data = torch.zeros((m-frames+1, n*frames))
    for i in range(m-frames+1):
    	data[i, :] = torch.reshape(rev_spec[:, i:i+frames], (1, -1))[0, :]
    return data

def prepare_data(X, y, display = False):

    """
    Use split_specgram to split all specgrams
    X: tensor containing reverberant spectrograms
    y: tensor containing target spectrograms
    """

    data0, target0 = split_specgram(X[0, :, :, :], y[0, :, :, :])

    total_data = data0.cuda()
    targets = target0.cuda()
  
    for i in range(1, X.shape[0]):
   	    if display: 
   	    	print("Specgram n°" + str(i)) 

   	    data_i, target_i = split_specgram(X[i, :, :, :], y[i, :, :, :])
   	    total_data = torch.cat((total_data, data_i.cuda()), 0)
   	    targets = torch.cat((targets, target_i.cuda()), 0)

    return  total_data, targets

def split_for_supression(rev_tensor, target_tensor):
    """
    Given reverberant and target tensor with shape (#examples, 1, 128, 340)
    return tensors with the same information, but with shape (#examples*340, 128)
    """
    rev_transform = torch.tensor([])
    target_transform = torch.tensor([])

    for example in range(rev_tensor.shape[0]):
    	rev_transform = torch.cat((rev_transform, rev_tensor[example, 0, :, :].T))
    
    if (target_tensor!=None):
    	for example in range(target_tensor.shape[0]):
    		target_transform = torch.cat((target_transform, target_tensor[example, 0, :, :].T))
  
    return rev_transform, target_transform

def normalize_per_frame(spec_transpose):
    """
    Normalize over spectrogram rows
    """
    means = []
    stds = []
    norm_spec = torch.zeros(spec_transpose.shape)

    for spec_row in range(norm_spec.shape[0]):
    	current_mean = spec_transpose[spec_row, :].mean()
    	current_std = spec_transpose[spec_row, :].std()
    	means.append(current_mean)
    	stds.append(current_std)
    	norm_spec[spec_row, :] = (spec_transpose[spec_row, :]- current_mean)/(current_std+1e-6) 
  
    return norm_spec, (means, stds)

def denormalize_per_frame(norm_spec_transpose, means, stds):
    """
    denormalize row by row using means and stds given by normalize_per_frame
    """
    denorm_spec = torch.zeros(norm_spec_transpose.shape)

    for spec_row in range(norm_spec_transpose.shape[0]):
    	denorm_spec[spec_row, :] = (norm_spec_transpose[spec_row, :])*(stds[spec_row] + 1e-6) + means[spec_row]
    
    return denorm_spec.T

def zero_pad(x, k):
    """
    add k zeros to x signal
    """
    return np.append(x, np.zeros(k))

def awgn(signal, regsnr):
    """
    add random noise to signal
    regsnr: signal to noise ratio
    """
    sigpower = sum([math.pow(abs(signal[i]), 2) for i in range(len(signal))])
    sigpower = sigpower / len(signal)
    noisepower = sigpower / (math.pow(10, regsnr / 10))
    sample = np.random.normal(0, 1, len(signal))
    noise = math.sqrt(noisepower) * sample
    return noise

def discrete_conv(x, h, x_fs, h_fs, snr=30, aug_factor=1):
    """
    Convolution using fft
    x: speech waveform
    h: RIR waveform
    x_fs: speech signal sampling rate (if is not 16000 the signal will be resampled)
    h_fs: RIR signal sampling rate (if is not 16000 the signal will be resampled)

    Based on https://github.com/vtolani95/convolution/blob/master/reverb.py
    """

    numSamples_h = round(len(h) / h_fs * 16000)
    numSamples_x = round(len(x) / x_fs * 16000)

    if h_fs != 16000:
        h = resample(h, numSamples_h) # resample RIR

    if x_fs != 16000:
        x = resample(x, numSamples_x) # resample speech signal

    L, P = len(x), len(h)
    h_zp = zero_pad(h, L - 1)
    x_zp = zero_pad(x, P - 1)
    X = np.fft.fft(x_zp)
    output = np.fft.ifft(X * np.fft.fft(h_zp)).real
    output = aug_factor * output + x_zp
    output = output + awgn(output, snr)
    return output

def graph_spec(spec, rate=16000, title="Log-Power spectrogram", save_path="./spec"):
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

def plot_time_wave(audio, counter, rate=16000):
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
    plt.savefig(f"plots_2/{counter}.jpg")


#####################################################################################
#                                     project data                                  #                     
#####################################################################################
# region Data_Download
"""
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xf train-clean-100.tar.gz
mv LibriSpeech/ LibriSpeechTrain/

wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xf test-clean.tar.gz
mv LibriSpeech/ LibriSpeechTest/

wget isophonics.net/files/irs/classroomOmni.zip
unzip classroomOmni
mv Omni/ ClassroomOmni/

wget isophonics.net/files/irs/octagonOmni.zip
unzip octagonOmni
mv Omni/ OctagonOmni/

wget isophonics.net/files/irs/greathallOmni.zip
unzip greathallOmni
mv Omni/ GreatHallOmni/

wget https://www.openslr.org/resources/83/southern_english_female.zip
tar -xf train-clean-100.tar.gz
mv LibriSpeech/ LibriSpeechTrain/



ls
"""
#endregion
