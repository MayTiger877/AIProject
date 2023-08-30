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


def check_audio_rate():
    # data sets

    sample_1_audio, sample_1_time, sample_1_rate = extract_audio("/home/may.tiger/AIProject/data/LibriSpeechTrain/LibriSpeech/train-clean-100/19/198/19-198-0000.flac")
    # print(f"sample_1_audio is {sample_1_audio}")
    # print(f"sample_1_time is {sample_1_time}")
    print(f"sample_1_rate is {sample_1_rate}")

    sample_2_audio, sample_2_time, sample_2_rate = extract_audio("/home/may.tiger/AIProject/data/LibriSpeechTest/LibriSpeech/test-clean/61/70968/61-70968-0000.flac")
    # print(f"sample_2_audio is {sample_2_audio}")
    # print(f"sample_2_time is {sample_2_time}")
    print(f"sample_2_rate is {sample_2_rate}")

    sample_3_audio, sample_3_time, sample_3_rate = extract_audio("/home/may.tiger/AIProject/model_2/clean/clean_trainset_28spk_wav/p226_001.wav")
    # print(f"sample_3_audio is {sample_3_audio}")
    # print(f"sample_3_time is {sample_3_time}")
    print(f"sample_3_rate is {sample_3_rate}")

    sample_4_audio, sample_4_time, sample_4_rate = extract_audio("/home/may.tiger/AIProject/model_2/noisy/noisy_trainset_28spk_wav/p226_001.wav")
    # print(f"sample_4_audio is {sample_4_audio}")
    # print(f"sample_4_time is {sample_4_time}")
    print(f"sample_4_rate is {sample_4_rate}")

    sample_5_audio, sample_5_time, sample_5_rate = extract_audio("/home/may.tiger/AIProject/fine_tuning/fine_tune_data/clean_trainset_56spk_wav/p234_001.wav")
    # print(f"sample_5_audio is {sample_5_audio}")
    # print(f"sample_5_time is {sample_5_time}")
    print(f"sample_5_rate is {sample_5_rate}")

    sample_6_audio, sample_6_time, sample_6_rate = extract_audio("/home/may.tiger/AIProject/fine_tuning/fine_tune_data/noisy_trainset_56spk_wav/p234_001.wav")
    # print(f"sample_6_audio is {sample_6_audio}")
    # print(f"sample_6_time is {sample_6_time}")
    # print(f"sample_6_rate is {sample_6_rate}")

def downsample_data_audio_to_16():
    # a function to itrerate through the data set and downsample the audio files to 16kHz if they are not already
    for root, dirs, files in os.walk("/home/may.tiger/AIProject/model_2/clean/clean_trainset_28spk_wav"):
        for file in files:
            if file.endswith(".wav"):
                audio, time, rate = extract_audio(os.path.join(root, file))
                if rate != 16000:
                    audio = resample(audio, round(len(audio) / rate * 16000))
                    sf.write(os.path.join("/home/may.tiger/AIProject/model_2/clean_16Hz", file), audio, 16000)


    # same function for noisy data
    for root, dirs, files in os.walk("/home/may.tiger/AIProject/model_2/noisy/noisy_trainset_28spk_wav"):
        for file in files:
            if file.endswith(".wav"):
                audio, time, rate = extract_audio(os.path.join(root, file))
                if rate != 16000:
                    audio = resample(audio, round(len(audio) / rate * 16000))
                    sf.write(os.path.join("/home/may.tiger/AIProject/model_2/noisy_16Hz", file), audio, 16000)


    # same for fine tuning data
    for root, dirs, files in os.walk("/home/may.tiger/AIProject/fine_tuning/fine_tune_data/clean_trainset_56spk_wav"):
        for file in files:
            if file.endswith(".wav"):
                audio, time, rate = extract_audio(os.path.join(root, file))
                if rate != 16000:
                    audio = resample(audio, round(len(audio) / rate * 16000))
                    sf.write(os.path.join("/home/may.tiger/AIProject/fine_tuning/fine_tune_data/clean_16Hz", file), audio, 16000)


    # same for fine tuning data noisy
    for root, dirs, files in os.walk("/home/may.tiger/AIProject/fine_tuning/fine_tune_data/noisy_trainset_56spk_wav"):
        for file in files:
            if file.endswith(".wav"):
                audio, time, rate = extract_audio(os.path.join(root, file))
                if rate != 16000:
                    audio = resample(audio, round(len(audio) / rate * 16000))
                    sf.write(os.path.join("/home/may.tiger/AIProject/fine_tuning/fine_tune_data/noisyreverb_16Hz", file), audio, 16000)

def check_files_rate():
    # check on all files
    tester_1 = "/home/may.tiger/AIProject/data/LibriSpeechTest/LibriSpeechTrain/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
    tester_2 = "/home/may.tiger/AIProject/data/LibriSpeechTest/LibriSpeechTest/LibriSpeech/test-clean/61/70968/61-70968-0000.flac"
    tester_3 = "/home/may.tiger/AIProject/model_2/clean_16Hz"
    tester_4 = "/home/may.tiger/AIProject/model_2/noisy_16Hz"
    tester_5 = "/home/may.tiger/AIProject/fine_tuning/fine_tune_data/clean_16Hz"
    tester_6 = "/home/may.tiger/AIProject/fine_tuning/fine_tune_data/noisyreverb_16Hz"

    testers = [tester_1, tester_2, tester_3, tester_4, tester_5, tester_6]
    # print only 1 file sample rate from testers
    for tester in testers:
        for root, dirs, files in os.walk(tester):
            for file in files:
                if file.endswith(".wav"):
                    audio, time, rate = extract_audio(os.path.join(root, file))
                    print(f"directory is {root}")
                    print(f"file is {file}")
                    print(f"rate is {rate}")
                    break
            break

def check_specs_output():
    modle_1_clean_specs = "/home/may.tiger/AIProject/data/our_data/y_test.pth"
    modle_1_reverb_specs = "/home/may.tiger/AIProject/data/our_data/X_test.pth"
    model_2_clean_specs = "/home/may.tiger/AIProject/model_2/generateSpecs/cleanspecs.pth"
    model_2_noisy_specs = "/home/may.tiger/AIProject/model_2/generateSpecs/noisyspecs.pth"
    model_3_clean_specs = "/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/y_test.pth"
    model_3_noisyreverb_specs = "/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/X_test.pth"

    models = [modle_1_clean_specs, modle_1_reverb_specs, model_2_clean_specs, model_2_noisy_specs, model_3_clean_specs, model_3_noisyreverb_specs]
    real_spec = torch.load(modle_1_reverb_specs)[0]
    print(real_spec.dtype)
    for model in models:
        loaded = torch.load(model)
        #model_name = model.split("/")[-2]
        print(f"model is {model}")
        print(f"model shape is {loaded.shape}")
        print(f"model type is {type(loaded)}")
        print(f"model dtype is {loaded.dtype}\n")
        example_tensor_spec = loaded[0]
        example_tensor_spec = example_tensor_spec[0, :, :]
        example_wave = reconstruct_wave(example_tensor_spec.numpy(), 16000)
        example_spec = generate_spec(example_wave, 16000)
        graph_spec(example_spec, save_path=model)

def check_data_banks():
    modle_1_clean_specs = "/home/may.tiger/AIProject/data/our_data/y_test.pth"
    modle_1_reverb_specs = "/home/may.tiger/AIProject/data/our_data/X_test.pth"
    model_2_clean_specs = "/home/may.tiger/AIProject/model_2/generateSpecs/cleanspecs.pth"
    model_2_noisy_specs = "/home/may.tiger/AIProject/model_2/generateSpecs/noisyspecs.pth"
    model_3_clean_specs = "/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/y_test.pth"
    model_3_noisyreverb_specs = "/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/X_test.pth"

    models = [modle_1_clean_specs, modle_1_reverb_specs, model_2_clean_specs, model_2_noisy_specs, model_3_clean_specs, model_3_noisyreverb_specs]
    for model in models:
        print(torch.load(model).shape)

def check_models_output():
    LateSupUnet_model = LateSupUnet(n_channels=1, bilinear=False)
    LateSupUnet_model.load_state_dict(torch.load('/home/may.tiger/AIProject/data//home/may.tiger/AIProject/data/training/model/LateSupUnet_state_dict.pth', map_location=lambda storage, loc: storage))
    LateSupUnet_model.eval()
    
    DeNoiser_model = DeNoiseUNet(n_channels=1, bilinear=False)
    DeNoiser_model.load_state_dict(torch.load('/home/may.tiger/AIProject/de_noising/training/model/DeNoiser_state_dict.pth', map_location=lambda storage, loc: storage))
    DeNoiser_model.eval()
    
    FineTuning_model = FineTuningUNet(DeNoiser_model, LateSupUnet)
    FineTuning_model.load_state_dict(torch.load('/home/may.tiger/AIProject/fine_tuning/training/model/FineTuning_state_dict.pth', map_location=lambda storage, loc: storage))
    FineTuning_model.eval()

    example_audio = "/home/may.tiger/AIProject/big_data_set/fine_tune_data/test/noisy_reverb/p234_001.wav"
    example_audio, time, rate = extract_audio(example_audio)
    example_spec = generate_spec(example_audio, rate)
    
    LateSupUnet_output = LateSupUnet_model(example_spec)
    DeNoiser_output = DeNoiser_model(example_spec)
    FineTuning_output = FineTuning_model(example_spec)
    
    # save outputs as wav files
    LateSupUnet_output = reconstruct_wave(LateSupUnet_output.detach().numpy(), rate)
    DeNoiser_output = reconstruct_wave(DeNoiser_output.detach().numpy(), rate)
    FineTuning_output = reconstruct_wave(FineTuning_output.detach().numpy(), rate)
    
    # save outputs aduido files
    write("/home/may.tiger/AIProject/orchestrator/outputs/LateSupUnet_output.wav", rate, LateSupUnet_output)
    write("/home/may.tiger/AIProject/orchestrator/outputs/DeNoiser_output.wav", rate, DeNoiser_output)
    write("/home/may.tiger/AIProject/orchestrator/outputs/FineTuning_output.wav", rate, FineTuning_output)
    
    # save outputs as spectrograms
    LateSupUnet_output_spec = generate_spec(LateSupUnet_output, rate)
    DeNoiser_output_spec = generate_spec(DeNoiser_output, rate)
    FineTuning_output_spec = generate_spec(FineTuning_output, rate)
    
def divide_and_move_data():
    clean_dir = "/home/may.tiger/AIProject/fine_tune_data/clean_16Hz"
    noisy_dir = "/home/may.tiger/AIProject/fine_tune_data/noisyreverb_16Hz"

def add_noise_to_data():
    clean_dir = "/home/may.tiger/AIProject/big_data_set/denoise_extra/dry"
    noisy_dir = "/home/may.tiger/AIProject/big_data_set/denoise_extra/wet"
    clean_files = os.listdir(clean_dir)
    noisy_files = os.listdir(noisy_dir)
    for i in tqdm(range(len(clean_files))):
        clean_file = clean_files[i]
        clean_path = os.path.join(clean_dir, clean_file)
        clean_data, clean_time, clean_rate = extract_audio(clean_path)
        
        noise_length = len(clean_data)
        noise = np.random.randn(noise_length).astype(np.float32)
        noise *= (clean_data.max() / np.abs(noise.max()))
        noise *= 0.25
        noisy = clean_data + noise
        noisy_path = os.path.join(noisy_dir, clean_file)
        print(noisy_path)
        sf.write(noisy_path, noisy, clean_rate)
        
def add_reverb_to_data():
    clean_dir = "/home/may.tiger/AIProject/big_data_set/denoise_extra/dry"
    noisy_dir = "/home/may.tiger/AIProject/big_data_set/denoise_extra/wet"
    clean_files = os.listdir(clean_dir)
    noisy_files = os.listdir(noisy_dir)
    for i in range(len(clean_files)):
        clean_file = clean_files[i]
        clean_path = os.path.join(clean_dir, clean_file)
        clean_data, clean_time, clean_rate = extract_audio(clean_path)
        
        noisy_file = noisy_files[i]
        noisy_path = os.path.join(noisy_dir, noisy_file)
        noisy_data, noisy_time, noisy_rate = extract_audio(noisy_path)
        
        reverbed_noisy_data = add_reverb(clean_data, clean_rate)
        reverbed_noisy_path = os.path.join(noisy_dir, clean_file)
        print(reverbed_noisy_path)
        sf.write(reverbed_noisy_path, reverbed_noisy_data, clean_rate)

def augment_data():    
    clean_dir = "/home/may.tiger/AIProject/data/LibriSpeechTrain/LibriSpeech/train-clean-100"
    noises_dir = "/home/may.tiger/AIProject/big_data_set/denoise_extra_2/UrbanSound/data"
    rir_dir = "/home/may.tiger/AIProject/data/ClassroomOmni"
    
    clean_file_names = []
    clean_file_paths = []
    noise_file_names = []
    noise_file_paths = []
    rir_file_names = []
    
    for root, dirs, files in os.walk(clean_dir):
        for file in files:
            if (file.endswith(".wav") or file.endswith(".flac")):
                clean_file_paths.append(os.path.join(root, file))
                clean_file_names.append(file)
    
    for root, dirs, files in os.walk(noises_dir):
        for file in files:
            if (file.endswith(".wav") or file.endswith(".flac")):
                noise_file_paths.append(os.path.join(root, file))
                noise_file_names.append(file)
                             
    for subdir, dirs, files in os.walk(rir_dir):
        for file in files:
            if (".wav" in file):
                rir_file_names.append(os.path.join(subdir,file))
                
    # data for denoising
    for i in tqdm(range(10000)):
        clean_data, clean_time, clean_rate = extract_audio(clean_file_paths[i])
        random_noise = random.sample(range(len(noise_file_paths)), 1)[0]
        random_noise_data, random_noise_time, random_noise_rate = extract_audio(noise_file_paths[random_noise])
        
        if clean_rate != 16000:
            clean_data = librosa.resample(clean_data, clean_rate, 16000)
        if random_noise_rate != 16000:
            random_noise_data = librosa.resample(random_noise_data, orig_sr=random_noise_rate, target_sr=16000)
                                                
            
        if (len(random_noise_data) > len(clean_data)):
            random_noise_data = random_noise_data[:len(clean_data)]
        else:
            while (len(random_noise_data) < len(clean_data)):
                random_noise_data = np.concatenate((random_noise_data, random_noise_data) , axis=0)
            random_noise_data = random_noise_data[:len(clean_data)]
        
        random_noise_data *= (clean_data.max() / np.abs(random_noise_data.max()))
        random_noise_data *= 0.25
        noisy_data = clean_data + random_noise_data
        
        clean_path = os.path.join("/home/may.tiger/AIProject/big_data_set/denoise_extra_2/dry", clean_file_names[i])
        noisy_path = os.path.join("/home/may.tiger/AIProject/big_data_set/denoise_extra_2/wet", clean_file_names[i])

        if (clean_path.endswith(".flac")):
            clean_path = clean_path.replace(".flac", ".wav")
        if (noisy_path.endswith(".flac")):
            noisy_path = noisy_path.replace(".flac", ".wav")
        
        sf.write(clean_path, clean_data, clean_rate)
        sf.write(noisy_path, noisy_data, clean_rate)
         
    # data for fine tuning
    for i in tqdm(range(10000, 20000)):
        clean_data, clean_time, clean_rate = extract_audio(clean_file_paths[i])
        random_noise = random.sample(range(len(noise_file_paths)), 1)[0]
        random_noise_data, random_noise_time, random_noise_rate = extract_audio(noise_file_paths[random_noise])
        
        if clean_rate != 16000:
            clean_data = librosa.resample(clean_data, clean_rate, 16000)
        if random_noise_rate != 16000:
            random_noise_data = librosa.resample(random_noise_data, orig_sr=random_noise_rate, target_sr=16000)
                                                  
        if (len(random_noise_data) > len(clean_data)):
            random_noise_data = random_noise_data[:len(clean_data)]
        else:
            while (len(random_noise_data) < len(clean_data)):
                random_noise_data = np.concatenate((random_noise_data, random_noise_data) , axis=0)
            random_noise_data = random_noise_data[:len(clean_data)]
        
        random_noise_data *= (clean_data.max() / np.abs(random_noise_data.max()))
        random_noise_data *= 0.25
        noisy_data = clean_data + random_noise_data
        
        clean_path = os.path.join("/home/may.tiger/AIProject/big_data_set/fine_tune_data_2/dry", clean_file_names[i])
        noisy_reverbed_path = os.path.join("/home/may.tiger/AIProject/big_data_set/fine_tune_data_2/wet", clean_file_names[i])

        if (clean_path.endswith(".flac")):
            clean_path = clean_path.replace(".flac", ".wav")
        if (noisy_reverbed_path.endswith(".flac")):
            noisy_reverbed_path = noisy_reverbed_path.replace(".flac", ".wav")
        
        #reverberate noisy data
                    
        rir_index = random.sample(range(len(rir_file_names)), 1)[0]
        ir_audio, ir_time, ir_rate = extract_audio(rir_file_names[rir_index])
        noisy_reverb_data = discrete_conv(noisy_data, ir_audio, 16000, ir_rate)
        noisy_reverb_data = noisy_reverb_data[0:len(noisy_data)]
        
        sf.write(clean_path, clean_data, clean_rate)
        sf.write(noisy_reverbed_path, noisy_reverb_data, 16000)
        
def present_outputs():
    dry_example =           "/home/may.tiger/AIProject/big_data_set/fine_tune_data/test/clean/p234_008.wav"
    noisy_example =         "/home/may.tiger/AIProject/big_data_set/fine_tune_data/test/noisy_only/p234_008.wav"
    revereb_noisy_example = "/home/may.tiger/AIProject/big_data_set/fine_tune_data/test/noisy_reverb/p234_008.wav" 
    # extract data
    dry_example_data, dry_example_time, dry_example_rate = extract_audio(dry_example)
    noisy_example_data, noisy_example_time, noisy_example_rate = extract_audio(noisy_example)
    noisy_reverb_example_data, noisy_reverb_example_time, noisy_reverb_example_rate = extract_audio(revereb_noisy_example)
    # make reverbed example
    rir = "/home/may.tiger/AIProject/data/GreatHallOmni/Omni/x01y00.wav"
    ir_audio, ir_time, ir_rate = extract_audio(rir)
    reverbed_example = discrete_conv(dry_example_data, ir_audio, 16000, ir_rate)
    reverbed_example = reverbed_example[0:len(dry_example_data)]
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/p234_008_reverb.wav', reverbed_example, 16000)
    reverbed_example_data, reverbed_example_time, reverbed_example_rate = extract_audio('/home/may.tiger/AIProject/orchestrator/outputs/p234_008_reverb.wav')
    # generate spectrograms
    dry_spec = generate_spec(dry_example_data, dry_example_rate)
    noisy_spec = generate_spec(noisy_example_data, noisy_example_rate)
    reverbed_spec = generate_spec(reverbed_example_data, reverbed_example_rate)
    noisy_reverb_spec = generate_spec(noisy_reverb_example_data, noisy_reverb_example_rate)
    # graph spectrograms
    graph_spec(dry_spec, title="dry_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/dry_spec')
    graph_spec(noisy_spec, title="noisy_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/noisy_spec')
    graph_spec(reverbed_spec, title="reverbed_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/reverbed_spec')
    graph_spec(noisy_reverb_spec, title="noisy_reverb_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/noisy_reverb_spec')
    # load models
    LateSupUNetModel = LateSupUnet(n_channels=1, bilinear=False).cuda()
    DeNoiserModel = DeNoiseUNet(n_channels=1, bilinear=False).cuda()
    model_FineTuning = FineTuningUNet(LateSupUNetModel, DeNoiserModel)
    model_FineTuning.load_state_dict(torch.load('/home/may.tiger/AIProject/fine_tuning/training/model/FineTuning_state_dict.pth', map_location=lambda storage, loc: storage))
    model_FineTuning.eval()
    LateSupUNetModel.load_state_dict(torch.load('/home/may.tiger/AIProject/data/training/model/LateSupUnet_state_dict.pth', map_location=lambda storage, loc: storage))
    LateSupUNetModel.eval()
    DeNoiserModel.load_state_dict(torch.load('/home/may.tiger/AIProject/de_noising/training/model/DeNoiser_state_dict.pth', map_location=lambda storage, loc: storage))
    DeNoiserModel.eval()
    # prepare for models input
    time_size = 340
    frequency_size = 128

    # dry example to hear the cv resize effect
    graph_spec(dry_spec, title="de_dry_spec_not_cvsized", save_path='/home/may.tiger/AIProject/orchestrator/outputs/de_dry_spec_not_cvsized')
    de_dry_wav_not_cvsized = reconstruct_wave(dry_spec, dry_example_rate)
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/de_dry_spec_not_cvsized.wav', de_dry_wav_not_cvsized, 16000)
    
    # save original spectrograms size
    dry_spec_original_size = dry_spec.shape
    
    dry_spec =          cv2.resize(dry_spec,          dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
    noisy_spec =        cv2.resize(noisy_spec,        dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
    reverbed_spec =     cv2.resize(reverbed_spec,     dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
    noisy_reverb_spec = cv2.resize(noisy_reverb_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
    
    noisy_spec = noisy_spec.reshape((1, noisy_spec.shape[0], noisy_spec.shape[1]))
    reverbed_spec = reverbed_spec.reshape((1, reverbed_spec.shape[0], reverbed_spec.shape[1]))
    noisy_reverb_spec = noisy_reverb_spec.reshape((1, noisy_reverb_spec.shape[0], noisy_reverb_spec.shape[1]))
    
    graph_spec(dry_spec, title="de_dry_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/de_dry_spec')
    
    net_input = torch.zeros((1, 1, noisy_spec.shape[1], noisy_spec.shape[2]))
    net_input[0, :, :, :] = torch.from_numpy(noisy_spec)
    de_noised_spec = DeNoiserModel(net_input.cuda())
    graph_spec(de_noised_spec[0, 0, :, :].cpu().detach().numpy(), title="de_noised_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/de_noised_spec')
    
    net_input = torch.zeros((1, 1, reverbed_spec.shape[1], reverbed_spec.shape[2]))
    net_input[0, :, :, :] = torch.from_numpy(reverbed_spec)
    de_reverbed_spec = LateSupUNetModel(net_input.cuda())
    graph_spec(de_reverbed_spec[0, 0, :, :].cpu().detach().numpy(), title="de_reverbed_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/de_reverbed_spec')
    
    net_input = torch.zeros((1, 1, noisy_reverb_spec.shape[1], noisy_reverb_spec.shape[2]))
    net_input[0, :, :, :] = torch.from_numpy(noisy_reverb_spec)
    de_noisy_de_reverbed_spec = model_FineTuning(net_input.cuda())
    graph_spec(de_noisy_de_reverbed_spec[0, 0, :, :].cpu().detach().numpy(), title="de_noisy_de_reverbed_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/de_noisy_de_reverbed_spec')
    
    # graph after cv resize
    resized_cv_de_noised_spec = de_noised_spec[0, 0, :, :].cpu().detach().numpy()
    resized_cv_de_reverbed_spec = de_reverbed_spec[0, 0, :, :].cpu().detach().numpy()
    resized_cv_de_noisy_de_reverbed_spec = de_noisy_de_reverbed_spec[0, 0, :, :].cpu().detach().numpy()
    resized_cv_de_noised_spec = cv2.resize(resized_cv_de_noised_spec, dsize = (dry_spec_original_size[1], dry_spec_original_size[0]), interpolation = cv2.INTER_LANCZOS4)
    resized_cv_de_reverbed_spec = cv2.resize(resized_cv_de_reverbed_spec, dsize = (dry_spec_original_size[1], dry_spec_original_size[0]), interpolation = cv2.INTER_LANCZOS4)
    resized_cv_de_noisy_de_reverbed_spec = cv2.resize(resized_cv_de_noisy_de_reverbed_spec, dsize = (dry_spec_original_size[1], dry_spec_original_size[0]), interpolation = cv2.INTER_LANCZOS4)
    graph_spec(resized_cv_de_noised_spec, title="resized_cv_de_noised_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/resized_cv_de_noised_spec')
    graph_spec(resized_cv_de_reverbed_spec, title="resized_cv_de_reverbed_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/resized_cv_de_reverbed_spec')
    graph_spec(resized_cv_de_noisy_de_reverbed_spec, title="resized_cv_de_noisy_de_reverbed_spec", save_path='/home/may.tiger/AIProject/orchestrator/outputs/resized_cv_de_noisy_de_reverbed_spec')
    
    # save wav files
    de_dry_wav = reconstruct_wave(dry_spec, dry_example_rate)
    de_noised_wav = reconstruct_wave(de_noised_spec[0, 0, :, :].cpu().detach().numpy(), noisy_example_rate)
    de_reverbed_wav = reconstruct_wave(de_reverbed_spec[0, 0, :, :].cpu().detach().numpy(), reverbed_example_rate)
    de_noisy_de_reverbed_wav = reconstruct_wave(de_noisy_de_reverbed_spec[0, 0, :, :].cpu().detach().numpy(), noisy_reverb_example_rate)
    resized_cv_de_noised_wav = reconstruct_wave(resized_cv_de_noised_spec, dry_example_rate)
    resized_cv_de_reverbed_wav = reconstruct_wave(resized_cv_de_reverbed_spec, dry_example_rate)
    resized_cv_de_noisy_de_reverbed_wav = reconstruct_wave(resized_cv_de_noisy_de_reverbed_spec, dry_example_rate)
    
    #save all wav files
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/de_dry_audio.wav', de_dry_wav, dry_example_rate)
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/de_noised_audio.wav', de_noised_wav, noisy_example_rate)
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/de_reverbed_audio.wav', de_reverbed_wav, reverbed_example_rate)
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/de_noisy_de_reverbed_audio.wav', de_noisy_de_reverbed_wav, noisy_reverb_example_rate)
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/resized_cv_de_noised_audio.wav', resized_cv_de_noised_wav, dry_example_rate)
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/resized_cv_de_reverbed_audio.wav', resized_cv_de_reverbed_wav, dry_example_rate)
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/resized_cv_de_noisy_de_reverbed_audio.wav', resized_cv_de_noisy_de_reverbed_wav, dry_example_rate)
    
    cutoff_freq = 7000
    filtered_waveform = librosa.effects.preemphasis(resized_cv_de_reverbed_wav, coef=cutoff_freq/16000)
    sf.write('/home/may.tiger/AIProject/orchestrator/outputs/de_dry_audio_filtered.wav', filtered_waveform, dry_example_rate)

    
    
#----------------------------------- activating functions --------------------------------------

# # LSU_model 
# LSUnet_train()
LSUnet_eval()

# # DeNoiser_model
# DENoise_train()
# DENoise_train_extra()
# DENoise_train_extra_2()
DENoise_eval()

# # FineTuning_model
# FineTuning_train()
# FineTuning_train_2()
FineTuning_eval()

''' helper functions '''

# FineTuning_srmr_test()

# add_noise_to_data()

# present_outputs()

# augment_data()