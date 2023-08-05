import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import librosa
import librosa.display as display
import soundfile as sf
import os
import cv2
import glob
from tqdm import tqdm
import copy
from scipy.io.wavfile import write
import json
from pysepm import llr, cepstrum_distance, fwSNRseg, srmr 
from pystoi import stoi
from pesq import pesq

from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../de_noising"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../fine_tuning"))
from utils import *
from LateSupressionUnet import *
from DeNoiserUnet import *
from FineTuneModel import *

################################# FineTuningUNet #################################
class FineTuningUNet(nn.Module):    
    def __init__(self, denoise_unet, late_sup_unet):
        super(FineTuningUNet, self).__init__()
        self.denoise_unet = denoise_unet
        self.late_sup_unet = late_sup_unet

    def forward(self, input):
        output_1 = self.denoise_unet(input)
        combined_output = self.late_sup_unet(output_1)
        return combined_output

    # Noise + Late Reverberation Supression U-net
    # def __init__(self, DeNoiseUNet, LateSupUNet):
    #     super().__init__()
    #     self.modelA = DeNoiseUNet
    #     self.modelB = LateSupUNet
    #     # self.classifier = nn.Linear(340, 340) #try maybe non/bi-linear or RNN layer... TODO

##############################################################################################

def trainer(model, train_loader, val_loader, checkpoints, nEpochs = 30, lr = 1e-3):
  """
  Train model

  model: FineTuning model on GPU
  train_loader: dataloader containing train examples
  val_loader: dataloader containing validation examples
  checkpoints: list of directories to save the model, train loss and Val loss respectively
  """

  loss_function = nn.MSELoss()
  beta1 = 0.8
  beta2 = 0.999
  lr_decay = 0.97
  decay_rate = 2

  optimizer = torch.optim.Adam(model.parameters(), lr, (beta1, beta2))

  train_loss = []
  val_loss = []
  
  model.train()
  for epoch in range(nEpochs):
      
    if (epoch == 14):
        lr = 0.0008
        optimizer = torch.optim.Adam(model.parameters(), lr, (beta1, beta2))
        
    temp_train_loss = 0.0
    corrects_train = 0
    for i, (noisy_data, clean_data) in enumerate(train_loader):
      noisy_data = noisy_data.cuda()
      clean_data = clean_data.cuda()

      optimizer.zero_grad()
      output = model(noisy_data)
      loss_train = loss_function(output, clean_data)
      loss_train.backward()
      optimizer.step()

      temp_train_loss += loss_train.item()/len(train_loader)

    temp_val_loss = 0.0    
    #Validacion
    model.eval()
    with torch.no_grad():
      for i, (noisy_data, clean_data) in enumerate(val_loader):
        noisy_data = noisy_data.cuda()
        clean_data = clean_data.cuda()
        output = model(noisy_data)
        loss_val = loss_function(output, clean_data)
        temp_val_loss += loss_val.item()/len(val_loader)

    train_loss.append(temp_train_loss)
    val_loss.append(temp_val_loss)

    print('Epoch : {} || Train Loss: {:.3f} || Val Loss: {:.3f}'\
         .format(epoch+1, loss_train.item(), loss_val.item()))
    model.train()
    
    if ((epoch+1)%2 == 0):
      torch.save(model.state_dict(), checkpoints[0])
      torch.save(train_loss, checkpoints[1])
      torch.save(val_loss, checkpoints[2])
      print("saved models")

    if (epoch % decay_rate == 1):
       optimizer.param_groups[0]['lr'] *= lr_decay

  return train_loss, val_loss

def training(model, train_loader, val_loader, model_name, lr=2e-3, num_epochs=10):

    loss_function = nn.MSELoss()
    beta1 = 0.5
    beta2 = 0.999
    lr_decay = 0.97
    decay_rate = 2
    optimizer = torch.optim.Adam(model.parameters(), lr, (beta1, beta2))
    # loss_function = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    lowest_val_loss = np.inf
    best_model = None

    print("\nSTARTING TRAINIG\n")
    for epoch in tqdm(range(num_epochs)):

        print('Epoch: {} | Learning rate: {}'.format(epoch + 1, scheduler.get_last_lr()))

        for phase in ['train', 'val']:

            epoch_loss = 0
            epoch_correct_items = 0
            epoch_items = 0

            if phase == 'train':
                model.train()
                with torch.enable_grad():
                    for samples, targets in train_loader:
                        samples = samples.to(device)
                        targets = targets.to(device)

                        optimizer.zero_grad()
                        outputs = model(samples)
                        loss = loss_function(outputs, targets)
                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets).float().sum()
                        
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items.item()
                        epoch_items += len(targets)

                train_loss_array.append(epoch_loss / epoch_items)
                train_acc_array.append(epoch_correct_items / epoch_items)

                scheduler.step()

            elif phase == 'val':
                model.eval()
                with torch.no_grad():
                    for samples, targets in val_loader:
                        samples = samples.to(device)
                        targets = targets.to(device)

                        outputs = model(samples)
                        loss = loss_function(outputs, targets)
                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets).float().sum()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items.item()
                        epoch_items += len(targets)

                val_loss_array.append(epoch_loss / epoch_items)
                val_acc_array.append(epoch_correct_items / epoch_items)

                if epoch_loss / epoch_items < lowest_val_loss:
                    lowest_val_loss = epoch_loss / epoch_items
                    torch.save(model.state_dict(), '{}_weights.pth'.format(model_name))
                    best_model = copy.deepcopy(model)
                    print("\t| New lowest val loss for {}: {}".format(model_name, lowest_val_loss))

    return best_model, train_loss_array, train_acc_array, val_loss_array, val_acc_array


def evaluate(net, dataset, path, noisyspecs, init_example, end_example, speech_rate = 16000, initial = True, normalize_data = False):
  """
  net: U-net model used 
  dataset: reverb_dataset object to extract examples
  path: base directory to save results
  waves: directory containing reverb waveforms (not recovered from spectrograms)
  init_example: first example in dataset to be considered
  end_example: last example in dataset to be considered
  speech_rate: sampling rate of speech audios
  initial: True if is the first time executing, False if is not the first time
           if is False the new results are added to the existing results
  normalize_data: True if the net assumed normalized input 
  """

  device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
  print('Current device: ' + str(device))

  if (initial):
    reverb_pesq_list = []
    reverb_stoi_list = []
    reverb_llr_list = []
    reverb_cd_list = []
    reverb_fwSNRseg_list = []
    ##reverb_srmr_list = []

    dereverb_pesq_list = []
    dereverb_stoi_list = []
    dereverb_llr_list = []
    dereverb_cd_list = []
    dereverb_fwSNRseg_list = []
    ##de#reverb_srmr_list = []
  
  else:
    reverb_pesq_list = torch.load(path + 'reverb_pesq_SupUnet.pth')
    reverb_stoi_list = torch.load(path + 'reverb_stoi_SupUnet.pth')
    reverb_llr_list = torch.load(path + 'reverb_llr_SupUnet.pth')
    reverb_cd_list = torch.load(path + 'reverb_cd_SupUnet.pth')
    reverb_fwSNRseg_list = torch.load(path + 'reverb_fwSNRseg_SupUnet.pth')
    ##reverb_srmr_list = torch.load(path + 'reverb_srmr_SupUnet.pth')

    dereverb_pesq_list = torch.load(path + 'dereverb_pesq_SupUnet.pth')
    dereverb_stoi_list = torch.load(path + 'dereverb_stoi_SupUnet.pth')
    dereverb_llr_list = torch.load(path + 'dereverb_llr_SupUnet.pth')
    dereverb_cd_list = torch.load(path + 'dereverb_cd_SupUnet.pth')
    dereverb_fwSNRseg_list = torch.load(path + 'dereverb_fwSNRseg_SupUnet.pth')
    ##de#reverb_srmr_list = torch.load(path + 'dereverb_srmr_SupUnet.pth')
  
  net.eval()
  print("ENTER EVAL")
  
  for i in range(init_example, end_example):
    print("Processing Example n°{}".format(i+1))
    real_spec = noisyspecs[i]
    rev_spec, clean_spec = dataset.__getitem__(i)
    clean_spec = clean_spec[0, :, :]
    net_input = torch.zeros((1, 1, rev_spec.shape[1], rev_spec.shape[2]))

    if normalize_data:
      norm_example, norm_stats_rev = linear_scaler(rev_spec[0, :, :].numpy())
      net_input[0, 0, :, :] = torch.tensor(norm_example)
      net_response = net(net_input.to(device))
      net_response = net_response.clone().detach().cpu().numpy() if torch.cuda.is_available() else net_response.clone().detach().numpy()
      net_response = (net_response-norm_stats_rev[1])/norm_stats_rev[0]


    else:
      net_input[0, :, :, :] = rev_spec
      net_response = net(net_input.to(device))
      net_response = net_response.clone().detach().cpu().numpy() if torch.cuda.is_available() else net_response.clone().detach().numpy()
  
    
    try:
      net_response = net_response[0, 0, :, :]
      recon_spec = cv2.resize(net_response, dsize = (real_spec.shape[1], real_spec.shape[0]), interpolation = cv2.INTER_LANCZOS4)
      clean_spec = cv2.resize(clean_spec.numpy(), dsize = (real_spec.shape[1], real_spec.shape[0]), interpolation = cv2.INTER_LANCZOS4)
      rev_spec = cv2.resize(rev_spec[0, :, :].numpy(), dsize = (real_spec.shape[1], real_spec.shape[0]), interpolation = cv2.INTER_LANCZOS4)

      original = reconstruct_wave(clean_spec)
      reverb = reconstruct_wave(rev_spec)
      recon = reconstruct_wave(recon_spec)
      recon_srmr = recon

      pesq_metric_rev = pesq(original[0:len(reverb)], reverb, fs = speech_rate)
      stoi_metric_rev = stoi(original[0:len(reverb)], reverb, speech_rate)
      llr_metric_rev = llr(original[0:len(reverb)], reverb, speech_rate)
      cd_metric_rev = cepstrum_distance(original[0:len(reverb)], reverb, speech_rate)
      fwSNRseg_metric_rev = fwSNRseg(original[0:len(reverb)], reverb, speech_rate)
      #srmr_metric_rev = srmr(waves[i], speech_rate)

      pesq_metric_recon = pesq(original[0:len(recon)], recon, fs = speech_rate)
      stoi_metric_recon = stoi(original[0:len(recon)], recon, speech_rate)
      llr_metric_recon = llr(original[0:len(recon)], recon, speech_rate)
      cd_metric_recon = cepstrum_distance(original[0:len(recon)], recon, speech_rate)
      fwSNRseg_metric_recon = fwSNRseg(original[0:len(recon)], recon, speech_rate)
      #srmr_metric_recon = srmr(recon_srmr, speech_rate)

      reverb_pesq_list.append(pesq_metric_rev)
      reverb_stoi_list.append(stoi_metric_rev)
      reverb_llr_list.append(llr_metric_rev)
      reverb_cd_list.append(cd_metric_rev)
      reverb_fwSNRseg_list.append(fwSNRseg_metric_rev)
      ##reverb_srmr_list.append(srmr_metric_rev)

      dereverb_pesq_list.append(pesq_metric_recon)
      dereverb_stoi_list.append(stoi_metric_recon)
      dereverb_llr_list.append(llr_metric_recon)
      dereverb_cd_list.append(cd_metric_recon)
      dereverb_fwSNRseg_list.append(fwSNRseg_metric_recon)
      ##de#reverb_srmr_list.append(srmr_metric_recon)

      print('PESQ reverberated signal: {:.3f} || PESQ dereverberated signal: {:.3f}'.format(pesq_metric_rev, pesq_metric_recon))
      print('STOI reverberated signal: {:.3f} || STOI dereverberated signal: {:.3f}'.format(stoi_metric_rev, stoi_metric_recon))
      print('LLR reverberated signal: {:.3f} || LLR dereverberated signal: {:.3f}'.format(llr_metric_rev, llr_metric_recon))
      print('CD reverberated signal: {:.3f} || CD dereverberated signal: {:.3f}'.format(cd_metric_rev, cd_metric_recon))
      print('fwSNRseg reverberated signal: {:.3f} || fwSNRseg dereverberated signal: {:.3f}'.format(fwSNRseg_metric_rev, fwSNRseg_metric_recon))
      #print('SRMR reverberated signal: {:.3f} || SRMR dereverberated signal: {:.3f}'.format(srmr_metric_rev, srmr_metric_recon))

    except librosa.feature.inverse.ParameterError: 
      pass

    if ((i+1)%5 == 0):
      torch.save(reverb_pesq_list, path + 'reverb_pesq_SupUnet.pth')
      torch.save(reverb_stoi_list, path + 'reverb_stoi_SupUnet.pth')
      torch.save(reverb_llr_list, path + 'reverb_llr_SupUnet.pth')
      torch.save(reverb_cd_list, path + 'reverb_cd_SupUnet.pth')
      torch.save(reverb_fwSNRseg_list, path + 'reverb_fwSNRseg_SupUnet.pth')
      #torch.save(#reverb_srmr_list, path + 'reverb_srmr_SupUnet.pth')

      torch.save(dereverb_pesq_list, path + 'dereverb_pesq_SupUnet.pth')
      torch.save(dereverb_stoi_list, path + 'dereverb_stoi_SupUnet.pth')
      torch.save(dereverb_llr_list, path + 'dereverb_llr_SupUnet.pth')
      torch.save(dereverb_cd_list, path + 'dereverb_cd_SupUnet.pth')
      torch.save(dereverb_fwSNRseg_list, path + 'dereverb_fwSNRseg_SupUnet.pth')
      #torch.save(#de#reverb_srmr_list, path + 'dereverb_srmr_SupUnet.pth')
      
      print('Saved')

def visualize_training_results(train_loss_array,
                               val_loss_array,
                               train_acc_array,
                               val_acc_array,
                               num_epochs,
                               model_name,
                               batch_size):
    fig, axs = plt.subplots(1, 2, figsize=(14,4))
    fig.suptitle("{} training | Batch size: {}".format(model_name, batch_size), fontsize = 16)
    axs[0].plot(list(range(1, num_epochs+1)), train_loss_array, label="train_loss")
    axs[0].plot(list(range(1, num_epochs+1)), val_loss_array, label="val_loss")
    axs[0].legend(loc='best')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[1].plot(list(range(1, num_epochs+1)), train_acc_array, label="train_acc")
    axs[1].plot(list(range(1, num_epochs+1)), val_acc_array, label="val_acc")
    axs[1].legend(loc='best')
    axs[1].set(xlabel='epochs', ylabel='accuracy')
    plt.show()

#################################### TRAINING ####################################
def FineTuning_train():
  # # data set prep
  X = torch.load('/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/cleanspecs.pth')
  y = torch.load('/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/noisyspecs.pth')
   
  total_dataset = ReverbDataset(X, y)
  length_train = int(len(total_dataset)*0.85)
  length_val = len(total_dataset) - length_train
  lengths = [length_train, length_val]
  dataset_train, dataset_val = random_split(total_dataset, lengths)
   
  print(len(dataset_train))
  print(len(dataset_val))
   
  train_loader = DataLoader(dataset_train, batch_size = 16, shuffle = True, num_workers = 4, pin_memory = True)
  val_loader = DataLoader(dataset_val, batch_size = 16, shuffle = True, num_workers = 4, pin_memory = True)
   
  # models prep
  device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
  print('Current device: ' + str(device))
   
  LateSupUNetModel = LateSupUnet(n_channels=1, bilinear=False)
  LateSupUNetModel.load_state_dict(torch.load('/home/may.tiger/AIProject/data/training/model/LateSupUnet_state_dict.pth', map_location=lambda storage, loc: storage))
  LateSupUNetModel = LateSupUNetModel.cuda()
   
  DeNoiserModel = DeNoiseUNet(n_channels=1, bilinear=False)
  DeNoiserModel.load_state_dict(torch.load('/home/may.tiger/AIProject/de_noising/training/model/DeNoiser_state_dict.pth', map_location=lambda storage, loc: storage))
  DeNoiserModel = DeNoiserModel.cuda()
   
  FineTuning_Model = FineTuningUNet(LateSupUNetModel, DeNoiserModel)
  
  net = FineTuning_Model.cuda()
  
  checkpoints = ['/home/may.tiger/AIProject/fine_tuning/training/model/FineTuning_state_dict.pth', 
                 '/home/may.tiger/AIProject/fine_tuning/training/losses/train_loss_FineTuning.pth', 
                 '/home/may.tiger/AIProject/fine_tuning/training/losses/val_loss_FineTuning.pth']
  
  
  epochs = 20
  lr = 3e-2
  train_loss, val_loss = trainer(net, train_loader, val_loader, checkpoints, lr=lr, nEpochs = epochs)
  # best_model, train_loss_array, train_acc_array, val_loss_array, val_acc_array = training(net, train_loader, val_loader, "FineTuneSmartTrain", lr=3e-3, num_epochs = 10)
   
  plt.style.reload_library()
  train_loss = torch.load('/home/may.tiger/AIProject/fine_tuning/training/losses/train_loss_FineTuning.pth')
  val_loss = torch.load('/home/may.tiger/AIProject/fine_tuning/training/losses/val_loss_FineTuning.pth')
 
  matplotlib.rc('xtick', labelsize=10) 
  matplotlib.rc('ytick', labelsize=10)
  matplotlib.rcParams.update({'font.size': 10})
  plt.figure()
  plt.rcParams["font.family"] = "serif"
  plt.plot(np.arange(1, (epochs+1), 1), train_loss, '-', label = 'Train loss')
  plt.plot(np.arange(1, (epochs+1), 1), val_loss, '-', label = 'Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('MSE loss')
  plt.grid()
  plt.legend()
  plt.savefig(f"Fine_Tuneing_MSE_loss_to_epochs.jpg")
  #####################################################################################################
  # complex training
  '''
  for param in FineTuning_Model.parameters():
      param.requires_grad = False

  for param in FineTuning_Model.classifier.parameters():
      param.requires_grad = True    

  FineTuning_Model = FineTuning_Model.to(device)

  # training call
  ensemble_training_results = training(model=FineTuning_Model,
                                       model_name='/home/may.tiger/AIProject/fine_tuning/training/modelFineTuning',
                                       num_epochs=2, #TODO change epochs
                                       train_loader=train_loader,
                                       val_loader=val_loader,
                                       device=device)

  FineTuning_Model, train_loss_array, train_acc_array, val_loss_array, val_acc_array = ensemble_training_results

  # plot results
  min_loss = min(val_loss_array)
  min_loss_iteration = val_loss_array.index(min_loss)
  min_loss_accuracy = val_acc_array[min_loss_iteration]

  visualize_training_results(train_loss_array,
                             val_loss_array,
                             train_acc_array,
                             val_acc_array,
                             num_epochs=2,  #TODO change epochs
                             model_name='/home/may.tiger/AIProject/fine_tuning/training/modelFineTuning',
                             batch_size=16)

  print("\nTraining results:")
  print("\tMin val loss {:.4f} was achieved during iteration #{}".format(min_loss, min_loss_iteration + 1))
  print("\tVal accuracy during min val loss is {:.4f}".format(min_loss_accuracy))'''

#################################### EVALUATING ####################################
def FineTuning_eval():
  LateSupUNetModel = LateSupUnet(n_channels=1, bilinear=False).cuda()
  DeNoiserModel = DeNoiseUNet(n_channels=1, bilinear=False).cuda()
  model_FineTuning = FineTuningUNet(LateSupUNetModel, DeNoiserModel)
  model_FineTuning.load_state_dict(torch.load('/home/may.tiger/AIProject/fine_tuning/training/model/FineTuning_state_dict.pth', map_location=lambda storage, loc: storage))
  model_FineTuning = model_FineTuning.cuda()

  X_test = torch.load('/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/cleanspecs.pth')
  y_test = torch.load('/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/noisyspecs.pth')
  dataset_test = ReverbDataset(X_test, y_test)

  path = '/home/may.tiger/AIProject/fine_tuning/eval_results/'

  noisyspecs = torch.load('/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/noisyspecs.pth')

  #should we chang the speech_rate? (707) #TODO
  evaluate(model_FineTuning, dataset_test, path, noisyspecs, 0, 500, speech_rate = 16000, initial = True, normalize_data=False)

  path = '/home/may.tiger/AIProject/fine_tuning/eval_results/'
  reverb_pesq_list = torch.load(path + 'reverb_pesq_SupUnet.pth')
  reverb_stoi_list = torch.load(path + 'reverb_stoi_SupUnet.pth')
  reverb_llr_list = torch.load(path + 'reverb_llr_SupUnet.pth')
  reverb_cd_list = torch.load(path + 'reverb_cd_SupUnet.pth')
  reverb_fwSNRseg_list = torch.load(path + 'reverb_fwSNRseg_SupUnet.pth')
  ##reverb_srmr_list = torch.load(path + 'reverb_srmr_SupUnet.pth')

  dereverb_pesq_list = torch.load(path + 'dereverb_pesq_SupUnet.pth')
  dereverb_stoi_list = torch.load(path + 'dereverb_stoi_SupUnet.pth')
  dereverb_llr_list = torch.load(path + 'dereverb_llr_SupUnet.pth')
  dereverb_cd_list = torch.load(path + 'dereverb_cd_SupUnet.pth')
  dereverb_fwSNRseg_list = torch.load(path + 'dereverb_fwSNRseg_SupUnet.pth')
  ##de#reverb_srmr_list = torch.load(path + 'dereverb_srmr_SupUnet.pth')


  print('Results: \n')
  print('Reverberant signal:')
  print('PESQ: {:.2f}'.format(np.mean(reverb_pesq_list)))
  print('STOI: {:.2f}'.format(np.mean(reverb_stoi_list)))
  print('LLR: {:.2f}'.format(np.mean(reverb_llr_list)))
  print('CD: {:.2f}'.format(np.mean(reverb_cd_list)))
  print('fwSNRseg: {:.2f}'.format(np.mean(reverb_fwSNRseg_list)))
  ##print('SRMR: {:.2f}'.format(np.mean(#reverb_srmr_list)))

  print('\nDereverberated signal:')
  print('PESQ: {:.2f}'.format(np.mean(dereverb_pesq_list)))
  print('STOI: {:.2f}'.format(np.mean(dereverb_stoi_list)))
  print('LLR: {:.2f}'.format(np.mean(dereverb_llr_list)))
  print('CD: {:.2f}'.format(np.mean(dereverb_cd_list)))
  print('fwSNRseg: {:.2f}'.format(np.mean(dereverb_fwSNRseg_list)))
  ##print('SRMR: {:.2f}'.format(np.mean(#de#reverb_srmr_list)))

def FineTuning_pesq_test():
  ''' '''
  LateSupUNetModel = LateSupUnet(n_channels=1, bilinear=False).cuda()
  DeNoiserModel = DeNoiseUNet(n_channels=1, bilinear=False).cuda()
  model_FineTuning = FineTuningUNet(LateSupUNetModel, DeNoiserModel)
  model_FineTuning.load_state_dict(torch.load('/home/may.tiger/AIProject/fine_tuning/training/model/FineTuning_state_dict.pth', map_location=lambda storage, loc: storage))
  model_FineTuning = model_FineTuning.cuda()

  X_test = torch.load('/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/cleanspecs.pth')
  y_test = torch.load('/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/noisyspecs.pth')
  dataset_test = ReverbDataset(X_test, y_test)

  path = '/home/may.tiger/AIProject/fine_tuning/eval_results/'

  noisyspecs = torch.load('/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/noisyspecs.pth')
  pesq_evaluate(model_FineTuning, dataset_test, path, noisyspecs, 0, 1, speech_rate = 16000, initial = True, normalize_data=False)
  
def pesq_evaluate(net, dataset, path, noisyspecs, init_example, end_example, speech_rate = 16000, initial = True, normalize_data = False): 
  device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
  print('Current device: ' + str(device))
  reverb_pesq_list = []
  dereverb_pesq_list = []
  net.eval()
  print("ENTER PESQ EVALUATION")
  
  for i in range(init_example, end_example):
    print("Processing Example n°{}".format(i+1))
    real_spec = noisyspecs[i]
    # print(real_spec.shape)
    rev_spec, clean_spec = dataset.__getitem__(i)
    # print(rev_spec.shape)
    # print(clean_spec.shape)
    clean_spec = clean_spec[0, :, :]
    net_input = torch.zeros((1, 1, rev_spec.shape[1], rev_spec.shape[2]))
    
    net_input[0, :, :, :] = rev_spec
    net_response = net(net_input.to(device))
    net_response = net_response.clone().detach().cpu().numpy() if torch.cuda.is_available() else net_response.clone().detach().numpy()
  
    try:
      net_response = net_response[0, 0, :, :]
      recon_spec = cv2.resize(net_response, dsize = (real_spec.shape[1], real_spec.shape[0]), interpolation = cv2.INTER_LANCZOS4)
      clean_spec = cv2.resize(clean_spec.numpy(), dsize = (real_spec.shape[1], real_spec.shape[0]), interpolation = cv2.INTER_LANCZOS4)
      rev_spec = cv2.resize(rev_spec[0, :, :].numpy(), dsize = (real_spec.shape[1], real_spec.shape[0]), interpolation = cv2.INTER_LANCZOS4)

      original = reconstruct_wave(clean_spec)
      reverb = reconstruct_wave(rev_spec)
      recon = reconstruct_wave(recon_spec)
      recon_srmr = recon

      
      print(reverb.shape)
      if (isinstance(reverb, np.ndarray) and reverb.ndim == 1):
        print('reverb is a 1D array')
        
      pesq_metric_rev = pesq(original[0:len(reverb)], reverb, fs=speech_rate)

      pesq_metric_recon = pesq(original[0:len(recon)], recon, fs=speech_rate)

      reverb_pesq_list.append(pesq_metric_rev)

      dereverb_pesq_list.append(pesq_metric_recon)

      print('PESQ reverberated signal: {:.3f} || PESQ dereverberated signal: {:.3f}'.format(pesq_metric_rev, pesq_metric_recon))

    except librosa.feature.inverse.ParameterError: 
      pass

    torch.save(reverb_pesq_list, path + 'reverb_pesq_SupUnet.pth')
    torch.save(dereverb_pesq_list, path + 'dereverb_pesq_SupUnet.pth')
      
    print('Saved')