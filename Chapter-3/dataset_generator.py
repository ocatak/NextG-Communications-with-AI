###### Deep Learning Coordinated Beamforming with DeepMIMO #################
# Author: Umut Demirhan, Ahmed Alkhateeb
# Date: March 19, 2022 
# Paper: A. Alkhateeb, S. Alex, P. Varkey, Y. Li, Q. Qu and D. Tujkovic, 
# "Deep Learning Coordinated Beamforming for Highly-Mobile Millimeter Wave 
# Systems," in IEEE Access, vol. 6, pp. 37328-37348, 2018.
###########################################################################

import DeepMIMO
import os
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
from scipy.io import loadmat, savemat 
import glob
import re 

plt.rcParams['figure.figsize'] = [12, 8] # Set default plot size


def beamforming_codebook(ant_shape = np.array([1, 32, 1]), oversampling_rate = np.array([1, 1, 1]), kd = 0.5):
    
    kd = 2 * np.pi * kd
    codebook_size = ant_shape * oversampling_rate
    
    vecs = []
    for dim in range(3):
        ind = np.arange(ant_shape[dim]).reshape((-1, 1))
        codebook_ang = np.linspace(0, np.pi, codebook_size[dim], endpoint = False).reshape((1, -1))                                                                                                     
        vec = np.sqrt(1./ant_shape[dim]) * np.exp(-1j * kd * ind * np.cos(codebook_ang))
        vecs.append(vec)
        
    F = np.kron(vecs[2], np.kron(vecs[1], vecs[0]))
    
    return F
    
   
#%% # Generate the dataset
# # Load and print the default parameters
parameters = DeepMIMO.default_params()


# # Change parameters for the setup
# Scenario O1_60 extracted at the dataset_folder
parameters['scenario'] = 'O1_60'
parameters['dataset_folder'] = r'/Users/ozgur/Documents/datasets_for_the_book' # Set DeepMIMO dataset folder that has O1_60

parameters['num_paths'] = 5

# User rows 1-100
parameters['user_row_first'] = 1000
parameters['user_row_last'] = 1300

# Activate only the first basestation
parameters['active_BS'] = np.array([3, 4, 5, 6]) 

parameters['OFDM']['bandwidth'] = 0.5 # 50 MHz
parameters['OFDM']['subcarriers'] = 1024 # OFDM with 512 subcarriers
parameters['OFDM']['subcarriers_limit'] = 64 # Keep only first 64 subcarriers

parameters['enable_BS2BS'] = False

parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # Single antenna
parameters['bs_antenna']['shape'] = np.array([1, 32, 8]) # ULA of 32 elements
parameters['bs_antenna']['radiation_pattern'] = 'halfwave-dipole'
parameters['ue_antenna']['radiation_pattern'] = 'halfwave-dipole'
pprint(parameters, sort_dicts = False)

dataset = DeepMIMO.generate_data(parameters)

#%% Parameters and Codebook
F = beamforming_codebook(ant_shape = parameters['bs_antenna'][0]['shape'], oversampling_rate = np.array([1, 2, 1]), kd = parameters['bs_antenna'][0]['spacing'])

num_OFDM = int(parameters['OFDM']['subcarriers_limit']/parameters['OFDM']['subcarriers_sampling'])
num_beams = F.shape[1]
num_bs = len(parameters['active_BS'])
num_ue = len(parameters['active_UE'])

NF = 5             # Noise figure at the base station
Process_Gain = 10  # Channel estimation processing gain
BW = parameters['OFDM']['bandwidth'] * 1e9 # System bandwidth in Hz
noise_power_dB = -204 + 10*np.log10(BW/parameters['OFDM']['subcarriers']) + NF - Process_Gain; # Noise power in dB
noise_power = 10**(.1*(noise_power_dB)); # Noise power

#%% DL Input-Output
input_norm = np.zeros((num_bs, num_ue, num_OFDM), dtype=complex)
max_rates = np.zeros((num_bs, num_ue, num_beams))
for bs_idx in tqdm(range(num_bs), desc='Neural Network Input-Output Generation-BS', position=0, leave=True):
    for ue_idx in tqdm(range(num_ue), desc='Neural Network Input-Output Generation-BS-%i'%bs_idx, position=0, leave=True):
        ch = dataset[bs_idx]['user']['channel'][ue_idx].squeeze()
        ch = ch + np.sqrt(noise_power) * (np.random.randn(*(ch.shape)) + 1j * np.random.randn(*(ch.shape)))
        input_norm[bs_idx, ue_idx, :] = ch[0, :]
        max_rates[bs_idx, ue_idx, :] = np.sum(np.log2(1 + np.abs(ch.T.conj() @ F)**2),  axis = 0)/num_OFDM
        
# Input reshape - normalize
input_norm = np.transpose(input_norm, axes=[1, 0, 2])
input_norm = input_norm.reshape((num_ue, -1))
input_norm /=  np.amax(np.abs(input_norm))

# Output reshape - normalize
max_rates_norm_factor = np.amax(max_rates, axis=2, keepdims=True)
max_rates_norm_factor[max_rates_norm_factor== 0] = 1 # Do not normalize if all zeros
max_rates /= max_rates_norm_factor
max_rates = np.transpose(max_rates, axes=[1, 0, 2])
max_rates = max_rates.reshape((num_ue, -1))

if not os.path.exists('./DLCB_dataset'):
                      os.makedirs('DLCB_dataset')
savemat('./DLCB_input.mat', {'DL_input': input_norm})
savemat('./DLCB_output.mat', {'DL_output': max_rates})
