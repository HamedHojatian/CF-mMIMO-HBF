import torch.utils.data as data
from termcolor import colored
import torch.nn.functional as F
import torch
import scipy.io
from numpy import genfromtxt
import numpy as np
from utils_math import Th_comp_matmul, Th_inv, Th_pinv
import neptune
import re
import torch.nn as nn
import time
import math

# Database ####################################################################################################################
class Data_Reader(data.Dataset):
    def __init__(self, filename, Us, Mr, Nrf, K, N_BS):
        print(colored('You select Extended dataset', 'cyan'))
        print(colored(filename, 'yellow'), 'is loading ... ')
        np_data = np.load(filename)
        self.channel = np_data[:, 0:Us * Mr * N_BS]
        self.RSSI_N = np_data[:, Us * Mr * N_BS:].real.astype(float)
        self.n_samples = np_data.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return torch.tensor(self.channel[index]).type(torch.complex64), torch.tensor(self.RSSI_N[index])


# readme reader for HBF initial parameters ####################################################################################
def md_reader(DB_name):
    md = genfromtxt(''.join((DB_name, '/DATASET.md')), delimiter='\n', dtype='str')
    Us = int(re.findall(r'\d+', md[1])[0])
    Mr = int(re.findall(r'\d+', md[2])[0])
    Nrf = int(re.findall(r'\d+', md[3])[0])
    N_BS = int(re.findall(r'\d+', md[4])[0])
    K = int(re.findall(r'\d+', md[5])[0])
    Noise_pwr = 10 ** -(int(re.findall(r'\d+', md[7])[0]) / 10)
    return Us, Mr, Nrf, N_BS, K, Noise_pwr

class Initialization_Model_Params(object):
    def __init__(self,
                 DB_name,
                 Us,
                 Mr,
                 Nrf,
                 K,
                 K_limited,
                 N_BS,
                 Noise_pwr,
                 device,
                 device_ids
                 ):
        self.DB_name = DB_name
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.K = K
        self.K_limited = K_limited
        self.N_BS = N_BS
        self.Noise_pwr = Noise_pwr
        self.device = device
        self.dev_id = device_ids

    def Data_Load(self):
        if self.SSB_Type == 'parallel':
            DataBase = Data_Reader(''.join((self.DB_name, '/dataSet_withRSSI_32SSB_Par.npy')), self.Us, self.Mr, self.Nrf, self.K, self.N_BS)
        elif self.SSB_Type == 'series':
            DataBase = Data_Reader(''.join((self.DB_name, '/dataSet_withRSSI_32SSB_Ser.npy')), self.Us, self.Mr, self.Nrf, self.K, self.N_BS)
        return DataBase  # , uniq_dis_label

    def Code_Read(self):
        mat_C1 = torch.tensor(np.array(scipy.io.loadmat('dataSet4x64x8x4/BS4/Codebook_SSB.mat')['codebook'])).type(torch.complex64)
        mat_C2 = torch.tensor(np.array(scipy.io.loadmat('dataSet4x64x8x4/BS5/Codebook_SSB.mat')['codebook'])).type(torch.complex64)
        mat_C3 = torch.tensor(np.array(scipy.io.loadmat('dataSet4x64x8x4/BS8/Codebook_SSB.mat')['codebook'])).type(torch.complex64)
        mat_C4 = torch.tensor(np.array(scipy.io.loadmat('dataSet4x64x8x4/BS9/Codebook_SSB.mat')['codebook'])).type(torch.complex64)
        return [mat_C1[0:2, :], mat_C2[0:6, :], mat_C3[0:6, :], mat_C4[0:2, :]], \
            [len(mat_C1[0:2, :]), len(mat_C2[0:6, :]), len(mat_C3[0:6, :]), len(mat_C4[0:2, :])]

    def plot_grad_flow(self, named_parameters):
        # ave_grads = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                # ave_grads.append(p.grad.abs().mean())
                neptune.send_metric(f'layer{n}', p.grad.abs().mean())


class Loss_FCDP_Rate_Based(torch.nn.Module):
    def __init__(self, Us, Mr, Nrf, N_BS, Noise_pwr):
        super(Loss_FCDP_Rate_Based, self).__init__()
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.N_BS = N_BS
        self.noise_power = Noise_pwr

    def rate_calculator(self, FDP, channel):
        W = torch.abs(torch.matmul(torch.conj(channel), FDP).sum(1)) ** 2
        SINR = torch.diagonal(W, dim1=1, dim2=2) / (torch.sum(W, 2) - torch.diagonal(W, dim1=1, dim2=2) + self.noise_power)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(1)
        avgRate = userRates.mean(1)
        return sumRate, avgRate

    def forward(self, FDP, channel):
        FDP = FDP / torch.linalg.norm(FDP, dim=1, keepdim=True)
        FDP = FDP.view(-1, self.Us, self.Mr, self.N_BS).permute(0, 3, 2, 1)
        sum_rate, _ = Loss_FCDP_Rate_Based.rate_calculator(self, FDP, channel)
        return - sum_rate.mean()

    def evaluate_rate(self, FDP, channel):
        FDP = FDP / torch.linalg.norm(FDP, dim=1, keepdim=True)
        FDP = FDP.view(-1, self.Us, self.Mr, self.N_BS).permute(0, 3, 2, 1)
        sum_rate, avg_rate = Loss_FCDP_Rate_Based.rate_calculator(self, FDP, channel)
        return sum_rate.mean(), avg_rate.mean()

class Loss_HCBF_Rate_Based(torch.nn.Module):
    def __init__(self, Us, Mr, Nrf, N_BS, Noise_pwr):
        super(Loss_HCBF_Rate_Based, self).__init__()
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.N_BS = N_BS
        self.noise_power = Noise_pwr

    def rate_calculator(self, FDP, channel):
        W = torch.abs(torch.matmul(torch.conj(channel), FDP).sum(1)) ** 2
        SINR = torch.diagonal(W, dim1=1, dim2=2) / (torch.sum(W, 2) - torch.diagonal(W, dim1=1, dim2=2) + self.noise_power)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(1)
        avgRate = userRates.mean(1)
        return sumRate, avgRate, userRates

    def rate_calculator_4d(self, FDP, channel):
        W = torch.abs(torch.matmul(torch.conj(channel), FDP).sum(5)) ** 2
        SINR = torch.diagonal(W, dim1=5, dim2=6) / (torch.sum(W, 6) - torch.diagonal(W, dim1=5, dim2=6) + self.noise_power)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(5)
        avgRate = userRates.mean(5)
        return sumRate, avgRate

    def forward(self, W, channel, A):
        HBF = torch.matmul(A.view(-1, len(channel), self.Nrf, self.Mr, self.N_BS).permute(0, 1, 4, 3, 2), W)
        HBF = HBF / torch.unsqueeze(torch.unsqueeze(torch.linalg.norm(HBF.flatten(2), dim=2).unsqueeze(2), 3), 4)
        sum_rate, _ = Loss_HCBF_Rate_Based.rate_calculator_4d(self, HBF, channel)
        return sum_rate.T

    def evaluate_rate(self, W, channel, A):
        HBF = torch.matmul(A.view(len(channel), self.Nrf, self.Mr, self.N_BS).permute(0, 3, 2, 1), W)
        HBF = HBF / torch.unsqueeze(torch.unsqueeze(torch.linalg.norm(HBF.flatten(1), dim=1).unsqueeze(1), 2), 3)
        sum_rate, avgRate = Loss_HCBF_Rate_Based.rate_calculator(self, HBF, channel)
        return sum_rate.mean(), avgRate.mean()

class Loss_HCBF_S_Rate_Based(torch.nn.Module):
    def __init__(self, Us, Mr, Nrf, N_BS, Noise_pwr):
        super(Loss_HCBF_S_Rate_Based, self).__init__()
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.N_BS = N_BS
        self.noise_power = Noise_pwr

    def forward(self, W, channel, A_s1, A_s2, A_s3, A_s4, sinr_3d):
        A = torch.cat((A_s1.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, sinr_3d.shape[1], sinr_3d.shape[2], sinr_3d.shape[3], 1, 1, 1),
                       A_s2.unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat(sinr_3d.shape[0], 1, sinr_3d.shape[2], sinr_3d.shape[3], 1, 1, 1),
                       A_s3.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(sinr_3d.shape[0], sinr_3d.shape[1], 1, sinr_3d.shape[3], 1, 1, 1),
                       A_s4.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(sinr_3d.shape[0], sinr_3d.shape[1], sinr_3d.shape[2], 1, 1, 1, 1)), axis=5)

        HBF = torch.matmul(A.view(sinr_3d.shape[0],
                                  sinr_3d.shape[1],
                                  sinr_3d.shape[2],
                                  sinr_3d.shape[3], len(channel), self.N_BS, self.Nrf, self.Mr).permute(0, 1, 2, 3, 4, 5, 7, 6), W)

        power = torch.unsqueeze(torch.unsqueeze(torch.linalg.norm(HBF.flatten(5), dim=5).unsqueeze(5), 6), 7)
        HBF = HBF / power
        sinr_3d = Loss_HCBF_Rate_Based.rate_calculator_4d(self, HBF, channel)[0]
        return sinr_3d, power

    def evaluate_rate(self, W, channel, A):
        HBF = torch.matmul(A.view(len(channel), self.N_BS, self.Nrf, self.Mr).permute(0, 1, 3, 2), W)
        Power = torch.unsqueeze(torch.unsqueeze(torch.linalg.norm(HBF.flatten(1), dim=1).unsqueeze(1), 2), 3)
        HBF = HBF / Power
        sum_rate, avgRate, userRates = Loss_HCBF_Rate_Based.rate_calculator(self, HBF, channel)
        return sum_rate.mean(), avgRate.mean(), userRates, Power.mean()


def FLP_loss(x, y):
    log_prob = - 1.0 * F.softmax(x, 1)
    temp = log_prob * y
    cel = temp.sum(dim=1)
    cel = cel.mean()
    return cel

def FLP_loss_s(x, y):
    log_prob = - 1.0 * x
    temp = log_prob * y
    cel = temp.sum(dim=1).sum(dim=1).sum(dim=1).sum(dim=1)
    cel = cel.mean()
    return cel
