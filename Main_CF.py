import numpy as np
import os
import csv
import torch.nn as nn
import torch.nn.functional as F
from numpy import genfromtxt
from networks_activation_CF import Networks_activations
import torch as th
from utils_CF import md_reader, Initialization_Model_Params, Loss_FCDP_Rate_Based, Loss_HCBF_Rate_Based, FLP_loss, FLP_loss_s, Loss_HCBF_S_Rate_Based
from utils_math import Th_pinv, Th_comp_matmul, Th_inv
from dataset_prepration_deepMIMO_CF import DB_pro
from termcolor import colored
from torchsummary import summary
import numpy.matlib
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import neptune.new as npt
import neptune as npt

###############################################################################
# Directory file
###############################################################################
DB_name = 'dataSet4x64x8x4/130dB'

###############################################################################
# Processor selection ss
###############################################################################
device = th.device("cuda:2" if th.cuda.is_available() else "cpu")
device_ids = [2]
print("Is Cuda available? ", colored('True', 'green')
    if th.cuda.is_available() else colored('False', 'red'))
print("Which devide?", colored(device, 'cyan'))

###################################################################################
# Setup Parameters
###################################################################################

# Prediction  FDP, FLP_W_FD, FLP_W_PD #############################################
BF_approach = 'FLP_W_FD'

# Model Types 2(FullyDecentralized), 3(PartiallyDecentralized)  ###################
Net_MT_Type = 2 if BF_approach == 'FLP_W_FD' else 3

###############################################################################
# Beamfroming and DNN Parameters
###############################################################################
os.chdir('/export/tmp/datasets/deepMIMO/HH_channels/deepMIMO/DataBase/CF_DeepMIMO')
Us, Mr, Nrf, N_BS, K, Noise_pwr = md_reader(DB_name)
K = 32
rho_u = 1                                                     # Probability of connected users (0 all disconnected, 1 all conneceted)
K_limited = int(K / 8)                                        # defualt change when you want !
batch_size = 1000                                             # Batch size
epoch_size = 500                                              # Number of training epoches
lr = 0.001                                                    # Learning rate
wd = 1e-6                                                     # Weight decay
n_input = Us * K_limited                                      # Input dimensions
n_hidden = 1024                                               # Size of FCL layers
out_channel = 32                                              # Size of CL channels
kernel_s = 3                                                  # Size of Kernels in CL
padding = 1                                                   # Size of padding in CL
p_dropout = 0.02                                              # Probability of dropout
export = 'N'                                                  # Save the output (0,1)
n_input = Us * N_BS * K_limited
if BF_approach in ['FLP_W_FD', 'FLP_W_PD']:
    n_output_reg = Us * Nrf
elif BF_approach in ['FDP']:
    n_output_reg = Us * Mr
else:
    raise Exception('BF_approach value is wrong !!')

###############################################################################
# Main Menu of configuration
###############################################################################
Main_Menu = Initialization_Model_Params(DB_name,
                                        Us,
                                        Mr,
                                        Nrf,
                                        K,
                                        K_limited,
                                        N_BS,
                                        Noise_pwr,
                                        device,
                                        device_ids)

###############################################################################
# Reading Database
###############################################################################
DataBase = Main_Menu.Data_Load()

###############################################################################
# Codeword dictionary
###############################################################################
Codebooks, Codebook_len = Main_Menu.Code_Read()

###############################################################################
# Trainset and testset generation
###############################################################################
train_size = int(0.85 * len(DataBase))
test_size = len(DataBase) - train_size
train_dataset, test_dataset = th.utils.data.random_split(DataBase, [train_size, test_size])

print(colored('The size of training set is ', 'yellow'), len(train_dataset))
print(colored('The size of Test set is ', 'yellow'), len(test_dataset))

###############################################################################
# Dataloaders
###############################################################################
my_dataloader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
my_testloader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

###############################################################################
# Networks Arch
###############################################################################
Networks_Main_Menu = Networks_activations(DB_name,
                                        Us,
                                        Mr,
                                        Nrf,
                                        K,
                                        K_limited,
                                        N_BS,
                                        Noise_pwr,
                                        Net_MT_Type,
                                        device,
                                        device_ids,
                                        n_input,
                                        n_hidden,
                                        n_output_reg,
                                        Codebook_len,
                                        p_dropout,
                                        out_channel,
                                        kernel_s,
                                        padding)

Model_CF = Networks_Main_Menu.Network_CF()

###############################################################################
# OPTIMIZER
###############################################################################
optimizer_CF = th.optim.Adam(Model_CF.parameters(), lr=lr, weight_decay=wd)

###############################################################################
# scheduler Lr
###############################################################################
scheduler_MT = ReduceLROnPlateau(optimizer_CF, mode='max', factor=0.1, patience=5, verbose=True)

###############################################################################
# Main training loop
###############################################################################
if BF_approach == 'FLP_W_FD':
    criterium_clas_4d = Loss_HCBF_S_Rate_Based(Us, Mr, Nrf, N_BS, Noise_pwr).to(device)
    for i in range(1, epoch_size):
        for k, (channel, RSSI) in enumerate(my_dataloader):

            Inputs_Reg = Networks_Main_Menu.Inp_MT(RSSI)

            channel = channel.view(-1, Us, Mr, N_BS).permute(0, 3, 1, 2).to(device)

            # Set gradient to 0.
            optimizer_CF.zero_grad()

            # Feed forward Reg
            Model_CF.train()
            outR_reg, outI_Reg, outC1, outC2, outC3, outC4 = Model_CF(Inputs_Reg[:, 0:1, :, :],
                                                                      Inputs_Reg[:, 1:2, :, :],
                                                                      Inputs_Reg[:, 2:3, :, :],
                                                                      Inputs_Reg[:, 3:4, :, :])
            out_reg = outR_reg + 1j * outI_Reg

            # W calc
            w_out = out_reg.view(-1, N_BS, Us, Nrf)
            sinr_3d = th.zeros(Codebook_len[0], Codebook_len[1], Codebook_len[2], Codebook_len[3], len(RSSI))

            HBF_all_4d, power = criterium_clas_4d(w_out.permute(0, 1, 3, 2), channel,
                                           th.unsqueeze(Codebooks[0].unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device),
                                           th.unsqueeze(Codebooks[1].unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device),
                                           th.unsqueeze(Codebooks[2].unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device),
                                           th.unsqueeze(Codebooks[3].unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device), sinr_3d)

            prob_3d = th.zeros(len(RSSI), outC1.shape[1], outC2.shape[1], outC3.shape[1], outC4.shape[1]).to(device)
            for s1 in range(Codebook_len[0]):
                for s2 in range(Codebook_len[1]):
                    for s3 in range(Codebook_len[2]):
                        for s4 in range(Codebook_len[3]):
                            prob_3d[:, s1, s2, s3, s4] = outC1[:, s1] * outC2[:, s2] * outC3[:, s3] * outC4[:, s4]

            loss_clas = FLP_loss_s(prob_3d.to(device), HBF_all_4d.to(device).permute(4, 0, 1, 2, 3))
            loss = loss_clas

            # Gradient calculation.
            loss.backward()

            # Model weight modification based on the optimizer.
            optimizer_CF.step()

            if k == 0 or i % epoch_size == 0:
                with th.no_grad():
                    sumRate_predicted_HCF = []
                    avgRate_predicted_HCF = []
                    UserRate_predicted_HCF = []
                    power = []
                    for (tchannel, tRSSI) in my_testloader:

                        testInputs_Reg = Networks_Main_Menu.Inp_MT(tRSSI)

                        T_channel = tchannel.view(-1, Us, Mr, N_BS).permute(0, 3, 1, 2).to(device)

                        # Forward pass reg
                        Model_CF.eval()
                        pred1_reg, pred2_reg, pred_class_s1, pred_class_s2, pred_class_s3, pred_class_s4 = Model_CF(testInputs_Reg[:, 0:1, :, :],
                                                                                                                    testInputs_Reg[:, 1:2, :, :],
                                                                                                                    testInputs_Reg[:, 2:3, :, :],
                                                                                                                    testInputs_Reg[:, 3:4, :, :])
                        pred_reg = pred1_reg + 1j * pred2_reg

                        _, predicted_s1 = th.max(pred_class_s1, 1)
                        _, predicted_s2 = th.max(pred_class_s2, 1)
                        _, predicted_s3 = th.max(pred_class_s3, 1)
                        _, predicted_s4 = th.max(pred_class_s4, 1)

                        # W calc
                        Analog_Predictedr_s1 = Codebooks[0][predicted_s1, :].to(device)
                        Analog_Predictedr_s2 = Codebooks[1][predicted_s2, :].to(device)
                        Analog_Predictedr_s3 = Codebooks[2][predicted_s3, :].to(device)
                        Analog_Predictedr_s4 = Codebooks[3][predicted_s4, :].to(device)

                        An_Pred = th.cat((Analog_Predictedr_s1, Analog_Predictedr_s2, Analog_Predictedr_s3, Analog_Predictedr_s4), axis=1)

                        w_pre = pred_reg.view(-1, N_BS, Us, Nrf)

                        Temp = criterium_clas_4d.evaluate_rate(w_pre.permute(0, 1, 3, 2), T_channel, An_Pred)

                        sumRate_predicted_HCF.append(Temp[0])
                        avgRate_predicted_HCF.append(Temp[1])
                        UserRate_predicted_HCF.append(Temp[2])

                # Final Value for rate
                sumRATE_Predicted_HCF = sum(sumRate_predicted_HCF) / len(sumRate_predicted_HCF)
                avgRATE_Predicted_HCF = sum(avgRate_predicted_HCF) / len(avgRate_predicted_HCF)

                scheduler_MT.step(sumRATE_Predicted_HCF)

                # Plots on Neptun Rate
                npt.send_metric('Sum Rate Value HBF', sumRATE_Predicted_HCF)
                npt.send_metric('Average Rate Value HBF', avgRATE_Predicted_HCF)

                print('Iter:==>{:3d} Loss_Class:{:.3f} sumRate_pre_HCBF:{:.2f} avgRate_pre_HCBF:{:.2f}'.
                    format(i, loss_clas, sumRATE_Predicted_HCF, avgRATE_Predicted_HCF))

elif BF_approach == 'FLP_W_PD':
    criterium_clas_4d = Loss_HCBF_S_Rate_Based(Us, Mr, Nrf, N_BS, Noise_pwr).to(device)
    for i in range(1, epoch_size):
        for k, (channel, RSSI) in enumerate(my_dataloader):

            Inputs_Reg = Networks_Main_Menu.Inp_MT(RSSI)

            channel = channel.view(-1, Us, Mr, N_BS).permute(0, 3, 1, 2).to(device)

            # Set gradient to 0.
            optimizer_CF.zero_grad()

            # Feed forward Reg
            Model_CF.train()
            outR_reg, outI_Reg, outC1, outC2, outC3, outC4 = Model_CF(Inputs_Reg)
            out_reg = outR_reg + 1j * outI_Reg

            # W calc
            w_out = out_reg.view(-1, N_BS, Us, Nrf)
            sinr_3d = th.zeros(Codebook_len[0], Codebook_len[1], Codebook_len[2], Codebook_len[3], len(RSSI))

            HBF_all_4d = criterium_clas_4d(w_out.permute(0, 1, 3, 2), channel,
                                           th.unsqueeze(Codebooks[0].unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device),
                                           th.unsqueeze(Codebooks[1].unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device),
                                           th.unsqueeze(Codebooks[2].unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device),
                                           th.unsqueeze(Codebooks[3].unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device), sinr_3d)

            prob_3d = th.zeros(len(RSSI), outC1.shape[1], outC2.shape[1], outC3.shape[1], outC4.shape[1]).to(device)
            for s1 in range(Codebook_len[0]):
                for s2 in range(Codebook_len[1]):
                    for s3 in range(Codebook_len[2]):
                        for s4 in range(Codebook_len[3]):
                            prob_3d[:, s1, s2, s3, s4] = outC1[:, s1] * outC2[:, s2] * outC3[:, s3] * outC4[:, s4]

            loss_clas = FLP_loss_s(prob_3d.to(device), HBF_all_4d.to(device).permute(4, 0, 1, 2, 3))

            # Gradient calculation.
            loss_clas.backward()

            # Model weight modification based on the optimizer.
            optimizer_CF.step()

            if k == 0 or i % epoch_size == 0:
                if i == 1:
                    Model_CF.eval()
                    if Net_MT_Type in [0, 1]:
                        summary(Model_CF, (n_input,))
                    elif Net_MT_Type in [2, 3]:
                        if SSB_Type == 'parallel':
                            summary(Model_CF, (1, Us, K_limited))
                        else:
                            summary(Model_CF, (N_BS, Us, K_limited))
                # iterate through test dataset
                with th.no_grad():
                    sumRate_predicted_HCF = []
                    avgRate_predicted_HCF = []
                    for (tchannel, tRSSI) in my_testloader:

                        testInputs_Reg = Networks_Main_Menu.Inp_MT(tRSSI)

                        T_channel = tchannel.view(-1, Us, Mr, N_BS).permute(0, 3, 1, 2).to(device)

                        # Forward pass reg
                        Model_CF.eval()
                        pred1_reg, pred2_reg, pred_class_s1, pred_class_s2, pred_class_s3, pred_class_s4 = Model_CF(testInputs_Reg)
                        pred_reg = pred1_reg + 1j * pred2_reg

                        _, predicted_s1 = th.max(pred_class_s1, 1)
                        _, predicted_s2 = th.max(pred_class_s2, 1)
                        _, predicted_s3 = th.max(pred_class_s3, 1)
                        _, predicted_s4 = th.max(pred_class_s4, 1)

                        # W calc
                        Analog_Predictedr_s1 = Codebooks[0][predicted_s1, :].to(device)
                        Analog_Predictedr_s2 = Codebooks[1][predicted_s2, :].to(device)
                        Analog_Predictedr_s3 = Codebooks[2][predicted_s3, :].to(device)
                        Analog_Predictedr_s4 = Codebooks[3][predicted_s4, :].to(device)

                        An_Pred = th.cat((Analog_Predictedr_s1, Analog_Predictedr_s2, Analog_Predictedr_s3, Analog_Predictedr_s4), axis=1)

                        w_pre = pred_reg.view(-1, N_BS, Us, Nrf)

                        sumRate_predicted_HCF.append(criterium_clas_4d.evaluate_rate(w_pre.permute(0, 1, 3, 2), T_channel, An_Pred)[0])
                        avgRate_predicted_HCF.append(criterium_clas_4d.evaluate_rate(w_pre.permute(0, 1, 3, 2), T_channel, An_Pred)[1])

                # Final Value for rate
                sumRATE_Predicted_HCF = sum(sumRate_predicted_HCF) / len(sumRate_predicted_HCF)
                avgRATE_Predicted_HCF = sum(avgRate_predicted_HCF) / len(avgRate_predicted_HCF)

                scheduler_MT.step(sumRATE_Predicted_HCF)

                # Plots on Neptun Rate
                npt.send_metric('Sum Rate Value HBF', sumRATE_Predicted_HCF)
                npt.send_metric('Average Rate Value HBF', avgRATE_Predicted_HCF)

                print('Iter:==>{:3d} Loss_Class:{:.3f} sumRate_pre_HCBF:{:.2f} avgRate_pre_HCBF:{:.2f}'.
                    format(i, loss_clas, sumRATE_Predicted_HCF, avgRATE_Predicted_HCF))

else:
    raise Exception('BF_approach is wrong !!')
