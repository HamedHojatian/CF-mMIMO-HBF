import torch.nn as nn
from pytorch_complex_tensor import ComplexTensor
import torch
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            noise = torch.tensor(0).type(torch.float).to(x.device)
            scale = self.sigma * x.detach().to(x.device) if self.is_relative_detach else self.sigma * x.to(x.device)
            sampled_noise = noise.repeat(*x.size()).normal_() * scale.to(x.device)
            x = x + sampled_noise.to(x.device)
        return x

class Net_FD_CNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out_Reg, Codebook_len, p_dropout, U, Ass, out_channel, kernel_s, padding):
        super(Net_FD_CNN, self).__init__()

        self.cnn11 = nn.Conv2d(in_channels=n_in, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu11 = nn.LeakyReLU()
        self.bn11 = nn.BatchNorm2d(num_features=out_channel)
        self.do11 = nn.Dropout2d(p_dropout)

        self.cnn21 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu21 = nn.LeakyReLU()
        self.bn21 = nn.BatchNorm2d(num_features=out_channel)
        self.do21 = nn.Dropout2d(p_dropout)

        self.cnn12 = nn.Conv2d(in_channels=n_in, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu12 = nn.LeakyReLU()
        self.bn12 = nn.BatchNorm2d(num_features=out_channel)
        self.do12 = nn.Dropout2d(p_dropout)

        self.cnn22 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu22 = nn.LeakyReLU()
        self.bn22 = nn.BatchNorm2d(num_features=out_channel)
        self.do22 = nn.Dropout2d(p_dropout)

        self.cnn13 = nn.Conv2d(in_channels=n_in, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu13 = nn.LeakyReLU()
        self.bn13 = nn.BatchNorm2d(num_features=out_channel)
        self.do13 = nn.Dropout2d(p_dropout)

        self.cnn23 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu23 = nn.LeakyReLU()
        self.bn23 = nn.BatchNorm2d(num_features=out_channel)
        self.do23 = nn.Dropout2d(p_dropout)

        self.cnn14 = nn.Conv2d(in_channels=n_in, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu14 = nn.LeakyReLU()
        self.bn14 = nn.BatchNorm2d(num_features=out_channel)
        self.do14 = nn.Dropout2d(p_dropout)

        self.cnn24 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu24 = nn.LeakyReLU()
        self.bn24 = nn.BatchNorm2d(num_features=out_channel)
        self.do24 = nn.Dropout2d(p_dropout)

        # Fully connected 1 (readout)
        x_new = (U + 2 * padding - kernel_s) + 1
        y_new = (Ass + 2 * padding - kernel_s) + 1

        nn_in_fc = out_channel * (x_new + 2 * padding - kernel_s + 1) * (y_new + 2 * padding - kernel_s + 1)

        self.fc61 = nn.Linear(nn_in_fc, n_hidden)
        self.bn61 = nn.BatchNorm1d(n_hidden)
        self.relu61 = nn.LeakyReLU()
        self.do61 = nn.Dropout(p_dropout)

        self.fc62 = nn.Linear(nn_in_fc, n_hidden)
        self.bn62 = nn.BatchNorm1d(n_hidden)
        self.relu62 = nn.LeakyReLU()
        self.do62 = nn.Dropout(p_dropout)

        self.fc63 = nn.Linear(nn_in_fc, n_hidden)
        self.bn63 = nn.BatchNorm1d(n_hidden)
        self.relu63 = nn.LeakyReLU()
        self.do63 = nn.Dropout(p_dropout)

        self.fc64 = nn.Linear(nn_in_fc, n_hidden)
        self.bn64 = nn.BatchNorm1d(n_hidden)
        self.relu64 = nn.LeakyReLU()
        self.do64 = nn.Dropout(p_dropout)

        # Linear function (readout)
        self.fc61R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc61I = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc62R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc62I = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc63R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc63I = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc64R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc64I = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc91C = nn.Linear(n_hidden, Codebook_len[0])
        # Linear function (readout)
        self.fc92C = nn.Linear(n_hidden, Codebook_len[1])
        # Linear function (readout)
        self.fc93C = nn.Linear(n_hidden, Codebook_len[2])
        # Linear function (readout)
        self.fc94C = nn.Linear(n_hidden, Codebook_len[3])

    def forward(self, x1, x2, x3, x4):  # always

        out1 = self.cnn11(x1)
        out1 = self.do11(out1)
        out1 = self.bn11(out1)
        out1 = self.relu11(out1)

        out1 = self.cnn21(out1)
        out1 = self.do21(out1)
        out1 = self.bn21(out1)
        out1 = self.relu21(out1)

        out2 = self.cnn12(x2)
        out2 = self.do12(out2)
        out2 = self.bn12(out2)
        out2 = self.relu12(out2)

        out2 = self.cnn22(out2)
        out2 = self.do22(out2)
        out2 = self.bn22(out2)
        out2 = self.relu22(out2)

        out3 = self.cnn13(x3)
        out3 = self.do13(out3)
        out3 = self.bn13(out3)
        out3 = self.relu13(out3)

        out3 = self.cnn23(out3)
        out3 = self.do23(out3)
        out3 = self.bn23(out3)
        out3 = self.relu23(out3)

        out4 = self.cnn14(x4)
        out4 = self.do14(out4)
        out4 = self.bn14(out4)
        out4 = self.relu14(out4)

        out4 = self.cnn24(out4)
        out4 = self.do24(out4)
        out4 = self.bn24(out4)
        out4 = self.relu24(out4)

        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out3 = out3.view(out3.size(0), -1)
        out4 = out4.view(out4.size(0), -1)

        # Linear function  ****** LINEAR ******
        out1 = self.fc61(out1)
        out1 = self.bn61(out1)
        out1 = self.relu61(out1)
        out1 = self.do61(out1)

        # Linear function  ****** LINEAR ******
        out2 = self.fc62(out2)
        out2 = self.bn62(out2)
        out2 = self.relu62(out2)
        out2 = self.do62(out2)

        # Linear function  ****** LINEAR ******
        out3 = self.fc63(out3)
        out3 = self.bn63(out3)
        out3 = self.relu63(out3)
        out3 = self.do63(out3)

        # Linear function  ****** LINEAR ******
        out4 = self.fc64(out4)
        out4 = self.bn64(out4)
        out4 = self.relu64(out4)
        out4 = self.do64(out4)

        # Linear function (readout)  ****** LINEAR ******
        outR = torch.cat((self.fc61R(out1), self.fc62R(out2), self.fc63R(out3), self.fc64R(out4)), dim=1)

        # Linear function (readout)  ****** LINEAR ******
        outI = torch.cat((self.fc61I(out1), self.fc62I(out2), self.fc63I(out3), self.fc64I(out4)), dim=1)

        # Linear function (readout)  ****** LINEAR ******
        outC1 = self.fc91C(out1)

        # Linear function (readout)  ****** LINEAR ******
        outC2 = self.fc92C(out2)

        # Linear function (readout)  ****** LINEAR ******
        outC3 = self.fc93C(out3)

        # Linear function (readout)  ****** LINEAR ******
        outC4 = self.fc94C(out4)

        return outR, outI, F.softmax(outC1, 1), F.softmax(outC2, 1), F.softmax(outC3, 1), F.softmax(outC4, 1)

class Net_PD_CNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out_Reg, Codebook_len, p_dropout, U, Ass, out_channel, kernel_s, padding):
        super(Net_PD_CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=4, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.do1 = nn.Dropout2d(p_dropout)

        self.cnn2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.do2 = nn.Dropout2d(p_dropout)

        # Fully connected 1 (readout)
        x_new = (U + 2 * padding - kernel_s) + 1
        y_new = (Ass + 2 * padding - kernel_s) + 1

        nn_in_fc = out_channel * (x_new + 2 * padding - kernel_s + 1) * (y_new + 2 * padding - kernel_s + 1)

        self.fc6 = nn.Linear(nn_in_fc, int(n_hidden / 8))
        self.bn6 = nn.BatchNorm1d(int(n_hidden / 8))
        self.relu6 = nn.LeakyReLU()
        self.do6 = nn.Dropout(p_dropout)

        # Linear Function
        self.fc51 = nn.Linear(int(n_hidden / 8), n_hidden)
        self.bn51 = nn.BatchNorm1d(n_hidden)
        self.relu51 = nn.LeakyReLU()
        self.do51 = nn.Dropout(p_dropout)

        # Linear Function
        self.fc52 = nn.Linear(int(n_hidden / 8), n_hidden)
        self.bn52 = nn.BatchNorm1d(n_hidden)
        self.relu52 = nn.LeakyReLU()
        self.do52 = nn.Dropout(p_dropout)

        # Linear Function
        self.fc53 = nn.Linear(int(n_hidden / 8), n_hidden)
        self.bn53 = nn.BatchNorm1d(n_hidden)
        self.relu53 = nn.LeakyReLU()
        self.do53 = nn.Dropout(p_dropout)

        # Linear Function
        self.fc54 = nn.Linear(int(n_hidden / 8), n_hidden)
        self.bn54 = nn.BatchNorm1d(n_hidden)
        self.relu54 = nn.LeakyReLU()
        self.do54 = nn.Dropout(p_dropout)

        # Linear function (readout)
        self.fc61R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc61I = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc62R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc62I = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc63R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc63I = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc64R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc64I = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc91C = nn.Linear(n_hidden, Codebook_len[0])
        # Linear function (readout)
        self.fc92C = nn.Linear(n_hidden, Codebook_len[1])
        # Linear function (readout)
        self.fc93C = nn.Linear(n_hidden, Codebook_len[2])
        # Linear function (readout)
        self.fc94C = nn.Linear(n_hidden, Codebook_len[3])

    def forward(self, x):  # always

        out = self.cnn1(x)
        out = self.do1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.cnn2(out)
        out = self.do2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = out.view(out.size(0), -1)

        # Linear function  ****** LINEAR ******
        out = self.fc6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.do6(out)

        # Linear function  ****** LINEAR ******
        out1 = self.fc51(out)
        out1 = self.bn51(out1)
        out1 = self.relu51(out1)
        out1 = self.do51(out1)

        # Linear function  ****** LINEAR ******
        out2 = self.fc52(out)
        out2 = self.bn52(out2)
        out2 = self.relu52(out2)
        out2 = self.do52(out2)

        # Linear function  ****** LINEAR ******
        out3 = self.fc53(out)
        out3 = self.bn53(out3)
        out3 = self.relu53(out3)
        out3 = self.do53(out3)

        # Linear function  ****** LINEAR ******
        out4 = self.fc54(out)
        out4 = self.bn54(out4)
        out4 = self.relu54(out4)
        out4 = self.do54(out4)

        # Linear function (readout)  ****** LINEAR ******
        outR = torch.cat((self.fc61R(out1), self.fc62R(out2), self.fc63R(out3), self.fc64R(out4)), dim=1)

        # Linear function (readout)  ****** LINEAR ******
        outI = torch.cat((self.fc61I(out1), self.fc62I(out2), self.fc63I(out3), self.fc64I(out4)), dim=1)

        # Linear function (readout)  ****** LINEAR ******
        outC1 = self.fc91C(out1)

        # Linear function (readout)  ****** LINEAR ******
        outC2 = self.fc92C(out2)

        # Linear function (readout)  ****** LINEAR ******
        outC3 = self.fc93C(out3)

        # Linear function (readout)  ****** LINEAR ******
        outC4 = self.fc94C(out4)

        return outR, outI, F.softmax(outC1, 1), F.softmax(outC2, 1), F.softmax(outC3, 1), F.softmax(outC4, 1)
