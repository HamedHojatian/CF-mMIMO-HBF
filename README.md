# Limited-Fronthaul Cell-Free Hybrid Beamforming with Distributed Neural Networks

In this repository you can find the simulation source code of: "Limited-Fronthaul Cell-Free Hybrid Beamforming with Distributed Neural Networks".


## Channel model

A realistic ray-tracing channel model is considered to evaluate the proposed solution. It has been introduced by Alkhateeb, et al, in "[DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications](<https://arxiv.org/abs/1902.06435>)"


## Content

**1.DATASET.md:** all parameters related to system model such as number of users, number of antennas, etc.

**2.Codebook:** designed codebook using the proposed algorithm in the paper.

**3..py files:** simulation source codes


## Dataset
**dataSet_withRSSI_32SSB_Ser.npy:** is the dataset where 4 APs with 64 antenna and 8 RF chains serving 4 single antenna users. You can find it [here](https://drive.google.com/drive/folders/1I07LB--sRGPCgWl6Ci_VeveKSzeeQyKK).

It is the core dataset with 1e6 samples. It consist of RSSI, channel.

## Requirements
1. torch 1.8.0
2. numpy 1.19.2

## Copyright
Feel free to use this code as a starting point for your own research project. If you do, we kindly ask that you cite the following paper: "Limited-Fronthaul Cell-Free Hybrid Beamforming with Distributed Neural Networks".


Copyright (C): GNU General Public License v3.0 or later
