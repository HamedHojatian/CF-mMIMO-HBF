# Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised Learning

In this repository you can find the simulation source code of: "[Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised Learning](<https://128.84.4.60/abs/2106.16194>)".


## Channel model

A realistic ray-tracing channel model is considered to evaluate the proposed solution. It has been introduced by Alkhateeb, et al, in "[DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications](<https://arxiv.org/abs/1902.06435>)"


## Content

**1.DATASET.md:** all parameters related to system model such as number of users, number of antennas, etc.

**2.Codebook:** designed codebook for each BS (4,5,8,9) in deepMIMO chanel model is available in the zip file.

**3..py files:** simulation source codes


## Dataset
The dataset where 4 APs with 64 antenna and 8 RF chains serving 4 single antenna users. We consider BS number 4,5,8,9 is active and other information is in the paper. The dataset name should be "dataSet_130.npy". The RSSI value must be normalized and the order of data in .npy can be found in codes (import .npy).

## Requirements
1. torch 1.8.0 (Support Complex Tensor)
2. numpy 1.19.2

## Copyright
Feel free to use this code as a starting point for your own research project. If you do, we kindly ask that you cite the following paper: "[Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised Learning](<https://128.84.4.60/abs/2106.16194>)".

```
@ARTICLE{9729183,
  author={Hojatian, Hamed and Nadal, Jérémy and Frigon, Jean-François and Leduc-Primeau, François},
  journal={IEEE Communications Letters}, 
  title={Decentralized Beamforming for Cell-Free Massive MIMO with Unsupervised Learning}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LCOMM.2022.3157161}}

```
Copyright (C): GNU General Public License v3.0 or later
