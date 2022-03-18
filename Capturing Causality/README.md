# Capturing Causality in Bias in Human Action Recognition

Script provided is in relation with the following paper:

- K. Lai, S. Yanushkevich, V. Shmerko, and M. Hou, [Capturing Causality in Bias in Human Action Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0167865521001380), in Pattern Recognition Letters, vol. 147, July 2021, pp. 164-171.

Architure used in the paper is as follows:

[](tcn.pdf)

The Res-TCN network is composed of four block of residual units. Each residual unit is composed of three sets of sub-blocks where each sub-block is the combination of Batch Normalization (BatchNorm), Rectified Linear Unit (ReLu), and Convolution layers. The sub-block structure is illustrated in the Figure above. Res-U(32, 8, 1) represents a sub-block containing a convolutional layer with 32 filters (F = 32), filter size of 8 (K = 8), and stride of 1 (S = 1)

## Dataset
The experiment in this paper is based on the [FALL-UP Dataset](https://sites.google.com/up.edu.mx/har-up/). A description of the dataset can be found in:
 
*Lourdes Martínez-Villaseñor, Hiram Ponce, Jorge Brieva, Ernesto Moya-Albor, José Núñez-Martínez, Carlos Peñafort-Asturiano, “UP-Fall Detection Dataset: A Multimodal Approach”, Sensors 19(9), 1988: 2019, doi:10.3390/s19091988.*

## Setup
Libraries:
- numpy 1.18.1
- keras 2.2.4
- tensorflow 1.13.1
- scipy 1.2.1
- pillow 5.4.1
- opencv 3.4.1
- opencv-python 4.2.0.32

Download the [CompleteDataSet.csv](https://drive.google.com/file/d/1JBGU5W2uq9rl8h7bJNt2lN4SjfZnFxmQ/view) from the FALL-UP Dataset and place with root files.

To run with default parameters:
```
python3 fall.py
```

or custom parameters:
```
python3 fall.py --timestep=20 --cycle=10 --base=100 --mult=1 --modeltype=0 --lr=0.001 --wd=0.
```
Timestep is the number frames to be used, cycle is the number of times to restart cycle based on Cosine annealing, base is the number of epoches in a cycle, mult is the multiplier to increase number of epoches after each cycle, lr is the learning rate, and wd is the weight decay.
