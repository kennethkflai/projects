# Stress Demo

Script provided is in relation with the following paper:

- K. Lai, S. N. Yanushkevich and V. P. Shmerko, [Intelligent Stress Monitoring Assistant for First Responders](https://ieeexplore.ieee.org/document/9348878), in IEEE Access, vol. 9, February 2021, pp. 25314-25329.

Architure used in the paper is as follows:
![](tcn.png)

## Dataset
The experiment in this paper is based on the [WESAD Dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29). A description of the dataset can be found in:

*P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, and K. Van Laerhoven, “Introducing wesad, a multimodal dataset for wearable stress and affect detection,” Proc. of the Int. Conf. on Multimodal Interaction, pp. 400–408, 2018.*

## Setup
Libraries:
- numpy 1.18.1
- keras 2.2.4
- tensorflow 1.13.1
- scipy 1.2.1
- pillow 5.4.1
- opencv 3.4.1
- opencv-python 4.2.0.32
- pyqt 5.9.2

## Usage
To run:
```
python3 gui.py
```

Load Model 
```
File->Load Model
```

Open File
```
File->Open File or click white box
Navigate to WESAD dataset
```

![](disp.png)
Display window controls what signal is being drawn on the window to the left.

![](inp.png)
Input window controls what signals are being used by the machine-learning model for prediction.

![](pred.png)
Prediction window shows the prediction by the machine-learning model as well as the ground-truth label.