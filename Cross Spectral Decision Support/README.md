# Cross Spectral Biometric Decision Support

Script provided is in relation with the following paper:

- K. Lai, S. Yanushkevich, and V. Shmerko, [Reliability of Decision Support in Cross-spectral Biometric-enabled Systems](https://ieeexplore.ieee.org/document/9283460), IEEE International Conference on Systems, Man, and Cybernetics, October 2020, pp. 1-6.

## Dataset
The experiment in this paper is based on the [Tufts Face Database](http://tdface.ece.tufts.edu/). A description of the dataset can be found in:
 
*K. Panetta, Q. Wan, S. Agaian, S. Rajeev, S. Kamath, R. Rajendran, S. Rao, A. Kaszowska, H. Taylor, A. Samani et al., "A comprehensive database for benchmarking imaging systems," IEEE Trans. on Pattern
Analysis and Machine Intelligence, 2018.*

## Setup
Libraries:
- numpy 1.18.1
- keras 2.2.4
- tensorflow 1.13.1
- scipy 1.2.1
- pillow 5.4.1
- opencv 3.4.1
- opencv-python 4.2.0.32
- matplotlib 3.0.3

Requires additional installation of YOLO for face detection and vggface for face recognition.

- [YOLO](https://github.com/sthanhng/yoloface/tree/master/yolo)
- [keras_vggface](https://github.com/rcmalli/keras-vggface/tree/master/keras_vggface)

## Usage
To run face detection and face recognition:
```
python3 det_rec.py
```

To run emotion recognition:
```
python3 emotion.py
```

Analyze and output results:
```
python3 result.py
```