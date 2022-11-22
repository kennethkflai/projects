# Gesture Classification

Script provided is in relation with the following papers:

- K Lai, V. Shmerko, and S. Yanushkevich, Measuring Fairness of Deep Neural Network Classifiers for Face Biometrics, International Conference on Cognitive Machine Intelligence, December 2022.
 
## Dataset
The experiment in this paper is based on the [SpeakingFaces Dataset](https://github.com/IS2AI/SpeakingFaces) and [Thermal-Mask Dataset](https://ieeexplore.ieee.org/document/9497521). A description of the dataset can be found in:

*M. Abdrakhmanova, A. Kuzdeuov, S. Jarju, Y. Khassanov, M. Lewis, and H. A. Varol, “Speakingfaces: A large-scale multimodal dataset of voice commands with visual and thermal video streams,” Sensors, vol. 21, no. 10, p. 3465, 2021.*

*L. Queiroz, H. Oliveira, and S. Yanushkevich, “Thermal-mask–a dataset for facial mask detection and breathing rate measurement,” in International Conference on Information and Digital Technologies (IDT),
2021, pp.142-151*

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
- scikit-learn 0.21.3

## Usage
Initiate training for Thermal-Mask dataset:
```
python3 run-ffhq-masknet.py
```

Initiate training for SpeakingFaces dataset:
```
python3 run-speakerface.py
```

Run metric calculations:
```
python3 metrics.py
```