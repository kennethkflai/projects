import sys
import os as os
import numpy as np
import matplotlib.pyplot as plt
import shutil as shutil
import matplotlib

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, QTimeLine, QPoint, QRect
from PyQt5 import QtCore
import itertools
# from scipy.misc import imresize
from keras.models import Sequential
from keras.utils import print_summary
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers import LSTM, Dense, Activation, Flatten, TimeDistributed,Lambda, Input, add,GlobalAveragePooling1D, concatenate, BatchNormalization, Conv1D, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from scipy import misc
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from util.data_process import data_process
import time

activation_custom = 'relu'
def precustom(layer):
    layer = BatchNormalization(axis=-1)(layer)
    layer = Activation(activation_custom)(layer)
    layer = GlobalAveragePooling1D()(layer)
    return layer

def convert_to_label(index):
    classes=["Base", "Stress", "Amuse"]
    return (classes[index])

def MLP(fc_units, t):
    t = Dense(fc_units)(t)
    t = BatchNormalization(axis=-1)(t)
    t = Activation(activation_custom)(t)
    t = Dropout(0.5)(t)
    t = Dense(fc_units)(t)
    t = BatchNormalization(axis=-1)(t)
    t = Activation(activation_custom)(t)
    t = Dropout(0.5)(t)
    return t


def TCN_Block(inp, activation_custom, vals, jump=True, length=8):
    t = Conv1D(vals[0], length, padding='same')(inp)

    def sub_block(activation_custom, fc_units, stride, inp, length):
        t1 = Conv1D(fc_units, 1, strides=stride, padding='same')(inp)
        t = BatchNormalization(axis=-1)(inp)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(stride), dilation_rate=1, padding='causal')(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same')(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), dilation_rate=2, padding='causal')(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same')(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), dilation_rate=4, padding='causal')(t)
        t = add([t1, t2])
        return t

    tout1 = sub_block(activation_custom, vals[0],1,t, length)
    tout2 = sub_block(activation_custom, vals[1],jump+1,tout1, length)
    tout3 = sub_block(activation_custom, vals[2],jump+1,tout2, length)
    tout4 = sub_block(activation_custom, vals[3],jump+1,tout3, length)

    return tout1, tout2, tout3, tout4

class Example(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def init(self):
        main_input = Input(shape=(240, 14))
        vals = [8, 16, 32, 64]
        skip = 6
        acc_x, acc_y, acc_z, ecg, eda, emg, resp, temp, wacc_x, wacc_y, wacc_z, wbvp, weda, wtemp  = Lambda(lambda x: tf.split(x,num_or_size_splits=14,axis=-1))(main_input)
        wacc = concatenate([wacc_x, wacc_y, wacc_z],)
        _, _, _, wacc = TCN_Block(wacc, activation_custom, vals, jump=True, length=skip)
        _, _, _, wbvp = TCN_Block(wbvp, activation_custom, vals, jump=True, length=skip)
        _, _, _, weda = TCN_Block(weda, activation_custom, vals, jump=True, length=skip)
        _, _, _, wtemp = TCN_Block(wtemp, activation_custom, vals, jump=True, length=skip)

        acc = concatenate([acc_x, acc_y, acc_z],)
        _, _, _, acc = TCN_Block(acc, activation_custom, vals, jump=True, length=skip)
        _, _, _, ecg = TCN_Block(ecg, activation_custom, vals, jump=True, length=skip)
        _, _, _, eda = TCN_Block(eda, activation_custom, vals, jump=True, length=skip)
        _, _, _, emg = TCN_Block(emg, activation_custom, vals, jump=True, length=skip)
        _, _, _, resp = TCN_Block(resp, activation_custom, vals, jump=True, length=skip)
        _, _, _, temp = TCN_Block(temp, activation_custom, vals, jump=True, length=skip)


        wacc = precustom(wacc)
        wbvp = precustom(wbvp)
        weda = precustom(weda)
        wtemp = precustom(wtemp)

        acc = precustom(acc)
        ecg = precustom(ecg)
        eda = precustom(eda)
        emg = precustom(emg)
        resp = precustom(resp)
        temp = precustom(temp)
        t = concatenate([acc, ecg, eda, emg, resp, temp, wacc, wbvp, weda, wtemp])
        t = MLP(1024, t)

        t = Dense(3)(t)
        t = Activation('softmax', name='t1')(t)

        self.model = Model(inputs=main_input, output=[t])

    def slider_pressed(self):
        print("rawr")

    def load_model(self):
        start = time.time()
        self.init()
        end = time.time()
        print('init model: ' + str(end - start))

        self.model_loaded = True
        self.weight_loaded = False

        self.NprobD.show()
        self.probD.show()


    def openDirectory(self):
#        self.reset_lbl()
        path_string=""
        fname = QFileDialog.getExistingDirectory(self, 'Open Directory', path_string )

        if len(fname)==0:
            self.ctext.setText("Invalid")
            return
        self.slider.valueChanged.connect(self.slider_pressed)
        self.slider.disconnect()
        self.appendDisplay=[]
        self.appendDisplay2=[]
        self.slider.setEnabled(False)
        self.disp.setEnabled(True)
        self.ctext.setText(fname)

        fname=self.ctext.toPlainText()
        self.subject_index = fname[fname.rfind('/')+1:]

#from util.data_process import data_process
        load_path = "data\\" + self.subject_index + ".npy"

        if os.path.exists(load_path):
            print("Load from process data")
            data = np.load(load_path, allow_pickle=True)
            self.wrist_data = data[0]
            self.chest_data = data[1]
            self.labels = data[2]
        else:
            start = time.time()
            data_model = data_process(fname)
            end = time.time()
            print('load data time: ' + str(end - start))
            self.wrist_data, self.chest_data, self.labels = data_model.get_data()
            np.save(load_path, [self.wrist_data, self.chest_data, self.labels] )

        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.labels))
        self.slider.setValue(1)
        self.slider.show()

        self.append_data = np.zeros((1,240,14))
        if self.model_loaded == True:
            subs = {'S2':0, 'S3':1, 'S4':2, 'S5':3, 'S6':4, 'S7':5, 'S8':6, 'S9':7, 'S10':8, 'S11':9,
                    'S13':10, 'S14':11, 'S15':12, 'S16':13, 'S17':14}

            weight_path = 'models//0-fusionFL-Chest_Wrist-cv' + str(subs[self.subject_index]) + '.hdf5'
            print(weight_path)
            self.model.load_weights(weight_path)
            self.weight_loaded = True
            self.call_recognition()

        self.timeLine.setFrameRange(0,len(self.labels))
        self.timeLine.setStartFrame(0)
        self.timeLine.setDuration(1000*(10000)*1/60)
        self.timeLine.start()

    def click_directory(self,event):
        self.openDirectory()

    def tl_finished(self):
        if self.model_loaded==True:
            self.call_recognition()
        self.slider.setEnabled(True)
        self.slider.valueChanged.connect(self.slider_pressed)

    def image_click(self,event):
        if(self.timeLine.state() == 1):
            self.timeLine.resume()
        elif (self.timeLine.state() == 2):
            self.timeLine.setPaused(True)
            if self.model_loaded==True:

                if self.weight_loaded == False:
                    start = time.time()
                    subs = {'S2':0, 'S3':1, 'S4':2, 'S5':3, 'S6':4, 'S7':5, 'S8':6, 'S9':7, 'S10':8, 'S11':9,
                        'S13':10, 'S14':11, 'S15':12, 'S16':13, 'S17':14}

                    weight_path = 'models//0-fusionFL-Chest_Wrist-cv' + str(subs[self.subject_index]) + '.hdf5'
                    self.model.load_weights(weight_path)
                    end = time.time()
                    print('load weights: ' + str(end - start))
                    self.weight_loaded = True
                self.call_recognition()

    def draw_signals(self, data_point, index):
        img = QPixmap(640, 480).toImage()
        painter= QPainter()
        painter.begin(img)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRect(0, 0, 640, 480)

        pen = QPen()
        pen.setWidth(1)

        for i in range(len(data_point[0])):
            min_pt = np.min(data_point[:,i])
            max_pt = np.max(data_point[:,i])
            for pos in range(0, len(data_point)-1):
                clr = QColor((self.labels[index+pos-len(data_point)]==1) *255,
                             (self.labels[index+pos-len(data_point)]==2) *255,
                             (self.labels[index+pos-len(data_point)]==3) *255)
                pen.setColor(clr);
                painter.setPen(pen)
                painter.setBrush(clr);

                starty = i*480/len(data_point[0]) + 480/len(data_point[0])*(data_point[pos,i]-min_pt)/(max_pt-min_pt)
                startx = 640*pos/len(data_point)

                endy = i*480/len(data_point[0]) + 480/len(data_point[0])*(data_point[pos+1,i]-min_pt)/(max_pt-min_pt)
                endx = 640*(pos+1)/len(data_point)

                painter.drawLine(int(startx), int(starty), int(endx), int(endy))

        pen.setWidth(3)

        pen.setColor(QColor((self.labels[index]==1) *255, (self.labels[index]==2) *255, (self.labels[index]==3) *255,50));
        painter.setPen(pen)
        painter.setBrush(QColor((self.labels[index]==1) *255, (self.labels[index]==2) *255, (self.labels[index]==3) *255,50))

        if index>240:
            start_x = 640*(1-(240/len(data_point)))
        else:
            start_x = 0
        painter.drawRect(start_x, 0, 640, 480)

        painter.end()
        self.disp.setPixmap(QPixmap(img))

    def preRecognition(self, index):
        cap = 1000
        chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']

        if self.chestCB.checkState() >0:
            for i in range(len(chest_sensor_keys)):
                if self.chest_array[i].checkState() >0:
                    data = self.chest_data[i]
                    break;
        else:
            for i in range(len(wrist_sensor_keys)):
                if self.wrist_array[i].checkState() >0:
                    data = self.wrist_data[i]
                    break;


        if index >cap:
            data_point = data[index-cap:index]
        else:
            data_point = data[0:index]

        self.draw_signals(data_point,index)

        append_data = []

        for j in range(len(self.chest_data)):
            data = self.chest_data[j]
            if self.chest_array_2[j].checkState()>0:
                for i in range(0,len(data[0])):
                    if index >240:
                        temp = data[index-240:index,i]
                    else:
                        temp = np.zeros((240,))
                        temp[240-index:] = data[:index,i]

                    append_data.append(temp.copy())
            else:
                for i in range(0,len(data[0])):
                    append_data.append(np.zeros(240,))

        for j in range(len(self.wrist_data)):
            data = self.wrist_data[j]
            if self.wrist_array_2[j].checkState()>0:
                for i in range(0,len(data[0])):
                    if index >240:
                        temp = data[index-240:index,i]
                    else:
                        temp = np.zeros((240,))
                        temp[240-index:] = data[:index,i]

                    append_data.append(temp.copy())
            else:
                for i in range(0,len(data[0])):
                    append_data.append(np.zeros(240,))

        append_data = np.array(append_data)
        append_data = append_data[np.newaxis,:,:]

        self.append_data = np.swapaxes(append_data,1,2)

    def call_recognition(self):
        probability = self.model.predict(self.append_data)

        probStr="<font color=white>Prediction: <br></font>"
        probStrD="<br>"
        clr = {0:"red",1:"green",2:"blue"}
        for i in range(0,len(probability[0])):
            str1="<font color=" + clr[i] + "> " + convert_to_label(i) + " ("+str(i+1)+"):" + " </font>"
            str2="<font color=" + clr[i] + "> " + "%5.2f" % (probability[0][i]*100) + " </font>"
            probStr= probStr + str1 + "<br>"
            probStrD= probStrD + str2 + "<br>"

        truth = self.labels[self.slider.value()]
        truth_probability = np.zeros((10))
        truth_probability[int(truth)]=1

        probStr= probStr +   "<br><br><font color=black>Truth: <br></font>"
        probStrD= probStrD + "<br><br><font color=black>"+ " ("+str(int(truth))+")" +  "<br></font>"

        probStr= probStr +  "<br><br><font color=white>Difference: <br></font>"
        probStrD= probStrD +  "<br><br><br>"

        for i in range(0,len(probability[0])):
            str1="<font color=" + clr[i] + "> " + convert_to_label(i) + " ("+str(i+1)+"):" + " </font>"
            str2="<font color=" + clr[i] + "> " + "%5.2f" % ((truth_probability[i+1]-probability[0][i])*100) + " </font>"
            probStr= probStr + str1 + "<br>"
            probStrD= probStrD + str2 + "<br>"

        self.NprobD.setText(probStr)
        self.probD.setText(probStrD)



    def process_directory(self,index):
        if(index ==0):
            return
        # print(index)
        self.ctext.toPlainText()

        self.preRecognition(index)

        if self.model_loaded==True:
            self.call_recognition()

        self.slider.setValue(index)

    def signal_type(self, cb):
        self.preRecognition(self.slider.value())
        self.call_recognition()

    def sensor_type(self, cb):
        chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']

        if cb.text() == 'Chest' and cb.isChecked():
            cb.setEnabled(False)
            self.wristCB.setCheckState(0)
            self.wristCB.setEnabled(True)
            for i in range(len(chest_sensor_keys)):
                self.chest_array[i].setEnabled(True)
            for i in range(len(wrist_sensor_keys)):
                self.wrist_array[i].setEnabled(False)

        if cb.text() == 'Wrist' and cb.isChecked():
            cb.setEnabled(False)
            self.chestCB.setCheckState(0)
            self.chestCB.setEnabled(True)

            for i in range(len(chest_sensor_keys)):
                self.chest_array[i].setEnabled(False)
            for i in range(len(wrist_sensor_keys)):
                self.wrist_array[i].setEnabled(True)

    def initUI(self):
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(255,255,255))
        self.setPalette(p)
        self.model_loaded = False

        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        impFileAct = QAction('Open File', self)
        impFileAct.setShortcut('Ctrl+O')
        impFileAct.setStatusTip('Open')
        impFileAct.triggered.connect(self.openDirectory)

        loadAct = QAction('Load Model', self)
        loadAct.setShortcut('Ctrl+L')
        loadAct.setStatusTip('Load Model')
        loadAct.triggered.connect(self.load_model)

        fileMenu.addAction(impFileAct)
        fileMenu.addAction(loadAct)
        fileMenu.addAction(exitAct)

        self.slider = QSlider(QtCore.Qt.Horizontal,self)
        self.slider.setGeometry(0,0,640,50)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.move(10,480+80)
        self.slider.setEnabled(False)
        self.slider.hide()

        self.disp = QLabel("", self)
        self.disp.setGeometry(0,0,640,480)
        self.disp.setAlignment(QtCore.Qt.AlignCenter)
        self.disp.setStyleSheet("background-color:#D5D8DC; border: transparent")
        self.disp.mousePressEvent = self.image_click
        self.disp.move(10,80)
        self.disp.setEnabled(False)

        dispRate= QLabel("Display:",self)
        dispRate.setGeometry(0,0,240,220)
        dispRate.setAlignment(QtCore.Qt.AlignTop)
        dispRate.setFont(QFont("Arial",20))
        dispRate.setStyleSheet("background-color:#F5EEF8; border: transparent")
        dispRate.move(660,40)

        self.NprobD= QTextEdit("",self)
        self.NprobD.setTextColor(QColor('#FFFFFF'))
        self.NprobD.setGeometry(0,0,140,520)
        self.NprobD.setAlignment(QtCore.Qt.AlignLeft)
        self.NprobD.setStyleSheet("background-color:#a6a6a6; border: transparent")
        self.NprobD.setReadOnly(True)
        self.NprobD.setFont(QFont("Arial",15))
        self.NprobD.move(910,40)

        self.probD= QTextEdit("",self)
        self.probD.setGeometry(0,0,100,520)
        self.probD.setAlignment(QtCore.Qt.AlignRight)
        self.probD.setStyleSheet("background-color:#a6a6a6;  border: transparent")
        self.probD.setReadOnly(True)
        self.probD.setFont(QFont("Arial",15))
        self.probD.move(910+140,40)

        self.NprobD.hide()
        self.probD.hide()

        self.timeLine = QTimeLine(10000,self)
        self.timeLine.setCurveShape(3)
        self.timeLine.frameChanged.connect(self.process_directory)
        self.timeLine.finished.connect(self.tl_finished)

        self.ctext = QTextEdit("", self)
        self.ctext.setGeometry(0,0,640,30)
        self.ctext.setReadOnly(True)
        self.ctext.mousePressEvent = self.click_directory
        self.ctext.move(10,40)

        self.chestCB = QCheckBox('Chest', self)
        self.chestCB.setFont(QFont("Arial",10))
        self.chestCB.move(700, 80)
        self.wristCB = QCheckBox('Wrist', self)
        self.wristCB.setFont(QFont("Arial",10))
        self.wristCB.move(800, 80)

        self.chestCB.setEnabled(False)
        self.chestCB.setCheckState(2)
        self.wristCB.setEnabled(True)

        self.chestCB.stateChanged.connect(lambda:self.sensor_type(self.chestCB))
        self.wristCB.stateChanged.connect(lambda:self.sensor_type(self.wristCB))

        chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']

        self.chest_array = []
        for i in range(len(chest_sensor_keys)):
            self.chest_array.append(QCheckBox(chest_sensor_keys[i], self))
            self.chest_array[i].move(700, 120+20*i)
            self.chest_array[i].setEnabled(True)


        self.chest_array[0].setCheckState(2)

        self.wrist_array = []
        for i in range(len(wrist_sensor_keys)):
            self.wrist_array.append(QCheckBox(wrist_sensor_keys[i], self))
            self.wrist_array[i].move(800, 120+20*i)
            self.wrist_array[i].setEnabled(False)

        self.wrist_array[0].setCheckState(2)


        dispRate2= QLabel("Input:",self)
        dispRate2.setGeometry(0,0,240,220)
        dispRate2.setAlignment(QtCore.Qt.AlignTop)
        dispRate2.setFont(QFont("Arial",20))
        dispRate2.setStyleSheet("background-color:#ebc3ab; border: transparent")
        dispRate2.move(660,340)

        self.chest_array_2 = []
        for i in range(len(chest_sensor_keys)):
            self.chest_array_2.append(QCheckBox(chest_sensor_keys[i], self))
            self.chest_array_2[i].move(700, 400+20*i)
            self.chest_array_2[i].setEnabled(True)
            self.chest_array_2[i].setCheckState(2)
            self.chest_array_2[i].stateChanged.connect(lambda:self.signal_type(self.wrist_array[i]))

        self.wrist_array_2 = []
        for i in range(len(wrist_sensor_keys)):
            self.wrist_array_2.append(QCheckBox(wrist_sensor_keys[i], self))
            self.wrist_array_2[i].move(800, 400+20*i)
            self.wrist_array_2[i].setEnabled(True)
            self.wrist_array_2[i].setCheckState(2)
            self.wrist_array_2[i].stateChanged.connect(lambda:self.signal_type(self.wrist_array[i]))


        self.setGeometry(200, 200, 1200, 650)
        self.setWindowTitle('Stress Detection')
        self.show()

if __name__ == '__main__':

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    ex = Example()
    app.quit()
    sys.exit(app.exec_())