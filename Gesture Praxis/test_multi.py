from glob import glob
import csv
import numpy as np

import keras
from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, CuDNNLSTM, LSTM, Activation, Dropout, add, Input, Dense, GlobalAveragePooling1D
from keras.regularizers import L1L2
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score
from sklearn.metrics import classification_report
l1 = 0
l2 = 0
activation_custom = 'relu'

def precustom(layer):
    layer = BatchNormalization(axis=-1)(layer)
    layer = Activation(activation_custom)(layer)
    layer = GlobalAveragePooling1D()(layer)
    return layer


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

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def TCN_Block(inp, activation_custom, vals, jump=True, length=8):
    t = Conv1D(vals[0], length, padding='same')(inp)

    def sub_block(activation_custom, fc_units, stride, inp, length):
        t1 = Conv1D(fc_units, 1, strides=stride, padding='same', kernel_regularizer=L1L2(l1, l2))(inp)
        t = BatchNormalization(axis=-1)(inp)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(stride), dilation_rate=1, padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same', kernel_regularizer=L1L2(l1, l2))(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), dilation_rate=2, padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same', kernel_regularizer=L1L2(l1, l2))(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), dilation_rate=4, padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])
        return t

    tout1 = sub_block(activation_custom, vals[0],1,t, length)
    tout2 = sub_block(activation_custom, vals[1],jump+1,tout1, length)
    tout3 = sub_block(activation_custom, vals[2],jump+1,tout2, length)
    tout4 = sub_block(activation_custom, vals[3],jump+1,tout3, length)

    return tout1, tout2, tout3, tout4


def draw_cm(trueP, modelP, nm, cmap=plt.cm.Purples):
    import itertools
    classes = [i for i in range(len(np.unique(trueP)))]



    matrix = confusion_matrix(trueP, modelP)
    np.set_printoptions(precision=8)
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    matrix = matrix.astype('float') / matrix.sum(axis=1)[ np.newaxis, :] * 100

    plt.imshow(matrix, interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, fontsize=5)
    plt.yticks(tick_marks, classes, fontsize=5)
    thresh = matrix.max()/2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        if matrix[i, j] > thresh:
            clr = "white"
        else:
            clr = "black"
        plt.text(j, i, format(matrix[i, j], '2.1f'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color=clr, fontsize=5)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(nm) as pdf:
        pdf.savefig(fig,bbox_inches='tight')

data_path = r'Data\\*\\Skeleton\\*'
label_path = r'labels_time_stamps_release\\*'

data_list = glob(data_path)
label_list = glob(label_path)

def toTime(string):
    components = string.split('-')
    components[0] = float(components[0])*3600
    components[1] = float(components[1])*60
    components[2] = float(components[2])
    return float(components[0] + components[1] + components[2])


toAction = {'A1_1':0, 'A1_2':1, 'A1_3':2, 'A1_4':3, 'A1_5':4,
               'A2_1':5, 'A2_2':6, 'A2_3':7, 'A2_4':8, 'A2_5':9,
               'S1_1':10, 'S1_2':11, 'S1_3':12, 'S1_4':13, 'S1_5':14,
               'S2_1':15, 'S2_2':16, 'S2_3':17, 'S2_4':18,
               'P1_1':19, 'P1_2':20, 'P1_3':21, 'P1_4':22, 'P1_5':23,
               'P2_1':24, 'P2_2':25, 'P2_3':26, 'P2_4':27, 'P2_5':28}

dynamic_gestures = {'0':0, '1':0, '2':0, '3':0, '4':0,
                           '5':0, '6':1, '7':0, '8':0, '9':0,
                           '10':0, '11':0, '12':0, '13':1, '14':1,
                           '15':1, '16':0, '17':0, '18':0,
                           '19':1, '20':1, '21':1, '22':1, '23':1,
                           '24':1, '25':1, '26':1, '27':1, '28':1}
subject_info = []

for path in label_list:
    with open (path, 'r') as file:
        reader = csv.reader(file)
        label_info = []
        for row in reader:
            info = {'Action': toAction[row[0]], 'Start': toTime(row[1]), 'End': toTime(row[2]), 'Correct': int(row[3]), 'Type': row[4]}
            label_info.append(info)
    subject_info.append(label_info)

def getAction(time, info):
    for action in info:
        if time>= action['Start'] and time<=action['End']:
            return action['Action']
    return -1

def toNumber(string):
    store = []
    for ss in string:
        index = ss.find("e")
        base = float(ss[:index])
        expo = int(ss[index+1:])
        store.append(np.float32(base * 10**expo))
    return store

if not os.path.isfile('data.npy'):
    subject_data=[[[] for j in range(29)] for i in range(len(subject_info))]
    previous_label = 0
    data=[]
    from tqdm import tqdm
    prog = tqdm(range(0, len(data_list)))

    for path_index in prog:
        path = data_list[path_index]
        subject = int(path[path.find('_')+1:path.find('_')+3])-1
        time_string = path[path.rfind('T')+1:path.rfind('-')]
        time = toTime(time_string)

        label = getAction(time, subject_info[subject])

        with open(path, 'r') as f:
            lines = f.readlines()
            temp_data=[]
            for index in range(len(lines)):
                temp_string = lines[index].split()[6:14]
                temp_data.append(toNumber(temp_string))
            temp_data = np.array(temp_data)

            if label != previous_label:

                if data != []:
                    data = np.array(data)
                    for i in range(len(data[0])):
                        for j in range(len(data[0,i])):
                            data[:,i,j] = savgol_filter(data[:,i,j],5,2)
                    subject_data[subject][previous_label].append(np.array(data))
                data = []

            data.append(temp_data)
            previous_label = label

    np.save('data.npy', subject_data)
else:
    subject_data = np.load('data.npy', allow_pickle=True)

#for structure in range(0,2):
#    for del_flag in range(0,2):
#        for angle_flag in range(0,5):
#            for ts in range(5, 10):
#                for dyn in range(0,3):
for structure in range(0,1):
    for del_flag in range(0,1):
        for angle_flag in range(2,3):
            for ts in range(7, 8):
                for dyn in range(2,3):
                    number_frame = 2**ts
                    skip = 1
                    train_data=[]
                    train_label=[]
                    test_data=[]
                    test_label=[]

                    train_subject = [i for i in range(16)]
                    for sub in range(len(subject_data)):
                        for gest in range(len(subject_data[sub])):
                            if dyn <2:
                                if dynamic_gestures[str(gest)] == dyn:
                                    continue
                            for trial in range(len(subject_data[sub][gest])):
                                temp = subject_data[sub][gest][trial].copy()

                                x_n = temp[0][0][6]
                                y_n = temp[0][1][6]
                                temp = np.delete(temp, 2, 1)
                                if del_flag == 0:
                                    temp = np.delete(temp, -1, 1)
                                    temp = np.delete(temp, -1, 1)

                                if angle_flag == 0:
                                    temp[:,0]=np.sqrt((temp[:,0]-x_n)**2+ (temp[:,1]-y_n) **2)
                                    temp[:,1]=np.arctan2((temp[:,0]-x_n), (temp[:,1]-y_n))
                                elif angle_flag == 1:
                                    temp[:,0]=(temp[:,0]-x_n)/x_n
                                    temp[:,1]=(temp[:,1]-y_n)/y_n
                                elif angle_flag == 2:
                                    temp[:,0]=(temp[:,0]-x_n)
                                    temp[:,1]=(temp[:,1]-y_n)
                                elif angle_flag == 3:
                                    n_temp = np.zeros((len(temp), len(temp[0])+2, len(temp[0][0])))
                                    n_temp[:,2]=(temp[:,0]-x_n)/x_n
                                    n_temp[:,3]=(temp[:,1]-y_n)/y_n
                                    n_temp[:,0]=np.sqrt((temp[:,0]-x_n)**2+ (temp[:,1]-y_n) **2)
                                    n_temp[:,1]=np.arctan2((temp[:,0]-x_n), (temp[:,1]-y_n))
                                    temp = n_temp
                                elif angle_flag == 4:
                                    n_temp = np.zeros((len(temp), len(temp[0])+2, len(temp[0][0])))
                                    n_temp[:,2]=(temp[:,0]-x_n)
                                    n_temp[:,3]=(temp[:,1]-y_n)
                                    n_temp[:,0]=np.sqrt((temp[:,0]-x_n)**2+ (temp[:,1]-y_n) **2)
                                    n_temp[:,1]=np.arctan2((temp[:,0]-x_n), (temp[:,1]-y_n))
                                    temp = n_temp

                                for index in range(0, len(temp)-number_frame, skip):
                                    t = temp[index:index+number_frame]
                                    t = np.reshape(t, (len(t), len(t[0])*len(t[0][0])))

                                    if sub not in train_subject:
                                        train_data.append(t)
                                        train_label.append(gest)
                                    else:
                                        test_data.append(t)
                                        test_label.append(gest)

                                if len(temp)-number_frame <=0:
                                    t = np.zeros((number_frame, len(temp[0]),len(temp[0][0])))
                                    t[number_frame-len(temp):] = temp
                                    t = np.reshape(t, (len(t), len(t[0])*len(t[0][0])))

                                    if sub not in train_subject:
                                        train_data.append(t)
                                        train_label.append(gest)
                                    else:
                                        test_data.append(t)
                                        test_label.append(gest)

                    train_data = np.array(train_data)
                    test_data = np.array(test_data)

                    total_classes = len(np.unique(train_label))

                    temp_vector = np.unique(train_label)
                    remap = {temp_vector[i]: i for i in range(total_classes)}
#                    total_classes = 2
                    new_label = []
                    for index in range(len(train_label)):
                        new_label.append(remap[train_label[index]])

                    train_label = new_label.copy()

                    new_label = []
                    for index in range(len(test_label)):
                        new_label.append(remap[test_label[index]])

                    test_label = new_label.copy()

                    save_train_label = train_label.copy()
                    save_test_label = test_label.copy()
                    train_label=[]
                    test_label=[]

                    for classes in range(total_classes):
                        train_label = (np.where(np.array(save_train_label) == classes, 1,0))
                        test_label = (np.where(np.array(save_test_label) == classes, 1,0))
                        number_classes = len(np.unique(train_label))

                        main_input = Input(shape=(len(train_data[0]), len(train_data[0][0])))
                        if structure == 0:
                            t = CuDNNLSTM(256, return_sequences=True)(main_input)
                            t = CuDNNLSTM(256, return_sequences=False)(t)
                            t = BatchNormalization(axis=-1)(t)#before here
                        elif structure == 1:
                            vals = [4, 8, 16, 32]
                            _, _, _, t = TCN_Block(main_input, 'relu', vals, jump=True, length=6)
                            t = precustom(t)

                        t = MLP(1024, t)
                        t = Dense(number_classes, activation='softmax', name='out')(t)

                        model = Model(inputs=main_input, output=t)
                        model.summary()

                        model.compile(loss=keras.losses.categorical_crossentropy,
                                      optimizer=keras.optimizers.Adam(),
                                      metrics=[])


                        filepath = 'save\\'
                        os.makedirs(filepath, exist_ok=True)
                        filepath = 'save\\' + str(angle_flag) + 'best' + str(number_frame) + '.hdf5'
                        train_label = list(train_label)
                        cw = {ii: len(train_label)/len(np.unique(train_label)) * (1/((train_label.count(ii))))
                                    for ii in range(len(np.unique(train_label)))}
                        train_label = np.array(train_label)

                        train_label_hot = (keras.utils.to_categorical(train_label, number_classes))
                        test_label_hot = (keras.utils.to_categorical(test_label, number_classes))

                        class MetricsCallback(keras.callbacks.Callback):
                            def __init__(self):
                                super(MetricsCallback, self).__init__()
                            def  on_train_begin(self,logs={}):
                              self.f1_macro=[]
                            def on_epoch_end(self, epoch, logs=None):
                              y_pred=self.model.predict(test_data)
                              y_true=test_label_hot

                              score=classification_report(np.argmax(y_true,1), np.argmax(y_pred,1),output_dict=True)
                              print(classes, ":", score['1']['precision']*100)


                        metrics=MetricsCallback()

                        checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc',
                                                     verbose=0, save_best_only=True, save_weights_only=True,
                                                     mode='max')

                        early=EarlyStopping(monitor='loss', patience=10,verbose=0,mode='auto')
                        callbacks_list = [early, checkpoint1, metrics]



                        model.fit(train_data, train_label_hot, shuffle=True,
                                  batch_size=128,
                                  epochs=10000,
                                  verbose=1,class_weight=cw, #validation_data=(test_data, test_label_hot),
                                  callbacks=callbacks_list)

    #                    model.load_weights(filepath)
                        prediction = model.predict(test_data)

    #                        acc = np.sum(np.argmax(prediction,1)==test_label)/len(test_label)
    #                        f = open('acc.txt', 'a')
    #                        f.write("struct: %d, dyn: %d, df: %d, angle_flag: %d, tr: %d, frame: %d, it: %d, Accuracy: %.2f%%\n"
    #                                % (structure, dyn, del_flag, angle_flag, len(train_label), number_frame, 1024, acc*100))
    #                        f.close()



                        acc = classification_report(test_label, np.argmax(prediction,1),output_dict=True)


                        print("avg: " + str(acc['1']['precision']))

                        f = open('acc.txt', 'a')
                        f.write("struct: %d, dyn: %d, df: %d, angle_flag: %d, tr: %d, frame: %d, it: %d, Accuracy: %.2f%%\n"
                                % (structure, dyn, del_flag, angle_flag, len(train_label), number_frame, 1024, acc['1']['precision']))
                        f.close()
                        #draw_cm(test_label, np.argmax(prediction,1), f"fr{number_frame}_p{0}.pdf")

                        del model
                        K.clear_session()