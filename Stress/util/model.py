from keras.layers import Dense, Activation, GRU, Lambda, CuDNNLSTM
from keras.layers import multiply, add, concatenate, BatchNormalization
from keras.layers import Reshape, Input, LSTM, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Conv1D, SpatialDropout1D, Conv2D, MaxPooling2D, MaxPooling3D, Conv3D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from keras.utils import multi_gpu_model
from keras.layers import Flatten, TimeDistributed, maximum, SeparableConv1D,GlobalAveragePooling3D, average, GlobalAveragePooling1D, AveragePooling1D, Conv3D,SpatialDropout3D
from keras.regularizers import L1L2
from keras import backend
import numpy as np

import tensorflow as tf
l1=0#1e-4
l2=0#0.001
activation_custom = 'relu'
opti='adam'
def recall_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))


def MLP(fc_units, t):
    t = Dense(fc_units, kernel_regularizer=L1L2(l1, l2))(t)
    t = BatchNormalization(axis=-1)(t)
    t = Activation(activation_custom)(t)
    t = Dropout(0.5)(t)
    t = Dense(fc_units, kernel_regularizer=L1L2(l1, l2))(t)
    t = BatchNormalization(axis=-1)(t)
    t = Activation(activation_custom)(t)
    t = Dropout(0.5)(t)
    return t

class SnapshotEnsemble(Callback):
    # constructor
    def __init__(self, n_cycles, num, batch_size=512, cycle_mult_factor=1.5, verbose=0):
        self.steps_per_epoch = int(np.floor(num/batch_size))

        self.cycle_length = n_cycles
        self.cycle_mult_factor = cycle_mult_factor
        self.epoch_counter = 0

    def on_train_begin(self, logs={}):
        '''Set the number of training batches of the first restart cycle to steps_per_cycle'''
        try:
            backend.set_value(self.model.optimizer.steps_per_cycle, self.steps_per_epoch * self.cycle_length)
            backend.set_value(self.model.optimizer.t_cur, 0)
            self.epoch_counter = 0
        except:
            print()

    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        # check if we can save model
        self.epoch_counter += 1
        print(backend.get_value(self.model.optimizer.eta))
        if epoch != 0 and (self.epoch_counter) % self.cycle_length == 0:
            backend.set_value(self.model.optimizer.t_cur,0)

            self.cycle_length = np.ceil(self.cycle_length*self.cycle_mult_factor)
            self.epoch_counter = 0
            backend.set_value(self.model.optimizer.steps_per_cycle, self.steps_per_epoch * self.cycle_length)
            print('new cycle')



def lstm_block(size, units, inp):
    t = Reshape((size))(inp)
    t = LSTM(units, recurrent_regularizer=L1L2(l1, l2), recurrent_dropout=0.5, dropout=0.2, return_sequences=True)(t)
    t = LSTM(units, recurrent_regularizer=L1L2(l1, l2), recurrent_dropout=0.5, dropout=0.5, return_sequences=False)(t)
    return t

def cudnnlstm_block(size, units, inp):
    t = Reshape((size))(inp)
    t = CuDNNLSTM(units, return_sequences=True)(t)
    t = CuDNNLSTM(units, return_sequences=False)(t)
    return t

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


def custom_loss(layer, lay2, T):
    def loss(y_true,y_pred):
        from keras.losses import categorical_crossentropy as logloss
        from keras.losses import kullback_leibler_divergence as kld
        ce = logloss(y_true, y_pred)
        if T>= 1:
            y_pred_soft = backend.softmax(lay2/T)
            layer_soft = backend.softmax(layer/T)
            kl = kld(layer_soft, y_pred_soft)
            return ce + (T**2)*kl
        elif T==0:
            return ce
        else:
            return ce*0
    return loss


class model(object):
    def __init__(self, num_classes=8, model_type=(0,0), wd=0.01, lr=1e-3, num_frame=32, feature_size=(51,), independent=0):
        self.model = None
        self.model_type = model_type
        self.num_frame = num_frame

        self.create_model(num_frame, lr, wd, num_classes, feature_size, independent)

    def create_model(self, num_frame, lr, wd, num_classes, feature_size, independent):
        main_input = Input(shape=(num_frame, feature_size[0]))
        vals = [8, 16, 32, 64]
        def precustom(layer):
            layer = BatchNormalization(axis=-1)(layer)
            layer = Activation(activation_custom)(layer)
            layer = GlobalAveragePooling1D()(layer)
            return layer

        skip = 6
        
        if independent == 0:
            t = Reshape((num_frame, feature_size[0]))(main_input)
            _, _, _, t = TCN_Block(t, activation_custom, vals, jump=True, length=skip)
            t = precustom(t)
            t = MLP(256, t)
        elif independent == 1: 
            acc_x, acc_y, acc_z, ecg, eda, emg, resp, temp = Lambda(lambda x: tf.split(x,num_or_size_splits=8,axis=-1))(main_input)        
            acc = concatenate([acc_x, acc_y, acc_z],)
            _, _, _, acc = TCN_Block(acc, activation_custom, vals, jump=True, length=skip)
            _, _, _, ecg = TCN_Block(ecg, activation_custom, vals, jump=True, length=skip)
            _, _, _, eda = TCN_Block(eda, activation_custom, vals, jump=True, length=skip)
            _, _, _, emg = TCN_Block(emg, activation_custom, vals, jump=True, length=skip)
            _, _, _, resp = TCN_Block(resp, activation_custom, vals, jump=True, length=skip)
            _, _, _, temp = TCN_Block(temp, activation_custom, vals, jump=True, length=skip)

            acc = precustom(acc)
            ecg = precustom(ecg)
            eda = precustom(eda)
            emg = precustom(emg)
            resp = precustom(resp)
            temp = precustom(temp)
            t = concatenate([acc, ecg, eda, emg, resp, temp])
            t = MLP(1024, t)
        elif independent == 2:
            wacc_x, wacc_y, wacc_z, wbvp, weda, wtemp  = Lambda(lambda x: tf.split(x,num_or_size_splits=6,axis=-1))(main_input)
            wacc = concatenate([wacc_x, wacc_y, wacc_z],)
            _, _, _, wacc = TCN_Block(wacc, activation_custom, vals, jump=True, length=skip)
            _, _, _, wbvp = TCN_Block(wbvp, activation_custom, vals, jump=True, length=skip)
            _, _, _, weda = TCN_Block(weda, activation_custom, vals, jump=True, length=skip)
            _, _, _, wtemp = TCN_Block(wtemp, activation_custom, vals, jump=True, length=skip)

            wacc = precustom(wacc)
            wbvp = precustom(wbvp)
            weda = precustom(weda)
            wtemp = precustom(wtemp)

            t = concatenate([wacc, wbvp, weda, wtemp])
            t = MLP(1024, t)
        else:
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

        act = 'softmax'
        l = 'categorical_crossentropy'

        t = Dense(num_classes, kernel_regularizer=L1L2(l1, l2))(t)
        t = Activation(act, name='t1')(t)

        self.model = Model(inputs=main_input, output=[t])
        # self.model.summary()
        losses = l

        self.model.compile(loss=losses, optimizer='adam', metrics=['acc'])

    def train(self, train_data, train_label, cv_index, filepath, mode, step, val_data=[], val_label=[], bs=1, base_epoch=10, epoch_mult=1, num_cycle=5):

        no_validation = False
        if len(np.unique(val_label)) is not len(np.unique(train_label)):
            no_validation = True

        cw_flag = True
        if cw_flag==True:
            cw = {i: 1/((train_label.count(i)/len(train_label)))
                  for i in range(len(np.unique(train_label)))}
        else:
            cw = {i: 1 for i in range(len(np.unique(train_label)))}

        train_label = to_categorical(train_label, num_classes=None)
        val_label = to_categorical(val_label, num_classes=None)

        checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=0, save_best_only=True, save_weights_only=True,
                                     mode='max')

        checkpoint2 = ModelCheckpoint(filepath, monitor='val_out_acc',
                                     verbose=0, save_best_only=True, save_weights_only=True,
                                     mode='max')

        ca = SnapshotEnsemble(base_epoch, len(train_label), bs, epoch_mult)
        callbacks_list = [checkpoint1, checkpoint2]

        if self.model_type[1] == 1:
            train_label = [train_label, train_label, train_label]

            val_label = [val_label,val_label,val_label]

        if no_validation == True:
            self.model.fit(train_data, train_label, batch_size=bs,
                       epochs=base_epoch*num_cycle, shuffle=True,
                       verbose=2, class_weight=cw,
                       callbacks=callbacks_list,
                       )
            self.model.save(filepath)
        else:
            self.model.fit(train_data, train_label, batch_size=bs,
                           epochs=base_epoch*num_cycle, shuffle=True,
                           validation_data=(val_data, val_label),
                           verbose=2, class_weight=cw,
                           callbacks=callbacks_list,
                           )

        return filepath

    def load(self, path):
        self.model.load_weights(path)

    def predict(self, data):
        label = self.model.predict(data)
        return label