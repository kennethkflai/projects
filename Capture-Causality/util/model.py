from keras.layers import Dense, Activation
from keras.layers import add, concatenate, BatchNormalization
from keras.layers import Reshape, Input, Dropout
from keras.layers import average, GlobalAveragePooling1D, AveragePooling1D
from keras.models import Model
from keras.layers import Conv1D, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback
from util.data_process import num_frame, num_point, num_channel
from util.opt2 import AdamW2

import numpy as np
from keras.regularizers import L1L2
from keras import backend
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

class Mish(Activation):

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(x):
    return x*backend.tanh(backend.softplus(x))

get_custom_objects().update({'Mish': Mish(mish)})
model_list = {0: "test", 1: "test1", 2: "test2", 3: "test3", 4: "test4"}

alr = True
l1=0.#1e-4
l2=0.

activation_custom = 'relu'
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

def custom_loss(layer, layer2, T):
    def loss(y_true,y_pred):
        from keras.losses import categorical_crossentropy as logloss
        from keras.losses import kullback_leibler_divergence as kld
        ce = logloss(y_true, y_pred)
        if T>= 1:
            y_pred_soft = backend.softmax(layer2/T)
            layer_soft = backend.softmax(layer/T)
            kl = kld(layer_soft, y_pred_soft)
            return ce + (T**2)*kl
        elif T==0:
            return ce
        else:
            return ce*0
    return loss


def custom_loss_total(lay,lay1,lay2,lay3,lay4, T):
    def loss(y_true,y_pred):
        from keras.losses import categorical_crossentropy as logloss
        from keras.losses import kullback_leibler_divergence as kld
        ce = logloss(y_true, y_pred)
        if T>= 1:
            layer_soft = backend.softmax(lay/T)

            y_pred_soft = backend.softmax(lay1/T)
            kl_1 = kld(layer_soft, y_pred_soft)

            y_pred_soft = backend.softmax(lay2/T)
            kl_2 = kld(layer_soft, y_pred_soft)

            y_pred_soft = backend.softmax(lay3/T)
            kl_3 = kld(layer_soft, y_pred_soft)

            y_pred_soft = backend.softmax(lay4/T)
            kl_4 = kld(layer_soft, y_pred_soft)

            return ce + (T**2)*(kl_1+kl_2+kl_3+kl_4)
        elif T==0:
            return ce
        else:
            return ce*0
    return loss

def TCN_Block(inp, activation_custom, vals, jump=True, length=8):
    t = Conv1D(vals[0], length, padding='same')(inp)

    def sub_block(activation_custom, fc_units, stride, inp, length):
        t1 = Conv1D(fc_units, 1, strides=stride, padding='same', kernel_regularizer=L1L2(l1, l2))(inp)
        t = BatchNormalization(axis=-1)(inp)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(stride), padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same', kernel_regularizer=L1L2(l1, l2))(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same', kernel_regularizer=L1L2(l1, l2))(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])
        return t

    tout1 = sub_block(activation_custom, vals[0],1,t, length)
    tout2 = sub_block(activation_custom, vals[1],jump+1,tout1, length)
    tout3 = sub_block(activation_custom, vals[2],jump+1,tout2, length)
    tout4 = sub_block(activation_custom, vals[3],jump+1,tout3, length)

    return tout1, tout2, tout3, tout4

def model_set(value, channel):
    global num_frame
    num_frame = value

    global num_channel
    num_channel = channel

class model(object):
    def __init__(self, num_classes=14, model_type=0, wd=0.01, lr=1e-3, binary=True):
        self.model = None

        self.create_test_model(lr, wd, num_classes, binary=binary)

        self.model_type = model_type

    def create_test_model(self, lr, wd, num_classes, binary):
        main_input = Input(shape=(num_frame, 4, 6))
        t = Reshape((num_frame, 6* 4))(main_input)
        vals = [32, 64, 128, 256]
        t1, t2, t3, t4 = TCN_Block(t, activation_custom, vals, jump=True, length=6)

        t1 = BatchNormalization(axis=-1)(t1)
        t1 = Activation(activation_custom)(t1)
        t1 = GlobalAveragePooling1D()(t1)

        t2 = BatchNormalization(axis=-1)(t2)
        t2 = Activation(activation_custom)(t2)
        t2 = GlobalAveragePooling1D()(t2)

        t3 = BatchNormalization(axis=-1)(t3)
        t3 = Activation(activation_custom)(t3)
        t3 = GlobalAveragePooling1D()(t3)

        t4 = BatchNormalization(axis=-1)(t4)
        t4 = Activation(activation_custom)(t4)
        t4 = GlobalAveragePooling1D()(t4)
        if binary==True:
            num_classes = 1
            act = 'sigmoid'
            l = 'binary_crossentropy'
        else:
            act = 'softmax'
            l = 'categorical_crossentropy'

        t1 = Dense(num_classes, kernel_regularizer=L1L2(l1, l2))(t1)
        tout1 = Activation(act, name='t1')(t1)

        t2 = Dense(num_classes, kernel_regularizer=L1L2(l1, l2))(t2)
        tout2 = Activation(act, name='t2')(t2)

        t3 = Dense(num_classes, kernel_regularizer=L1L2(l1, l2))(t3)
        tout3 = Activation(act, name='t3')(t3)

        t4 = Dense(num_classes, kernel_regularizer=L1L2(l1, l2))(t4)
        tout4 = Activation(act, name='t4')(t4)

        logit = concatenate([t1, t2, t3, t4])
        logit = Dense(num_classes, kernel_regularizer=L1L2(l1, l2))(logit)
        tout = Activation(act, name='out')(logit)

        opti = AdamW2(learning_rate=lr, weight_decay=wd, anneal_lr=alr)
        self.model = Model(inputs=main_input, output=[tout])

        losses = {"out": custom_loss_total(logit,t1,t2,t3,t4,3)}
        self.model.compile(loss=l, optimizer=opti, metrics=['accuracy'])

    def train(self, train_data, train_label, cv_index, val_data=[], val_label=[], bs=1, base_epoch=10, epoch_mult=1.5, num_cycle=4, binary=True):
        if val_data != []:
            if binary==False:
                train_label = to_categorical(train_label, num_classes=None)
                val_label = to_categorical(val_label, num_classes=None)

            ca = SnapshotEnsemble(base_epoch, len(train_label), bs, epoch_mult)

            filepath = model_list[self.model_type] + '-models//cv' + str(cv_index) + '.hdf5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',
                                         verbose=0, save_best_only=True, save_weights_only=True,
                                         mode='max')
            checkpoint2 = ModelCheckpoint(filepath, monitor='val_acc',
                                         verbose=0, save_best_only=True, save_weights_only=True,
                                         mode='max')
            callbacks_list = [checkpoint, checkpoint2, ca]

            self.model.fit(train_data, train_label, batch_size=bs,
                           epochs=base_epoch*num_cycle, shuffle=True,
                           validation_data=(val_data, val_label),
                           verbose=2,
                           callbacks=callbacks_list,
                           )

    def load(self, path):
        self.model.load_weights(path)

    def predict(self, data):
        if self.model_type == 0:
            test_data = data
        else:
            test_data = data

        label = self.model.predict(test_data)
        return label