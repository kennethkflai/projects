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

from util.opt2 import AdamW2
l1=0#1e-4
l2=0#0.001
activation_custom = 'relu'

class SnapshotEnsemble(Callback):
    # constructor
    def __init__(self, n_cycles, n_samples, batch_size=512, cycle_mult_factor=1.5, verbose=0):
        self.steps_per_epoch = int(np.floor(n_samples/batch_size))

        self.cycle_length = n_cycles
        self.cycle_mult_factor = cycle_mult_factor
        self.epoch_counter = 0

    def on_train_begin(self, logs={}):
        #Set the number of training batches of the first restart cycle to steps_per_cycle
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
    def __init__(self, num_classes=8, model_type=(0,0), wd=0.01, lr=1e-3, num_frame=32, feature_size=(17,2)):
        self.model = None

        if model_type[0] == 0:
            self.create_point_model(num_frame, lr, wd, num_classes, feature_size)
        else:
            self.create_image_model(num_frame, lr, wd, num_classes, feature_size)

        self.model_type = model_type


    def create_point_model(self, num_frame, lr, wd, num_classes, feature_size):

        main_input = Input(shape=(num_frame, feature_size[0] , feature_size[1]))
        t = Reshape((num_frame, feature_size[0]* feature_size[1]))(main_input)
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

        self.model = Model(inputs=main_input, output=[tout,tout1,tout2, tout3, tout4])
        self.model.summary()
        losses = {"out": l,
                  "t1": custom_loss(logit, t1, 3),
                  "t2": custom_loss(logit, t2, 3),
                  "t3": custom_loss(logit, t3, 3),
                  "t4": custom_loss(logit, t4, 3)
                  }

        opti = AdamW2(learning_rate=lr, weight_decay=wd, anneal_lr=True)
        self.model.compile(loss=losses, optimizer=opti, metrics=['accuracy'])

    def create_image_model(self, num_frame, lr, wd, num_classes, feature_size):
        print('proto')

        main_input = Input(shape=(num_frame, feature_size))
        #t = Reshape((num_frame, feature_size))(main_input)
        t = TimeDistributed(Dense(524))(main_input)
        t = TimeDistributed(Dense(524))(t)
        vals = [64, 128, 256, 256]
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

        #self.model = Model(inputs=main_input, output=[tout,tout1,tout2, tout3, tout4])
        self.model = Model(inputs=main_input, output=[tout])
        self.model.summary()
        losses = {"out": l,
                  "t1": custom_loss(logit, t1, 10),
                  "t2": custom_loss(logit, t2, 10),
                  "t3": custom_loss(logit, t3, 10),
                  "t4": custom_loss(logit, t4, 10)
                  }

        opti = AdamW2(learning_rate=lr, weight_decay=wd, anneal_lr=True)
        self.model.compile(loss=l, optimizer=opti, metrics=['accuracy'])

    def train(self, train_data, train_label, cv_index, val_data=[], val_label=[], bs=1, base_epoch=10, epoch_mult=1, num_cycle=4):
        model_name = {0:"VGG16",
                      1:"VGG19",
                      2:"ResNet50",
                      3:"InceptionV3",
                      4:"InceptionResNetV2",
                      5:"Xception",
                      6:"NASNetLarge",
                      7:"DenseNet201"}
        train_label = to_categorical(train_label, num_classes=None)
        val_label = to_categorical(val_label, num_classes=None)

        if self.model_type[0]==0:
            filepath = 'save//models//' + str(self.model_type[0]) + '-' + str(self.model_type[1]) +'-cv' + str(cv_index) + '.hdf5'
        else:
            filepath = 'save//models//' + str(self.model_type[0])+ '-' + model_name[self.model_type[1]]  +'-cv' + str(cv_index) + '.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_out_accuracy',
                                     verbose=0, save_best_only=True, save_weights_only=True,
                                     mode='max')
        checkpoint2 = ModelCheckpoint(filepath, monitor='val_out_acc',
                                     verbose=0, save_best_only=True, save_weights_only=True,
                                     mode='max')

        ca = SnapshotEnsemble(base_epoch, len(train_label), bs, epoch_mult)
        callbacks_list = [checkpoint, checkpoint2, ca]

        self.model.fit(train_data, [train_label,train_label,train_label,train_label,train_label], batch_size=bs,
                       epochs=base_epoch*num_cycle, shuffle=True,
                       validation_data=(val_data, [val_label,val_label,val_label,val_label,val_label]),
                       verbose=2,
                       callbacks=callbacks_list,
                       )
                       
        # self.model.fit(train_data, [train_label], batch_size=bs,
                       # epochs=base_epoch*num_cycle, shuffle=True,
                       # validation_data=(val_data, [val_label]),
                       # verbose=2,
                       # callbacks=callbacks_list,
                       # )
        return filepath

    def load(self, path):
        self.model.load_weights(path)

    def predict(self, data):
        label = self.model.predict(data)
        return label