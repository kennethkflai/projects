import cv2
from glob import glob
import bz2
import numpy as np
import os
from keras_vggface import VGGFace

path = r'database\FERET Colour Facial Image Database\*\data\ground_truths\name_value\*'
list_path = glob(path)
sub_dict = []
for p in list_path:
    subject = p[p.rfind('\\')+1:]
    sub_dict.append(subject)

sub_dict = {sub_dict[i]: i for i in range(len(sub_dict))}

path = r'data\*\*f*.png'

ss = glob(path)

train_data=[]
train_label=[]
test_data=[]
test_label=[]
test_name=[]
flag = True
for im_index in range(0, len(ss)):
    path = ss[im_index]
    subject = path[path.find('\\')+1:path.rfind('\\')]
    img = np.float64(cv2.imread(path)/255.0)

    if flag:
        train_data.append(img)
        train_label.append(sub_dict[subject])
        flag = 1-flag
    else:
        test_data.append(img)
        test_label.append(sub_dict[subject])
        test_name.append(path[path.rfind('\\')+1:-4])
        flag = 1-flag

from keras.utils.np_utils import to_categorical
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import adam, SGD
num_classes = len(sub_dict)
train_data = np.array(train_data)
test_data = np.array(test_data)
train_label = to_categorical(train_label, num_classes=None)
test_label = to_categorical(test_label, num_classes=None)

vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='classifier')(x)
model = Model(vgg_model.input, out)


model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

filepath="best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='max')
early=EarlyStopping(monitor='val_acc', patience=10,verbose=0,mode='auto')
callbacks_list=[]

model.fit(train_data, train_label,batch_size=8, verbose=1,
          validation_data=(test_data, test_label),
          callbacks=callbacks_list, epochs=1000)
model.load_weights("-" + filepath)
pred=model.predict(test_data)
s='predictions.txt'
f=open(s,'w')
pred_store=[]
for p in range(0,len(pred)):
    cp = np.argsort(pred[p])
    dp = np.sort(pred[p])
    for k in range(1, 6):
        f.write("%d : %f , " % (cp[-k], dp[-k]))
    f.write("%d\n" % (np.argmax(test_label[p])))

    pred_store.append([test_name[p], cp[-1], dp[-1], cp[-2], dp[-2], cp[-3], dp[-3], cp[-4], dp[-4],cp[-5], dp[-5], np.argmax(test_label[p])])

f.close()

