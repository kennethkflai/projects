# -*- coding: utf-8 -*-

from keras.applications.resnet50 import ResNet50
from glob import glob
from keras.layers import Input, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Lambda
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib.cm as cm
from tqdm import tqdm
from keras.utils.np_utils import to_categorical

import os

#import time
#
#start = time.time()
#for i in range(10000):
#    gaussian = np.random.normal(0,255,(256,256,3))
#end = time.time()
#print(end-start)
#
#start = time.time()
#gaussian = np.random.normal(0,255,(10000,256,256,3))
#end = time.time()
#print(end-start)

def load_data(path, csv_path):
    img_path = glob(path)
    meta = pd.read_csv(csv_path + 'meta_subject.csv')

    array = [[] for i in range(80)]

    counter = 0
    for i in tqdm(img_path):

        subject = i[i.rfind('\\')+1:i.rfind('_')]
        mean = 0
        var =100
        sigma = var ** 0.5

        img = cv2.imread(i)
#        img = cv2.imread(r'E:\ken\OneDrive - University of Calgary\SIAMESE_NET_Images\unmasked_visual\1_0.png')
        if counter == 0:
            gaussian = np.random.normal(mean, sigma, (len(img_path),img.shape[0], img.shape[1], img.shape[2]))
#        cv2.imshow('',img)
        img = img  + gaussian[counter]
        img = img.astype(np.uint8)
        img = img.clip(min=0,max=255)


#        cv2.imshow('2',img)
#
#        cv2.waitKey(0)

        array[np.int(subject)-1].append(img)


        counter+= 1


    train = []
    test = []
    train_label = []
    test_label = []

    for i in tqdm(range(len(array))):
        subject_data = array[i].copy()
        np.random.shuffle(subject_data)

        length = np.int(0.1*np.floor(len(subject_data)))

        train += list(subject_data[:length])

        age = meta[meta['Sub_ID']==i+1]['Age'][i]
        gender = meta[meta['Sub_ID']==i+1]['Gender'][i]
        ethnicity = meta[meta['Sub_ID']==i+1]['Ethnicity'][i]

        train_label += [(i, age, gender, ethnicity) for j in range(length)]
        test += list(subject_data[length:])
        test_label += [(i, age, gender, ethnicity) for j in range(len(subject_data)-length)]

    return np.array(train), np.array(test), np.array(train_label), np.array(test_label)

def model_predict(train_data, test_data, train_label, test_label,name, split=10):
    inp = Input(shape=(256,256,3))

    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inp)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)


#    model = ResNet50(include_top=False, input_tensor=inp, weights='imagenet')

#    x = GlobalAveragePooling2D()(model.layers[-1].output)
    x = GlobalAveragePooling2D()(x)
    out = Dense(len(np.unique(train_label[:,0])), activation='softmax')(x)

#    base_model = Model(inputs=model.inputs, outputs=x)
    base_model = Model(inputs=inp, outputs=x)
#    train_model = Model(inputs=model.inputs, outputs=out)
    train_model = Model(inputs=inp, outputs=out)

    train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint1 = ModelCheckpoint(name + '.hdf5', monitor='val_acc',
                                 verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='max')

    callbacks_list = [checkpoint1]

    label = to_categorical(train_label[:,0], num_classes=None)
    test_label = to_categorical(test_label[:,0], num_classes=None)
    train_model.fit(train_data, label,
                    batch_size=32,epochs=10, validation_data=(test_data, test_label),
                    verbose=1, shuffle=True,callbacks=callbacks_list)

    num_images = len(test_data)

    increments = np.int(np.floor(num_images/split))
    features = []
    predictions = []
    for i in range(split):
        if i == split-1:
            images = test_data[increments*i:]
        else:
            images = test_data[increments*i:increments*(i+1)]
        f = base_model.predict(np.array(images), verbose=1)
        p = train_model.predict(np.array(images), verbose=1)
        if i ==0:
            features = f
            predictions = p
        else:
            features = np.vstack((features, f))
            predictions = np.vstack((predictions, p))

    return features, predictions

def process(path, csv_path, name):

    if os.path.exists(name + "_train_data.npy") == False:
        train, test, train_label, test_label = load_data(path = path, csv_path = csv_path)
        np.save(name + "_train_data", train)
        np.save(name + "_test_data", test)
        np.save(name + "_train_labels", train_label)
        np.save(name + "_test_labels", test_label)
    else:
        train = np.load(name + "_train_data", allow_pickle=True)
        test = np.load(name + "_test_data", allow_pickle=True)
        train_label = np.load(name + "_train_labels", allow_pickle=True)
        test_label = np.load(name + "_test_labels", allow_pickle=True)

    features, predictions = model_predict(train, test, train_label, test_label, name)

    np.save(name + "_features", features)
    np.save(name + "_predictions",predictions)

    tsne = TSNE(n_components=2, verbose=1).fit_transform(features)
    tx = tsne[:,0]
    ty = tsne[:,1]
    np.save(name + "_tsne", tsne)

    plot_graph(tx,ty, test_label[:,1], name + '_age.pdf')
    plot_graph(tx,ty, test_label[:,2], name + '_gender.pdf')
    plot_graph(tx,ty, test_label[:,3], name + '_ethnicity.pdf')

def plot_graph(tx,ty,labels,name):
    fig = plt.figure(figsize=(25, 25))
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    ax = fig.add_subplot(111)

    x = np.arange(len(np.unique(labels)))
    ys = [i+x+(i*x)**2 for i in range(len(np.unique(labels)))]

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    colors_per_class = np.unique(labels)
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        index = np.where(colors_per_class==label)[0][0]

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, color=colors[index], label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(name) as pdf:
        pdf.savefig(fig,bbox_inches='tight')

    # finally, show the plot
#    plt.show()

csv_path = r'E:\ken\database\speaking\\'

path = r'E:\ken\OneDrive - University of Calgary\SIAMESE_NET_Images\unmasked_visual\*'
process(path, csv_path, "speak\\visual_normal")

path = r'E:\ken\OneDrive - University of Calgary\SIAMESE_NET_Images\masked_visual\*'
process(path, csv_path, "speak\\visual_mask")

path = r'E:\ken\OneDrive - University of Calgary\SIAMESE_NET_Images\unmasked_thermal\*'
process(path, csv_path, "speak\\thermal_normal")

path = r'E:\ken\OneDrive - University of Calgary\SIAMESE_NET_Images\masked_thermal\*'
process(path, csv_path, "speak\\thermal_mask")