# -*- coding: utf-8 -*-

from keras.applications.resnet50 import ResNet50
from glob import glob
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Lambda
from keras.models import Sequential, Model, load_model
import cv2
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib.cm as cm
from tqdm import tqdm

def load_data(path,json_path, limit):
    img_path = glob(path)
    age_label = []
    gender_label = []
    array = []
    for i in tqdm(img_path[0:limit]):

        subject = i[i.rfind('\\')+1:-4]
        if subject.rfind('_') >= 0:
            subject = subject[:subject.rfind('_')]

        img = cv2.imread(i)


        j = open(json_path + subject + '.json')
        data = json.load(j)
        if len(data)==0:
            continue;
        age_label.append(data[0]['faceAttributes']['age'])
        gender_label.append(data[0]['faceAttributes']['gender'])

        array.append(img)

    return array, age_label, gender_label

def model_predict(array,split=10):
    inp = Input(shape=(256,256,3))

    model = ResNet50(include_top=False, input_tensor=inp, weights='imagenet')

    x = GlobalAveragePooling2D()(model.layers[-1].output)
    out = Dense()
    base_model = Model(inputs=model.inputs, outputs=x)

    num_images = len(array)

    increments = np.int(np.floor(num_images/split))
    features = []
    for i in range(split):
        if i == split-1:
            images = array[increments*i:]
        else:
            images = array[increments*i:increments*(i+1)]
        pred = base_model.predict(np.array(images), verbose=1)
        if i ==0:
            features = pred
        else:
            features = np.vstack((features, pred))

    return features


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
    plt.show()

p = r'E:\ken\database\FFHQ\256\*'
array, age_label, gender_label = load_data(path = p,
                                           json_path = r'E:\ken\database\FFHQ\ffhq-features-dataset-master\json\\',
                                           limit=-1)
result = model_predict(array)

tsne_normal = TSNE(n_components=2, verbose=1).fit_transform(result[0:])
tx = tsne_normal[:,0]
ty = tsne_normal[:,1]
np.save("tsne_normal", tsne_normal)

plot_graph(tx,ty, age_label[0:], 'age_normal.pdf')
plot_graph(tx,ty, gender_label[0:], 'gender_normal.pdf')

p = r'E:\ken\database\MaskNet\256\*'
array, age_label, gender_label = load_data(path = p,
                                           json_path = r'E:\ken\database\FFHQ\ffhq-features-dataset-master\json\\',
                                           limit=-1)
result = model_predict(array)

tsne_mask = TSNE(n_components=2, verbose=1).fit_transform(result[0:])
tx = tsne_mask[:,0]
ty = tsne_mask[:,1]
np.save("tsne_mask", tsne_mask)
plot_graph(tx,ty, age_label[0:], 'age_mask.pdf')
plot_graph(tx,ty, gender_label[0:], 'gender_mask.pdf')