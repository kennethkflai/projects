# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 02:18:24 2022

@author: BT_Lab
"""

import numpy as np
import fairlearn
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from fairlearn.metrics import equalized_odds_difference,equalized_odds_ratio
from sklearn.metrics import classification_report
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import metrics
from metrics import demographic_parity_difference_personal, equalized_odds_difference_personal

def draw_cm(trueP, modelP, name, cmap=plt.cm.Purples):
    import itertools

    matrix = confusion_matrix(trueP, modelP)
    np.set_printoptions(precision=8)

    raw = matrix
    matrix = matrix.astype('float') / matrix.sum(axis=1)[: , np.newaxis] * 100

    classes = [i for i in range(len(np.unique(trueP)))]
    fig = plt.figure(figsize=(100, 100))

    fig.tight_layout()
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, fontsize=25)
    plt.yticks(tick_marks, classes, fontsize=25)
    thresh = matrix.max()/2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        if matrix[i, j] > thresh:
            clr = "white"
        else:
            clr = "black"

        if len(np.unique(trueP))>2:
            txt = f'{matrix[i,j]:2.2f}'
        else:
            txt = f'{raw[i,j]} ({matrix[i,j]:2.2f}%)'
        plt.text(j, i, txt,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color=clr, fontsize=15)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(name) as pdf:
        pdf.savefig(fig,bbox_inches='tight')

def difference(label_path, prediction_path):
    labels = np.load(label_path, allow_pickle=True)
    age_labels = labels[:,1]
    temp_list = []
    for index in range(len(age_labels)):
        val = int(age_labels[index])
        if val < 25:
            temp = 0
        elif val < 30:
            temp = 1
        elif val < 35:
            temp = 2
        else:
            temp = 3
        temp_list.append(temp)
    labels[:,1] = temp_list

    predictions = np.load(prediction_path, allow_pickle=True)
    if len(predictions.shape) >1:
        predictions = np.argmax(predictions,1)
    ground_truth = np.array([int(labels[i,0]) for i in range(len(labels))])

    performance = classification_report(ground_truth,predictions,output_dict=True)

    meta = pd.read_csv(r'E:\ken\database\speaking\\' + 'meta_subject.csv')

    l = []
    df=[]
    for i in range(80):
        info = performance[f'{i}']
        info.update({'subject':i+1})
        age = meta[meta['Sub_ID']==(i+1)]['Age'][i]
        gender = meta[meta['Sub_ID']==(i+1)]['Gender'][i]
        ethnicity = meta[meta['Sub_ID']==(i+1)]['Ethnicity'][i]

        info.update({'age':age})
        info.update({'gender':gender})
        info.update({'ethnicity':ethnicity})
        l.append(info)

    df = pd.DataFrame(l)
    df_edit = df.copy()

    support = df.iloc[:,3].sum(0)
    mean = df.iloc[:,:3].mean()
    std = df.iloc[:,:3].std(ddof=0)
    temp = df.iloc[:,:3].sum(0)/len(df)
    temp = temp.append(pd.Series({'support':support}))
    temp = temp.append(pd.Series({'subject':'baseline'}))
    df_edit = df_edit.append(temp, ignore_index=True)

    for age in df.iloc[:,5].unique():
        temp = df[df['age']==age]
        support = temp.iloc[:,3].sum(0)
        temp = temp.iloc[:,:3].sum(0)/len(temp)
        temp = temp.append(pd.Series({'support':support}))
        temp = temp.append(pd.Series({'subject':f'age:{age}'}))
        df_edit = df_edit.append(temp, ignore_index=True)

    for gender in df.iloc[:,6].unique():
        temp = df[df['gender']==gender]
        support = temp.iloc[:,3].sum(0)
        temp = temp.iloc[:,:3].sum(0)/len(temp)
        temp = temp.append(pd.Series({'support':support}))
        temp = temp.append(pd.Series({'subject':f'gender:{gender}'}))
        df_edit = df_edit.append(temp, ignore_index=True)\

    for ethnicity in df.iloc[:,7].unique():
        temp = df[df['ethnicity']==ethnicity]
        support = temp.iloc[:,3].sum(0)
        temp = temp.iloc[:,:3].sum(0)/len(temp)
        temp = temp.append(pd.Series({'support':support}))
        temp = temp.append(pd.Series({'subject':f'ethnicity:{ethnicity}'}))
        df_edit = df_edit.append(temp, ignore_index=True)

    df = df_edit
    # draw_cm(ground_truth,predictions , 'speak2\\cm.pdf')
    dpd = [[] for i in range(3)]
    eod = [[] for i in range(3)]

    for i in range(len(np.unique(ground_truth))):

        binary_truth = ground_truth==i
        binary_prediction = predictions==i

        for cohort in range(1,4):
            print(i, cohort)
            # dpd[cohort-1].append(metrics.demographic_parity_difference_personal(binary_truth,binary_prediction,sensitive_features=labels[:,cohort]))
            # eod[cohort-1].append(metrics.equalized_odds_difference_personal(binary_truth,binary_prediction,sensitive_features=labels[:,cohort]))

            dpd[cohort-1].append(metrics.demographic_parity_difference_personal(binary_truth,binary_prediction,group_label=labels[:,cohort]))
            eod[cohort-1].append(metrics.equalized_odds_difference_personal(binary_truth,binary_prediction,group_label=labels[:,cohort]))


    performance = np.vstack((mean,std,np.mean(dpd,1),np.std(dpd,1),np.mean(eod,1),np.std(eod,1)))
    # p1 = demographic_parity_difference(ground_truth, predictions,sensitive_features=labels[:,1])
    # p2 = demographic_parity_difference(ground_truth, predictions,sensitive_features=labels[:,2])
    # p3 = demographic_parity_difference(ground_truth, predictions,sensitive_features=labels[:,3])
    # performance = np.vstack((mean,std,(p1,p2,p3)))
    return df, performance, dpd, eod

def plot_graph(tx,ty,labels,name):
    fig = plt.figure(figsize=(10, 10))
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rc('legend', fontsize=15)
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
        ax.scatter(current_tx[::10], current_ty[::10], color=colors[index], label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(name) as pdf:
        pdf.savefig(fig,bbox_inches='tight')

    # finally, show the plot
    plt.show()

def plotting(label_path, tsne_path, name):
    test_label = np.load(label_path, allow_pickle=True)
    tsne = np.load(tsne_path, allow_pickle=True)
    tx = tsne[:,0]
    ty = tsne[:,1]
    plot_graph(tx,ty, test_label[:,1], name + '_age.pdf')
    plot_graph(tx,ty, test_label[:,2], name + '_gender.pdf')
    plot_graph(tx,ty, test_label[:,3], name + '_ethnicity.pdf')


# name = "speak2\\visual_normal"
# label_path = r'E:\ken\synthetic_fairness\speak_80_20\visual_normal_labels.npy'
# tsne_path = r'E:\ken\synthetic_fairness\speak_80_20\visual_normal_tsne.npy'
# plotting(label_path,tsne_path, name)

# name = "speak2\\visual_mask"
# label_path = r'E:\ken\synthetic_fairness\speak_80_20\visual_mask_labels.npy'
# tsne_path = r'E:\ken\synthetic_fairness\speak_80_20\visual_mask_tsne.npy'
# plotting(label_path,tsne_path, name)

# name = "speak2\\thermal_normal"
# label_path = r'E:\ken\synthetic_fairness\speak_80_20\thermal_normal_labels.npy'
# tsne_path = r'E:\ken\synthetic_fairness\speak_80_20\thermal_normal_tsne.npy'
# plotting(label_path,tsne_path, name)

# name = "speak2\\thermal_mask"
# label_path = r'E:\ken\synthetic_fairness\speak_80_20\thermal_mask_labels.npy'
# tsne_path = r'E:\ken\synthetic_fairness\speak_80_20\thermal_mask_tsne.npy'
# plotting(label_path,tsne_path, name)

# fnr, fpr
#cmc
#differential performance, differential outcome, false negative differential, and false positive differential
performance = [[] for i in range(4)]
df = [[] for i in range(4)]

sp = 'speak_10_90_nonoise_cnn2'
test = '_test'
# test = ''
label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_mask'+ test + '_labels.npy'
prediction_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_mask_predictions.npy'
df[0], performance[0],_,_ = difference(label_path, prediction_path)

label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_normal'+ test + '_labels.npy'
prediction_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_normal_predictions.npy'
df[1], performance[1],_,_ = difference(label_path, prediction_path)

label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\thermal_mask'+ test + '_labels.npy'
prediction_path = r'E:\ken\synthetic_fairness\\' + sp + '\\thermal_mask_predictions.npy'
df[2], performance[2],_,_ = difference(label_path, prediction_path)

label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\thermal_normal'+ test + '_labels.npy'
prediction_path = r'E:\ken\synthetic_fairness\\' + sp + '\\thermal_normal_predictions.npy'
df[3], performance[3],_,_ = difference(label_path, prediction_path)

#prediction = np.load(prediction_path)
#val = 100
#performance_ = [[] for i in range(val)]
#df_ = [[] for i in range(val)]
#
#for i in range(val):
#    path = f'rand_{i}.npy'
#    if os.path.exists(path) == False:
#        random_predictions = np.random.randint(0,80,len(prediction))
#        np.save(f"rand_{i}",random_predictions)
#
#    label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_mask'+ test + '_labels.npy'
#    df_[i], performance_[i], dpd, eod = difference(label_path, path)
#
#p = np.array(performance_)
#p_mean = np.mean(p,0)
#p_std = np.std(p,0)
