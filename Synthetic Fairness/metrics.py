# -*- coding: utf-8 -*-
"""
"""
#from fairlearn.metrics import demographic_parity_difference
#from fairlearn.metrics import equalized_odds_difference

import numpy as np
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import matplotlib.cm as cm

def values(truth, prediction):
    result = np.logical_and(prediction, truth)

    tp = np.sum(result)
    fp = np.sum(prediction) - tp
    fn = np.sum(truth) - np.sum(result)

    tn = len(result) - tp - fp - fn

    return tp, fp, tn, fn


def positive_rate(prediction,
                  group_label=[]):
    """
    Calculate the fraction of prediction is positive for each class in group_label.
    Also referred to as positive rate: (TP+FP)/(TP+FN+FP+TN)
    ----------
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """
    if group_label == []:
        return np.sum(prediction)/len(prediction)

    pr = {}
    groups = np.unique(group_label)
    for i in range(len(groups)):
        pr.update({groups[i]:i})

    for i in groups:
        pred = prediction[group_label==i]
        pred_list = np.unique(pred)
        if len(pred_list) > 2:
            temp_pr = np.zeros((pred_list))
            for j in range(pred_list):
                temp_class = pred_list[j]
                temp_pred = pred==temp_class
                temp_sum = np.sum(temp_pred)
                if temp_sum == 0:
                    temp_pr[j] = 1
                else:
                    temp_pr[j] = np.sum(temp_pred)/len(temp_pred)
            pr[i] = np.mean(temp_pr)
            # temp = np.unique(pred)[0]
            # temp = pred==temp
            # pr[i] = np.sum(temp)/len(temp)
        else:
            pr[i] = np.sum(pred)/len(pred)

    return pr

def demographic_parity_difference_personal(truth,
                                           prediction,
                                           group_label):
    """
    Calculate the demographic_parity_difference.
    Computes the positive rate difference for the class with highest positive rate with the class with the lowest positive rate
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """

    pr = positive_rate(prediction, group_label)
    return abs(pr[max(pr, key=pr.get)]-pr[min(pr, key=pr.get)])

def true_positive_rate(truth,
                       prediction,
                       group_label=[]):
    """
    Calculate the true_positive_rate: TP/(TP+FN)
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """
    if group_label == []:
        result = np.logical_and(prediction, truth)
        return np.sum(result)/np.sum(truth)

    tpr = {}
    groups = np.unique(group_label)
    for i in range(len(groups)):
        tpr.update({groups[i]:i})

    for i in groups:
        pred = prediction[group_label==i]
        true = truth[group_label == i]
        result = np.logical_and(pred, true)
        pred_list = np.unique(pred)
        
        if len(pred_list) > 2:
            temp_tpr = np.zeros((len(pred_list)))
            for j in range(len(pred_list)):
                temp_class = pred_list[j]
                temp_pred = pred==temp_class
                temp_true = true==temp_class
                temp_result = temp_pred & temp_true

                if np.sum(temp_true) == 0:
                    temp_tpr[j] = 0
                else:
                    temp_tpr[j]= np.sum(temp_result)/np.sum(temp_true)

            tpr[i] = np.mean(temp_tpr)
            # temp = np.unique(pred)[0]
            # temp = pred==temp
            # pr[i] = np.sum(temp)/len(temp)
        else:
            if np.sum(true) == 0:
                tpr[i] = 0
            else:
                tpr[i] = np.sum(result)/np.sum(true)

    return tpr

def false_positive_rate(truth,
                        prediction,
                        group_label=[]):
    """
    Calculate the false_positive_rate: FP/(FP+TN)
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """

    if group_label == []:
        result = np.logical_and(prediction, truth)
        return (np.sum(prediction)-np.sum(result))/(len(truth)-np.sum(truth))

    fpr = {}
    groups = np.unique(group_label)
    for i in range(len(groups)):
        fpr.update({groups[i]:i})

    for i in groups:
        pred = prediction[group_label==i]
        true = truth[group_label == i]
        result = np.logical_and(pred, true)

        pred_list = np.unique(prediction)
        if len(pred_list) > 2:
            temp_fpr = np.zeros((len(pred_list)))

            for j in range(len(pred_list)):
                temp_class = pred_list[j]
                temp_pred = pred==temp_class
                temp_true = true==temp_class
                temp_result = temp_pred & temp_true

                temp_fpr[j] = (np.sum(temp_pred)-np.sum(temp_result))/(len(temp_true)-np.sum(temp_true))

            fpr[i] = np.mean(temp_fpr)
            # temp = np.unique(pred)[0]
            # temp = pred==temp
            # pr[i] = np.sum(temp)/len(temp)
        else:
            fpr[i] = (np.sum(pred)-np.sum(result))/(len(true)-np.sum(true))

    return fpr

def positive_prediction_value(truth,
                        prediction,
                        group_label=[]):
    """
    Calculate the positive_prediction_value: TP/(FP+TP)
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """

    if group_label == []:
        result = np.logical_and(prediction, truth)
        if np.sum(prediction) == 0:
            return 0
        else:
            return np.sum(result)/np.sum(prediction)

    ppv = {}
    for i in range(len(np.unique(group_label))):
        ppv.update({np.unique(group_label)[i]:i})

    for i in np.unique(group_label):
        pred = prediction[group_label==i]
        true = truth[group_label == i]
        result = np.logical_and(pred, true)

        if len(np.unique(pred)) > 2:
            temp_ppv = np.zeros((len(np.unique(pred))))

            for j in range(len(np.unique(pred))):
                temp_class = np.unique(pred)[j]
                temp_pred = pred==temp_class
                temp_true = true==temp_class
                temp_result = temp_pred & temp_true

                if np.sum(pred) == 0:
                    temp_ppv[j] = 0
                else:
                    temp_ppv[j] = (np.sum(temp_result))/(np.sum(pred))

            ppv[i] = np.mean(temp_ppv)
        else:
            if np.sum(pred) == 0:
                ppv[i] = 0
            else:
                ppv[i] = (np.sum(result))/(np.sum(pred))

    return ppv

def negative_prediction_value(truth,
                        prediction,
                        group_label=[]):
    """
    Calculate the negative_prediction_value: TN/(FN+TN)
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """

    if group_label == []:
        result = np.logical_and(prediction==0, truth==0)
        return np.sum(result)/(len(prediction)-np.sum(prediction))


    npv = {}
    for i in range(len(np.unique(group_label))):
        npv.update({np.unique(group_label)[i]:i})

    for i in np.unique(group_label):
        pred = prediction[group_label==i]
        true = truth[group_label == i]
        result = np.logical_and(pred==0, true==0)

        if len(np.unique(pred)) > 2:
            temp_npv = np.zeros((len(np.unique(pred))))

            for j in range(len(np.unique(pred))):
                temp_class = np.unique(pred)[j]
                temp_pred = pred==temp_class
                temp_true = true==temp_class
                temp_result = np.logical_and(temp_pred==0, temp_true==0)

                temp_npv[j] = (np.sum(temp_result))/(len(pred)-np.sum(pred))

            npv[i] = np.mean(temp_npv)
        else:

            npv[i] = (np.sum(result))/(len(pred)-np.sum(pred))

    return npv

def positive_prediction_value_difference(truth,
                                  prediction,
                                  group_label):
    """
    Calculate the positive_prediction_value_difference or predictive parity.
    Computes the ppv difference between the class with highest ppv and the class with lowest ppv
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """
    ppv = positive_prediction_value(truth, prediction, group_label)
    return abs(ppv[max(ppv, key=ppv.get)]-ppv[min(ppv, key=ppv.get)])

def true_positive_rate_difference(truth,
                                  prediction,
                                  group_label):
    """
    Calculate the true_positive_rate_difference.
    Computes the tpr difference between the class with highest tpr and the class with lowest tpr
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """
    tpr = true_positive_rate(truth, prediction, group_label)
    return abs(tpr[max(tpr, key=tpr.get)]-tpr[min(tpr, key=tpr.get)])

def false_positive_rate_difference(truth,
                                  prediction,
                                  group_label):
    """
    Calculate the false_positive_rate_difference.
    Computes the fpr difference between the class with highest fpr and the class with lowest fpr
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """
    fpr = false_positive_rate(truth, prediction, group_label)
    return abs(fpr[max(fpr, key=fpr.get)]-fpr[min(fpr, key=fpr.get)])

def equalized_odds_difference_personal(truth,
                                           prediction,
                                           group_label):
    """
    Calculate the equalized_odds_difference.
    Takes the maximum value of two: true positive rate difference and false positive rate difference
    ----------
    truth : array_like
        The ground truth labels
    prediction : array_like
        The predicted labels
    group_label : array_like
        The labels to sensitive group (e.g. gender, age, etc)
    """

    tprd = true_positive_rate_difference(truth, prediction, group_label)
    fprd = false_positive_rate_difference(truth, prediction, group_label)

    return max(tprd,fprd)

def calculate_curve(df):

    curve = np.zeros((80))

    truth = df.loc[:,"Truth"].astype(int)
    for rank in range(1,len(curve)+1):
        tp = 0
        for n in range(rank):
            rank_prediction = df.loc[:,n]

            result = rank_prediction==truth
            tp += np.sum(result)
        curve[rank-1] = 1-tp/len(truth)

    return curve

def plot_graph(data, label, name):
    fig = plt.figure(figsize=(8, 8))

    plt.grid(True, which="both")
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rc('legend', fontsize=15)
    ax = fig.add_subplot(111)
    ax.ticklabel_format(useOffset=False)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)

    # colors = cm.tab20b(np.linspace(0, 1, len(label)))
    colors = ['red','blue','green','purple']
    # colors = [cm.Reds(np.linspace(0, 1, 10))[9],
    #           cm.Blues(np.linspace(0, 1, 10))[9],
    #           cm.Greens(np.linspace(0, 1, 10))[9],
    #           cm.Purples(np.linspace(0, 1, 10))[9]]
    marker = ['','','','']
    linestyle = ['solid','solid','dashed','dashed']
    
    for index in range(len(label)):
        ax.plot(np.arange(1,len(data[index])+1),data[index,:], linewidth=3, markersize=10, marker=marker[index], linestyle=linestyle[index], color=colors[index], label=label[index])

    # build a legend using the labels we set previously


    ax.legend(['Visual-Mask','Visual-Normal','Thermal-Mask','Thermal-Normal'],loc='best')

    plt.xticks([1,3,10,30,80])
    plt.xlim((0.9,50))
    ax.set_xticklabels([1,3,10,30,80])

    plt.yticks([10**-5,10**-4,10**-3,10**-2,10**-1])
    ax.set_yticklabels([0.00001,10**-4,10**-3,10**-2,10**-1])
    plt.ylim((0.00001,0.5))

    plt.xlabel("Rank", fontsize=25)
    plt.ylabel("FNIR", fontsize=25)

    plt.title(str(name), fontsize=35)
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(name + ".pdf") as pdf:
        pdf.savefig(fig,bbox_inches='tight')

    # finally, show the plot
    plt.show()

def plot_roc(data, label, name):
    fig = plt.figure(figsize=(8, 8))

    plt.grid(True, which="both")
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rc('legend', fontsize=15)
    ax = fig.add_subplot(111)
    ax.ticklabel_format(useOffset=False)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)

    # colors = cm.tab20b(np.linspace(0, 1, len(label)))
    colors = ['red','blue','green','purple']
    # colors = [cm.Reds(np.linspace(0, 1, 10))[9],
    #           cm.Blues(np.linspace(0, 1, 10))[9],
    #           cm.Greens(np.linspace(0, 1, 10))[9],
    #           cm.Purples(np.linspace(0, 1, 10))[9]]
    marker = ['','','','']
    linestyle = ['solid','solid','dashed','dashed']
    
    for index in range(len(label)):
        ax.plot(data[index][0],data[index][1], linewidth=3, markersize=10, marker=marker[index], linestyle=linestyle[index], color=colors[index], label=label[index])

    # build a legend using the labels we set previously


    ax.legend(['Visual-Mask','Visual-Normal','Thermal-Mask','Thermal-Normal'],loc='best')

    # plt.xticks([1e-4, 1e-3, 1e-2,1e-1,1])
    plt.xlim((1e-4,1.1))
    # ax.set_xticklabels([1e-4, 1e-3, 1e-2,1e-1,1])

    # plt.yticks([10**-5,10**-4,10**-3,10**-2,10**-1])
    # ax.set_yticklabels([0.00001,10**-4,10**-3,10**-2,10**-1])
    plt.ylim((1e-5,1))

    plt.xlabel("FPIR", fontsize=25)
    plt.ylabel("FNIR", fontsize=25)

    plt.title(str(name), fontsize=35)
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages("roc_" + name + ".pdf") as pdf:
        pdf.savefig(fig,bbox_inches='tight')

    # finally, show the plot
    plt.show()
    
def fpr_fnr(df,max,mult=100):

    argsort_prediction = np.array(df.iloc[:,4:84])
    sort_predictions = np.array(df.iloc[:,84:])
    rates = np.zeros((max*mult,3))
    truth = np.array(df.loc[:,"Truth"].astype(int))
    
    for i in range(max*mult):
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        subjects = 80
        for j in range(len(sort_predictions)):
            arr = sort_predictions[j,:]
            dif = np.abs(arr-i/mult)
            index = np.argsort(dif)[0]
            if arr[index] < i/mult:
                index += 1
            
            fp += subjects - index
            tn += index
            arr = argsort_prediction[j,index:]
            
            
            if truth[j] in arr:
                tp +=1
                fp -=1
            else:
                tn -=1
                fn +=1
            
        rates[i,0] = fp/(fp+tn)
        rates[i,1] = fn/(fn+tp)
        rates[i,2] = i/mult
        print(i/mult, rates[i,0], rates[i,1])
    return rates

def roc(label_path, prediction_path):
    labels = np.load(label_path, allow_pickle=True)
    df = pd.DataFrame(labels)
    df = df.rename(columns = {0:"Truth", 1:"Age", 2:"Gender", 3:"Ethnicity"})
    truth = df.loc[:,"Truth"].astype(int)
    
    predictions = np.log(np.load(prediction_path, allow_pickle=True))

    min = np.min(predictions)
    max = np.max(predictions)
    predictions = predictions - min
    sort_predictions = np.sort(predictions,1)
    argsort_prediction = np.argsort(predictions,1)
    max = int(np.ceil(max-min))
    
    df =  pd.concat([df,pd.DataFrame(argsort_prediction)],axis=1)
    df =  pd.concat([df,pd.DataFrame(sort_predictions)],axis=1)
    
    curve = [[] for i in range(4)]
    curve[0] = fpr_fnr(df,max)

    for gender in df['Gender'].unique():
        curve[2].append(("gender:" + gender,fpr_fnr(df[df["Gender"]==gender],max)))

    for ethnicity in df['Ethnicity'].unique():
        curve[3].append(("ethnicity:" + ethnicity,fpr_fnr(df[df["Ethnicity"]==ethnicity],max)))        
        
    data = [[] for i in range(2)]
    label = [[] for i in range(2)]

    data[0] = []
    data[0].append(curve[0])
    label[0] = []
    label[0].append("Baseline")
    for gender in range(len(curve[2])):
        data[0].append(curve[2][gender][1])
        label[0].append(curve[2][gender][0][curve[2][gender][0].find(":")+1:])

    data[1] = []
    data[1].append(curve[0])
    label[1] = []
    label[1].append("Baseline")
    for ethnicity in range(len(curve[3])):
        data[1].append(curve[3][ethnicity][1])
        label[1].append(curve[3][ethnicity][0][curve[3][ethnicity][0].find(":")+1:])


    return data
    
def cmc(label_path, prediction_path, name):
    labels = np.load(label_path, allow_pickle=True)
    df = pd.DataFrame(labels)
    df = df.rename(columns = {0:"Truth", 1:"Age", 2:"Gender", 3:"Ethnicity"})

    predictions = np.load(prediction_path, allow_pickle=True)

    arg_predictions = np.argsort(predictions,1)
    arg_predictions = np.fliplr(arg_predictions)
    df =  pd.concat([df,pd.DataFrame(arg_predictions)],axis=1)

    curve = [[] for i in range(4)]
    curve[0] = calculate_curve(df)
    # for age in df['Age'].unique():
    #     curve[1].append(("age:" + age,calculate_curve(df[df["Age"]==age])))

    for gender in df['Gender'].unique():
        curve[2].append(("gender:" + gender,calculate_curve(df[df["Gender"]==gender])))

    for ethnicity in df['Ethnicity'].unique():
        curve[3].append(("ethnicity:" + ethnicity,calculate_curve(df[df["Ethnicity"]==ethnicity])))

    # data = curve[0]
    # label = []
    # label.append("Baseline")
    # for age in range(len(curve[1])):
    #     data = np.vstack((data,curve[1][age][1]))
    #     label.append(curve[1][age][0][curve[1][age][0].find(":")+1:])

    # plot_graph(data,label,name + "_age")
    data = [[] for i in range(2)]
    label = [[] for i in range(2)]

    data[0] = curve[0]
    label[0] = []
    label[0].append("Baseline")
    for gender in range(len(curve[2])):
        data[0] = np.vstack((data[0],curve[2][gender][1]))
        label[0].append(curve[2][gender][0][curve[2][gender][0].find(":")+1:])

    # plot_graph(data[0],label[0],name + "_gender")

    data[1] = curve[0]
    label[1] = []
    label[1].append("Baseline")
    for ethnicity in range(len(curve[3])):
        data[1] = np.vstack((data[1],curve[3][ethnicity][1]))
        label[1].append(curve[3][ethnicity][0][curve[3][ethnicity][0].find(":")+1:])

    # plot_graph(data[1],label[1],name + "_ethn")

    return data, label

def ddifference(label_path, prediction_path):
    labels = np.load(label_path, allow_pickle=True)
    predictions = np.load(prediction_path, allow_pickle=True)

    if len(predictions.shape) >1:
        predictions = np.argmax(predictions,1)
    ground_truth = np.array([int(labels[i,0]) for i in range(len(labels))])
    
    meta = pd.read_csv(r'E:\ken\database\speaking\\' + 'meta_subject.csv')

    # l = []
    # df=[]
    
    # for i in range(80):
    #     info = {}
    #     info.update({'subject':i+1})
    #     age = meta[meta['Sub_ID']==(i+1)]['Age'][i]
    #     gender = meta[meta['Sub_ID']==(i+1)]['Gender'][i]
    #     ethnicity = meta[meta['Sub_ID']==(i+1)]['Ethnicity'][i]

    #     info.update({'age':age})
    #     info.update({'gender':gender})
    #     info.update({'ethnicity':ethnicity})
    #     l.append(info)

    # df = pd.DataFrame(l)
    # pred_label = []
    # for sub in predictions:
    #     row = np.array(df[df['subject']==sub+1])
    #     pred_label.append(row[0])
    # pred_label = np.array(pred_label)
    
    dpd = [[] for i in range(3)]
    eod = [[] for i in range(3)]
    
    bt = ground_truth
    bp = predictions
    # binary_truth = []
    # binary_prediction = []
    lbl = labels
    
    # for i in range(80):
    #     binary_truth.append(ground_truth ==i)
    #     binary_prediction.append(predictions ==i)
    #     if i>0:
    #         lbl = np.vstack((lbl,label))
        
    # bt = np.reshape(binary_truth,(len(binary_truth)*len(binary_truth[0])))
    # bp = np.reshape(binary_prediction,(len(binary_prediction)*len(binary_prediction[0])))

    for cohort in range(1,4):
        print(cohort)
        dpd[cohort-1].append(demographic_parity_difference_personal([],bp==bt,lbl[:,cohort]))
        eod[cohort-1].append(equalized_odds_difference_personal(bt,bp,lbl[:,cohort]))
        
    p2 = np.vstack((np.mean(dpd,1),np.std(dpd,1),np.mean(eod,1),np.std(eod,1)))
         
    return p2
        
def difference(label_path, prediction_path):
    labels = np.load(label_path, allow_pickle=True)
    predictions = np.load(prediction_path, allow_pickle=True)

    if len(predictions.shape) >1:
        predictions = np.argmax(predictions,1)
    ground_truth = np.array([int(labels[i,0]) for i in range(len(labels))])

    matrix = classification_report(ground_truth,predictions)

    performance_values = [{} for i in range(80)]

    dpd = [[] for i in range(3)]
    eod = [[] for i in range(3)]

    fprd = [[] for i in range(3)]
    tprd = [[] for i in range(3)]

    for i in range(80):

        binary_truth = ground_truth ==i
        binary_prediction = predictions==i

        # for cohort in range(1,4):

        truth = binary_truth
        prediction = binary_prediction
        # group_label = labels[:,cohort]

        tpr = true_positive_rate(binary_truth,binary_prediction)
        fpr = false_positive_rate(binary_truth,binary_prediction)
        ppv = positive_prediction_value(binary_truth,binary_prediction)
        npv = negative_prediction_value(binary_truth,binary_prediction)
        tp, fp, tn, fn = values(binary_truth,binary_prediction)
        performance_values[i].update({'TPR':tpr})
        performance_values[i].update({'FPR':fpr})
        performance_values[i].update({'PPV':ppv})
        performance_values[i].update({'NPV':npv})

        performance_values[i].update({'TP':tp})
        performance_values[i].update({'FP':fp})
        performance_values[i].update({'TN':tn})
        performance_values[i].update({'FN':fn})
        performance_values[i].update({'support':tp+fp+tn+fn})
        performance_values[i].update({'subject':i})

        for cohort in range(1,4):
            print(i, cohort)
            dpd[cohort-1].append(demographic_parity_difference_personal(binary_truth,binary_prediction,group_label=labels[:,cohort]))
            eod[cohort-1].append(equalized_odds_difference_personal(binary_truth,binary_prediction,group_label=labels[:,cohort]))
            tprd[cohort-1].append(equalized_odds_difference_personal(binary_truth,binary_prediction,group_label=labels[:,cohort]))
            fprd[cohort-1].append(equalized_odds_difference_personal(binary_truth,binary_prediction,group_label=labels[:,cohort]))

    meta = pd.read_csv(r'E:\ken\database\speaking\\' + 'meta_subject.csv')

    l = []
    df=[]
    for i in range(80):
        info = performance_values[i]
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

    support = df.iloc[:,4].sum(0)
    mean = df.iloc[:,:4].mean()
    std = df.iloc[:,:4].std(ddof=0)
    temp = df.iloc[:,:4].sum(0)/len(df)
    temp = temp.append(pd.Series({'support':support}))
    temp = temp.append(pd.Series({'subject':'baseline'}))
    df_edit = df_edit.append(temp, ignore_index=True)

    for age in df['age'].unique():
        temp = df[df['age']==age]
        support = temp.iloc[:,4].sum(0)
        temp = temp.iloc[:,:4].sum(0)/len(temp)
        temp = temp.append(pd.Series({'support':support}))
        temp = temp.append(pd.Series({'subject':f'age:{age}'}))
        df_edit = df_edit.append(temp, ignore_index=True)

    for gender in df['gender'].unique():
        temp = df[df['gender']==gender]
        support = temp.iloc[:,4].sum(0)
        temp = temp.iloc[:,:4].sum(0)/len(temp)
        temp = temp.append(pd.Series({'support':support}))
        temp = temp.append(pd.Series({'subject':f'gender:{gender}'}))
        df_edit = df_edit.append(temp, ignore_index=True)\

    for ethnicity in df['ethnicity'].unique():
        temp = df[df['ethnicity']==ethnicity]
        support = temp.iloc[:,4].sum(0)
        temp = temp.iloc[:,:4].sum(0)/len(temp)
        temp = temp.append(pd.Series({'support':support}))
        temp = temp.append(pd.Series({'subject':f'ethnicity:{ethnicity}'}))
        df_edit = df_edit.append(temp, ignore_index=True)

    df = df_edit
    p1 = np.vstack((mean,std))

    p2 = np.vstack((np.mean(dpd,1),np.std(dpd,1),np.mean(eod,1),np.std(eod,1),np.mean(tprd,1),np.std(tprd,1),np.mean(fprd,1),np.std(fprd,1)))

    return df, p1, p2

if __name__ == "__main__":

    sp = 'speak_10_90_nonoise_cnn3'
    test = '_test'
    test = ''

    p1 = [[] for i in range(4)]
    p2 = [[] for i in range(4)]
    df = [[] for i in range(4)]

    d = [[] for i in range(4)]
    l = [[] for i in range(4)]
    rates = [[] for i in range(4)]

    label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_mask'+ test + '_labels.npy'
    prediction_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_mask_predictions.npy'
    # d[0], l[0] = cmc(label_path, prediction_path, "vm")
    # rates[0] = roc(label_path, prediction_path)
    p2[0]= ddifference(label_path, prediction_path)
    # df[0], p1[0], p2[0]= difference(label_path, prediction_path)

    label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_normal'+ test + '_labels.npy'
    prediction_path = r'E:\ken\synthetic_fairness\\' + sp + '\\visual_normal_predictions.npy'
    # d[1], l[1] = cmc(label_path, prediction_path, "vn")
    # rates[1] = roc(label_path, prediction_path)
    p2[1]= ddifference(label_path, prediction_path)
    # df[1], p1[1], p2[1]= difference(label_path, prediction_path)

    label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\thermal_mask'+ test + '_labels.npy'
    prediction_path = r'E:\ken\synthetic_fairness\\' + sp + '\\thermal_mask_predictions.npy'
    # d[2], l[2] = cmc(label_path, prediction_path, "tm")
    # rates[2] = roc(label_path, prediction_path)
    p2[2]= ddifference(label_path, prediction_path)
    # df[2], p1[2], p2[2]= difference(label_path, prediction_path)

    label_path = r'E:\ken\synthetic_fairness\\' + sp + '\\thermal_normal'+ test + '_labels.npy'
    prediction_path = r'E:\ken\synthetic_fairness\\' + sp + '\\thermal_normal_predictions.npy'
    # d[3], l[3] = cmc(label_path, prediction_path, "tn")
    # rates[3] = roc(label_path, prediction_path)
    p2[3]= ddifference(label_path, prediction_path)
    # df[3], p1[3], p2[3]= difference(label_path, prediction_path)


    # data = []
    # label = []
    # for i in range(4):
    #     data.append(d[i][0][0])
    #     label.append(str(i) + l[i][0][0])
    # plot_graph(np.array(data),label,"Baseline")

    # gender = ['Gender-M','Gender-F']
    # for g in range(1,3):
    #     data = []
    #     label = []
    #     for i in range(4):
    #         data.append(d[i][0][g])
    #         label.append(str(i) + l[i][0][g])

    #     plot_graph(np.array(data),label,gender[g-1])


    # ethnicity = ['Ethnicity-A','Ethnicity-C','Ethnicity-B']
    # for eth in range(1,4):
    #     data = []
    #     label = []
    #     for i in range(4):
    #         data.append(d[i][1][eth])
    #         label.append(str(i) + l[i][1][eth])

    #     plot_graph(np.array(data),label,ethnicity[eth-1])

    # data = []
    # label = []
    # for i in range(4):
    #     data.append((rates[i][0][0][:,0],rates[i][0][0][:,1]))
    #     label.append(str(i) + l[i][0][0])
    # plot_roc(np.array(data),label,"Baseline")
    
    # gender = ['Gender-M','Gender-F']
    # for g in range(1,3):
    #     data = []
    #     label = []
    #     for i in range(4):
    #         data.append((rates[i][0][g][:,0],rates[i][0][g][:,1]))
    #         label.append(str(i) + l[i][0][g])

    #     plot_roc(np.array(data),label,gender[g-1])
        
    # ethnicity = ['Ethnicity-A','Ethnicity-C','Ethnicity-B']
    # for eth in range(1,4):
    #     data = []
    #     label = []
    #     for i in range(4):
    #         data.append((rates[i][1][eth][:,0],rates[i][1][eth][:,1]))
    #         label.append(str(i) + l[i][1][eth])

    #     plot_roc(np.array(data),label,ethnicity[eth-1])        