
from glob import glob

import numpy as np

import cv2

path = r'Assessing Risks of Biases\face\*\*'
p = glob(path)
for i in p:
    img = cv2.imread(i)
    cv2.imwrite(img,i[:-3] + '.png')

data = np.load('predictions.npy')
t_label = np.int32(data[:, -1])
pred_label = np.int32(data[:, 1])

print(sum(t_label==pred_label)/len(t_label))

path = r'database\FERET Colour Facial Image Database\*\data\ground_truths\name_value\*'
list_path = glob(path)
sub_dict = []
for p in list_path:
    subject = p[p.rfind('\\')+1:]
    sub_dict.append(subject)

sub_dict = {i: sub_dict[i] for i in range(len(sub_dict))}
a = np.random.randint(0,len(sub_dict),10)
a = np.sort(a)

gen_score=[]
imp_score=[]

for i in range(0, len(data)):
    for j in range(1, 10, 2):
        if data[i][-1]==data[i][j]:
            gen_score.append(data[i][j+1])
        else:
            imp_score.append(data[i][j+1])

gen_score = np.array(np.float32(gen_score))
imp_score = np.array(np.float32(imp_score))
attr=[]
for i in range(len(data)):
    name = data[i][0]
    subject = sub_dict[np.int32(data[i][-1])]

    path = 'data\\' + subject + '\\' + name + '.npy'
    t = np.load(path)
    if t[2].find('Asian') >= 0:
        t[2] = 'Asian'

    date = t[1][-2]
    t[1] = str('19' + date + '0s')
    attr.append(t)

attr = np.array(attr)

a = np.random.randint(0,len(sub_dict),10)
a = np.sort(a)
a= [15, 189, 143, 295, 504, 561,894,917,948,684]
t=[]
for i in range(len(a)):
    for ii in range(len(data)):
        if np.int32(data[ii][-1]) == a[i]:
            temp = [a[i], data[ii][1], data[ii][2], data[ii][3], data[ii][4], data[ii][5],
                    data[ii][6], data[ii][7], data[ii][8], data[ii][9], data[ii][10],
                        attr[ii,0], attr[ii,1], attr[ii,2],attr[ii,5], attr[ii,6], attr[ii,7]]
            t.append(temp)
a=[ 0, 1046, 271, 280, 312]
for i in a:
    temp = [data[i][-1], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5],
            data[i][6], data[i][7], data[i][8], data[i][9], data[i][10],
                attr[i,0], attr[i,1], attr[i,2],attr[i,5], attr[i,6], attr[ii,7]]
    t.append(temp)

def calc_rates(l):
    li = np.unique(l)
    rate = np.zeros((len(li)))
    for i in range(len(li)):
        temp = l==li[i]
        rate[i] = np.sum(temp)/len(temp)
    return rate, li

g_rate, g_label = calc_rates(attr[:,0])
yob_rate, yob_label = calc_rates(attr[:,1])
race_rate, race_label = calc_rates(attr[:,2])
pose_rate, pose_label = calc_rates(attr[:,3])
yaw_rate, yaw_label = calc_rates(attr[:,4])
glasses_rate, glasses_label = calc_rates(attr[:,5])
beard_rate, beard_label = calc_rates(attr[:,6])
mustache_rate, mustache_label = calc_rates(attr[:,7])

def perf_measure(matrix, div):
    sum_col = np.sum(matrix, axis=0)
    sum_row = np.sum(matrix, axis=1)


    tp = np.diag(matrix)
    fn = sum_col-tp
    temp = (tp+fn)
    zero_se = np.sum(temp==0)
    sensitivity_per_row = np.divide(tp,sum_col, out=np.zeros(tp.shape, dtype=float), where=sum_col!=0)

    tn = np.sum(tp)-tp
    fp = sum_row-tp

    temp = (tn+fp)
    zero_sp = np.sum(temp==0)
    specificity_per_row =np.divide(tn,temp, out=np.zeros(tn.shape, dtype=float), where=temp!=0)

    accuracy = np.sum(tp)/np.sum(matrix)

    sensitivity = sum(sensitivity_per_row)/(div)
    specificity = sum(specificity_per_row)/(len(specificity_per_row))
    return(accuracy, sensitivity, specificity)

def calc_rates2(truth, pred, l):
    from sklearn.metrics import confusion_matrix
    li = np.unique(l)
    rate = np.zeros((3,len(li)))
    for i in range(len(li)):
        temp = l==li[i]
        new_truth = truth[temp]
        new_pred = pred[temp]

        matrix = confusion_matrix(new_pred, new_truth)

        rate[0,i], rate[1,i], rate[2,i] = perf_measure(matrix, len(np.unique(new_truth)))
        rate[1,i] = 1-rate[1,i]
        rate[2,i] = 1-rate[2,i]
    return np.vstack((li, rate))

d=[]
for i in range(8):
    d.append(calc_rates2(t_label, pred_label, attr[:,i]))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(pred_label, t_label)

acc, se, sp = perf_measure(matrix, len(np.unique(t_label)))

print(acc, 1-se, 1-sp)

def draw_cm(trueP, modelP, nm):
    import matplotlib.pyplot as plt

    import itertools

    matrix = confusion_matrix(trueP, modelP)
    np.set_printoptions(precision=8)
    fig = plt.figure(figsize=(15, 15))
    fig.tight_layout()
    matrix = matrix.astype('float') / matrix.sum(axis=0)[ np.newaxis, :] * 100
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Purples)

    classes = ["FH", "FF", "FB", "FS", "FE", "W", "ST", "SI", "P", "J", "L"]

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, fontsize=25)
    plt.yticks(tick_marks, classes, fontsize=25)
    thresh = matrix.max()/2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        if matrix[i, j] > thresh:
            clr = "white"
        else:
            clr = "black"
        plt.text(j, i, format(matrix[i, j], '2.1f'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color=clr, fontsize=20)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(nm) as pdf:
        pdf.savefig(fig,bbox_inches='tight')

