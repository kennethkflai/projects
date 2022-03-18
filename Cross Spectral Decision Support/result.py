
from glob import glob

import matplotlib.pyplot as plt


import numpy as np
types = ['rgb', 'nir', 'ir', 'cs']
data = [[[] for t in types] for t in types]
for train_type in range(len(types)):
    for test_type in range(len(types)):
        pred_path = 'result//pred-' + types[test_type] + '-' + types[train_type] + '.npy'
        truth_path = 'result//truth-' + types[test_type] + '-' + types[train_type] + '.npy'
        pred = np.load(pred_path)
        truth = np.load(truth_path)
        p = np.argmax(pred,1)
        t = np.argmax(truth,1)
#        print(np.sum(p==t)/len(t))

        p = np.argsort(-pred,1)
        acc =0
        for j in range(0, len(p[0])):
            tp =0
            for i in range(0, len(p)):
                if p[i][j] == t[i]:
                    tp = tp +1
            acc += tp/len(p)
#            print(acc)
            data[train_type][test_type].append(acc)

app =np.zeros((4,4))

for i in range(len(data)):
    for j in range(len(data[0])):
        app[i][j]=data[i][j][9]

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
for j in range(len(types)):
    fig, ax = plt.subplots()

    for i in range(len(types)):
        ax.plot(np.arange(1, len(data[j][i])+1), data[j][i], linewidth=3)

    cap = [types[i].upper() for i in range(len(types))]
    ax.legend(cap, fontsize='xx-large')
    ax.set_xlabel('xlabel', fontsize=20)
    ax.set_ylabel('ylabel', fontsize=20)
    plt.xlim(1, 112)
    ax.set_xscale('log')
    plt.xticks([1, 5, 10, 20, 40, 100], [1, 5, 10, 20, 40, 100])
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set(xlabel='Rank', ylabel='Identification Rate')
    ax.grid()

    fig.savefig("test" + str(j) + ".pdf")

def draw_cm(trueP, modelP, nm, cmap=plt.cm.Purples):


    import itertools

    matrix = confusion_matrix(trueP, modelP)
    np.set_printoptions(precision=8)
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    matrix = matrix.astype('float') / matrix.sum(axis=1)[ np.newaxis, :] * 100
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    if len(matrix) == 5:
        classes = ["Neutral", "Smile", "Sleepy", "Shock", "Sunglasses"]
    else:
        classes = ["Neutral", "Smile", "Sleepy", "Shock"]

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, fontsize=25)
    plt.yticks(tick_marks, classes, fontsize=25)
#    plt.yticks(tick_marks, ["","","","",""], fontsize=0)
    thresh = matrix.max()/2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        if matrix[i, j] > thresh:
            clr = "white"
        else:
            clr = "black"
        plt.text(j, i, format(matrix[i, j], '2.1f'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color=clr, fontsize=25)
#    plt.xlabel('Predicted Label', fontsize=15)
#    plt.ylabel('True Label', fontsize=15)
#    plt.show()
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(nm) as pdf:
        pdf.savefig(fig,bbox_inches='tight')

from sklearn.metrics import confusion_matrix
acc=[]

typ = 'rgb'
pred_label=[]
t_label=[]
subject=[]

for i in range(0, 5):
    pred_path = 'result//emo' + typ + '2pred-' + str(i) + '.npy'
    truth_path = 'result//emo' + typ + '2truth-' + str(i) + '.npy'
    subject_path = 'result//emo' + typ + '2subject-' + str(i) + '.npy'
    pred = np.load(pred_path)
    truth = np.load(truth_path)
    subject += list(np.load(subject_path))
    pred_label += list(np.argmax(pred,1))
    t_label += list(np.argmax(truth,1))

sub_acc = np.zeros((np.max(subject)+1,2))
for i in range(len(pred_label)):
    if pred_label[i]==t_label[i]:
        sub_acc[subject[i],0] += 1
    sub_acc[subject[i],1] += 1

acc.append(sub_acc[:,0]/sub_acc[:,1])

matrix = confusion_matrix(pred_label, t_label)
draw_cm(t_label,pred_label, 'cm'+ typ + '2.pdf')

pred_label=[]
t_label=[]
subject=[]

for i in range(0, 5):
    pred_path = 'result//emo' + typ + 'pred-' + str(i) + '.npy'
    truth_path = 'result//emo' + typ + 'truth-' + str(i) + '.npy'
    subject_path = 'result//emo' + typ + 'subject-' + str(i) + '.npy'
    pred = np.load(pred_path)
    truth = np.load(truth_path)
    subject += list(np.load(subject_path))
    pred_label += list(np.argmax(pred,1))
    t_label += list(np.argmax(truth,1))

sub_acc = np.zeros((np.max(subject)+1,2))
for i in range(len(pred_label)):
    if pred_label[i]==t_label[i]:
        sub_acc[subject[i],0] += 1
    sub_acc[subject[i],1] += 1

acc.append(sub_acc[:,0]/sub_acc[:,1])

matrix = confusion_matrix(pred_label, t_label)
draw_cm(t_label,pred_label, 'cm'+ typ + '.pdf')


typ = 'ir'
pred_label=[]
t_label=[]
subject=[]

for i in range(0, 5):
    pred_path = 'result//emo' + typ + '2pred-' + str(i) + '.npy'
    truth_path = 'result//emo' + typ + '2truth-' + str(i) + '.npy'
    subject_path = 'result//emo' + typ + '2subject-' + str(i) + '.npy'
    pred = np.load(pred_path)
    truth = np.load(truth_path)
    subject += list(np.load(subject_path))
    pred_label += list(np.argmax(pred,1))
    t_label += list(np.argmax(truth,1))

sub_acc = np.zeros((np.max(subject)+1,2))
for i in range(len(pred_label)):
    if pred_label[i]==t_label[i]:
        sub_acc[subject[i],0] += 1
    sub_acc[subject[i],1] += 1

acc.append(sub_acc[:,0]/sub_acc[:,1])

matrix = confusion_matrix(pred_label, t_label)
draw_cm(t_label,pred_label, 'cm'+ typ + '2.pdf', cmap=plt.cm.Greens)

pred_label=[]
t_label=[]
subject=[]

for i in range(0, 5):
    pred_path = 'result//emo' + typ + 'pred-' + str(i) + '.npy'
    truth_path = 'result//emo' + typ + 'truth-' + str(i) + '.npy'
    subject_path = 'result//emo' + typ + 'subject-' + str(i) + '.npy'
    pred = np.load(pred_path)
    truth = np.load(truth_path)
    subject += list(np.load(subject_path))
    pred_label += list(np.argmax(pred,1))
    t_label += list(np.argmax(truth,1))

sub_acc = np.zeros((np.max(subject)+1,2))
for i in range(len(pred_label)):
    if pred_label[i]==t_label[i]:
        sub_acc[subject[i],0] += 1
    sub_acc[subject[i],1] += 1

acc.append(sub_acc[:,0]/sub_acc[:,1])

matrix = confusion_matrix(pred_label, t_label)
draw_cm(t_label,pred_label, 'cm'+ typ + '.pdf', cmap=plt.cm.Greens)
