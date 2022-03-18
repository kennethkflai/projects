from util.model import model
import numpy as np
from keras import backend as K
import argparse
import cv2
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
def draw_cm(trueP, modelP, nm, cmap=plt.cm.Purples):
    import itertools

    matrix = confusion_matrix(trueP, modelP)
    np.set_printoptions(precision=8)
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    matrix = matrix.astype('float') / matrix.sum(axis=1)[ np.newaxis, :] * 100
#    matrix = matrix.astype('float')
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    classes = ["Call", "Cough", "Drink", "Scratch",
               "Sneeze", "Stretch", "Wave", "Wipe"]

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
                 color=clr, fontsize=18)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(nm) as pdf:
        pdf.savefig(fig,bbox_inches='tight')


if __name__ == "__main__":
    num_frame = 64
    model_type = 0
    cv = 0
    lr = 0.001
    wd = 0.001
    num_cycle = 1
    base_epoch = 1000
    epoch_mult = 1
    batch_size = 4
    norm_point = 6#_args.normalize_point
    model_num = 0

    temp_data = np.load('./data/data_points.npy', allow_pickle=True)
    label = np.load('./data/label_points.npy', allow_pickle=True)
    fs=(17,2)

    gender_label = np.zeros((20,96))
    pose_label = np.zeros((20,96))
    view_label = np.zeros((20,96))

    gender = np.array([10, 11, 13, 14, 16, 17, 19, 20])-1
    for i in range(0, 20):
        for j in range(0, 96):
            if i in gender:
                gender_label[i][j] = 1

            if j %4 <2:
                pose_label[i][j] = 1

            if j% 12 <4:
                view_label[i][j]=1
            elif j%12 <8:
                view_label[i][j]=2

    subject = len(label)
    un = np.unique(label)
    un = [str(i) for i in un]
    lbl_dict = {un[i]:i  for i in range(0,len(un))}
    if model_num==0:
        data=[]
        for i in range(len(temp_data)):
            t_data=[]
            for j in range(len(temp_data[i])):
                subject_data = temp_data[i][j]
                subject_data = subject_data[:,:,0:2]
                subject_data = cv2.resize(subject_data,(17, num_frame), interpolation=cv2.INTER_LANCZOS4)

                for k in range(len(subject_data)):
                    subject_data[k,:,0] = (subject_data[k,:,0]-subject_data[k,norm_point,0])/290
                    subject_data[k,:,1] = (subject_data[k,:,1]-subject_data[k,norm_point,0])/480
                t_data.append(subject_data)
            data.append(t_data)

        train_data = []
        val_data = []
        train_label = []
        val_label = []
        x=[]
        y=[]
        z=[]
        for index in range(subject):
            if index <6 and index >0:
                val_data += list(data[index])
                val_label += list(label[index])

                x += list(gender_label[index])
                y += list(pose_label[index])
                z += list(view_label[index])
            else:
                train_data += list(data[index])
                train_label += list(label[index])
    elif model_num == 3:
        data=[]
        for i in range(len(temp_data)):
            t_data=[]
            for j in range(len(temp_data[i])):
                subject_data = temp_data[i][j]
                subject_data = subject_data[:,:,0:2]
                subject_data = cv2.resize(subject_data,(17, num_frame), interpolation=cv2.INTER_LANCZOS4)

                for k in range(len(subject_data)):
                    subject_data[k,:,0] = (subject_data[k,:,0]-subject_data[k,norm_point,0])/290
                    subject_data[k,:,1] = (subject_data[k,:,1]-subject_data[k,norm_point,0])/480
                t_data.append(subject_data)
            data.append(t_data)


        train_data = []
        val_data = []
        train_label = []
        val_label = []

        gender = np.array([10, 11, 13, 14, 16, 17, 19, 20])-1
        for index in range(subject):
            if cv:
                if index in gender:
                    val_data += list(data[index])
                    val_label += list(label[index])
                else:
                    train_data += list(data[index])
                    train_label += list(label[index])
            else:
                if index in gender:
                    train_data += list(data[index])
                    train_label += list(label[index])
                else:
                    val_data += list(data[index])
                    val_label += list(label[index])
    elif model_num ==4:
        p1 = []
        p2 = []
        p1_label = []
        p2_label = []
        for i in range(len(temp_data)):
            for j in range(len(temp_data[i])):
                subject_data = temp_data[i][j]
                subject_data = subject_data[:,:,0:2]
                subject_data = cv2.resize(subject_data,(17, num_frame), interpolation=cv2.INTER_LANCZOS4)

                for k in range(len(subject_data)):
                    subject_data[k,:,0] = (subject_data[k,:,0]-subject_data[k,norm_point,0])/290
                    subject_data[k,:,1] = (subject_data[k,:,1]-subject_data[k,norm_point,0])/480

                if j%4 <2:
                    p1.append(subject_data)
                    p1_label.append(label[i][j])
                else:
                    p2.append(subject_data)
                    p2_label.append(label[i][j])
        train_data = []
        val_data = []
        train_label = []
        val_label = []

        if cv:
            val_data = p1
            val_label = p1_label
            train_data = p2
            train_label = p2_label
        else:
            val_data = p2
            val_label = p2_label
            train_data = p1
            train_label = p1_label
    else:
        v1 = []
        v2 = []
        v3 = []
        v1_label = []
        v2_label = []
        v3_label = []
        for i in range(len(temp_data)):
            for j in range(len(temp_data[i])):
                subject_data = temp_data[i][j]
                subject_data = subject_data[:,:,0:2]
                subject_data = cv2.resize(subject_data,(17, num_frame), interpolation=cv2.INTER_LANCZOS4)

                for k in range(len(subject_data)):
                    subject_data[k,:,0] = (subject_data[k,:,0]-subject_data[k,norm_point,0])/290
                    subject_data[k,:,1] = (subject_data[k,:,1]-subject_data[k,norm_point,0])/480

                if j%12 <4:
                    v1.append(subject_data)
                    v1_label.append(label[i][j])
                elif j%12<8:
                    v2.append(subject_data)
                    v2_label.append(label[i][j])
                else:
                    v3.append(subject_data)
                    v3_label.append(label[i][j])

        train_data = []
        val_data = []
        train_label = []
        val_label = []
        if cv == 0:
            val_data = v1
            val_label = v1_label
            train_data = v2 + v3
            train_label = v2_label + v3_label
        elif cv == 1:
            val_data = v2
            val_label = v2_label
            train_data = v1 + v3
            train_label = v1_label + v3_label
        else:
            val_data = v3
            val_label = v3_label
            train_data = v1 + v2
            train_label = v1_label + v2_label

    train_label = [lbl_dict[i] for i in train_label]
    val_label = [lbl_dict[i] for i in val_label]

    num_classes = len(np.unique(train_label))

    t_model = model(num_classes, model_type=(model_type, model_num), wd=wd, lr=lr, num_frame=num_frame, feature_size = fs)

    train_data = np.array(train_data)
    val_data = np.array(val_data)

#    save_file = t_model.train(train_data, train_label, norm_point,
#                            val_data, val_label, batch_size,
#                            base_epoch=base_epoch, epoch_mult=epoch_mult,
#                            num_cycle=num_cycle)

#    save_file = r'.\save\models\best0-1-cv6.hdf5'
    save_file = r'.\save\models\\' + str(model_type) + '-' + str(model_num) + '-cv' + str(cv) + '.hdf5'
    t_model.load(save_file)
    pred_label = t_model.predict((val_data))

    pd =[]
    for i in range(0, len(pred_label)):

        pd.append(np.argmax(pred_label[i], 1))
        print(np.sum(pd[i]==val_label)/len(val_label))
        temp = pd[i] == val_label

        f = open('acc.txt', 'a')

        f.write("TS: %3.0f, model: %2d, BatchSize: %3d, Accuracy: %.2f%%\n"
                % (num_frame, i, batch_size, 100*np.sum(temp)/len(temp)))
        f.close()

    del t_model
    K.clear_session()

    for i in range(0, 1):
        ss=str(model_num) + '-' + str(cv) + '-' + str(i) + '.pdf'
        ss='cm.pdf'
        pred =[]
        val = []
        for j in range(len(val_label)):
#            if z[j]==0:
            pred.append(pd[i][j])
            val.append(val_label[j])

        draw_cm(val,pred, ss, cmap=plt.cm.Greens)

