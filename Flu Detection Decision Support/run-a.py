from util.model import model
import numpy as np
from keras import backend as K
import argparse
import cv2
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    _argparser = argparse.ArgumentParser(
            description='Gesture Recognition',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--timestep', type=int, default=5, metavar='INTEGER',
        help='Time step in network')
    _argparser.add_argument(
        '--cycle', type=int, default=5, metavar='INTEGER',
        help='Number of cycles')
    _argparser.add_argument(
        '--base', type=int, default=100, metavar='INTEGER',
        help='Starting number of epochs')
    _argparser.add_argument(
        '--mult', type=float, default=1., metavar='FLOAT',
        help='Epoch multiplier')
    _argparser.add_argument(
        '--cv', type=int, default=20, metavar='INTEGER',
        help='Cross-Validation Set')
    _argparser.add_argument(
        '--modeltype', type=int, default=1, metavar='INTEGER',
        help='Model Type')
    _argparser.add_argument(
        '--lr', type=float, default=1e-3, metavar='FLOAT',
        help='Learning rate')
    _argparser.add_argument(
        '--wd', type=float, default=0, metavar='FLOAT',
        help='Weight decay')
    _argparser.add_argument(
        '--batch_size', type=int, default=32, metavar='INTEGER',
        help='Batch Size')
    _argparser.add_argument(
        '--normalize_point', type=int, default=0, metavar='INTEGER',
        help='Normalize Skeleton Point')
    _argparser.add_argument(
        '--name', type=int, default=0, metavar='INTEGER',
        help='Features based on model')
    _args = _argparser.parse_args()

    num_frame = 2**_args.timestep
    model_type = _args.modeltype
    cv = _args.cv
    lr = _args.lr
    wd = _args.wd
    num_cycle = _args.cycle
    base_epoch = _args.base
    epoch_mult = _args.mult
    batch_size = _args.batch_size
    norm_point = _args.normalize_point
    model_num = _args.name


    model_name = {0:"VGG16",
                  1:"VGG19",
                  2:"ResNet50",
                  3:"InceptionV3",
                  4:"InceptionResNetV2",
                  5:"Xception",
                  6:"NASNetLarge",
                  7:"DenseNet201"}

    if model_type == 0:
        temp_data = np.load('./data/data_points.npy', allow_pickle=True)
        label = np.load('./data/label_points.npy', allow_pickle=True)
        fs=(17,2)
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
    elif model_type == 1:
        temp_data = np.load('./data/data_' + model_name[model_num] + '.npy', allow_pickle=True)
        label = np.load('./data/label_' + model_name[model_num] + '.npy', allow_pickle=True)

        data=[]
        for i in range(len(temp_data)):
            t_data=[]
            for j in range(len(temp_data[i])):
                subject_data = temp_data[i][j]
                fs=len(subject_data[0])
                subject_data = cv2.resize(subject_data,(fs, num_frame), interpolation=cv2.INTER_LANCZOS4)

                t_data.append(subject_data)
            data.append(t_data)

    else:
        data = np.load('./data/data_image.npy', allow_pickle=True)

        store = []
        for i in range(len(data)):
            act_store = []
            print(i)
            for j in range(len(data[i])):
                time_store = []
                different = num_frame - len(data[i][j])
                for cc in range(different):
                    time_store.append(np.zeros((200,100,3)))

                for k in range(len(data[i][j])):
                    arr = np.array(data[i][j][k]/255.)
                    time_store.append(arr)

                while(len(time_store) > num_frame):
                    time_store.pop(0)

                act_store.append(time_store)
            store.append(act_store)

        data = store
        label = np.load('./data/label_image.npy', allow_pickle=True)

    subject = len(label)
    un = np.unique(label)
    un = [str(i) for i in un]
    lbl_dict = {un[i]:i  for i in range(0,len(un))}

    train_data = []
    val_data = []
    train_label = []
    val_label = []

    for index in range(subject):
        if index <6 and index >0:
            val_data += list(data[index])
            val_label += list(label[index])
        else:
            train_data += list(data[index])
            train_label += list(label[index])


    train_label = [lbl_dict[i] for i in train_label]
    val_label = [lbl_dict[i] for i in val_label]

    num_classes = len(np.unique(train_label))

    t_model = model(num_classes, model_type=(model_type, model_num), wd=wd, lr=lr, num_frame=num_frame, feature_size = fs)

    train_data = np.array(train_data)
    val_data = np.array(val_data)

    save_file = t_model.train(train_data, train_label, norm_point,
                            val_data, val_label, batch_size,
                            base_epoch=base_epoch, epoch_mult=epoch_mult,
                            num_cycle=num_cycle)

    t_model.load(save_file)
    pred_label = t_model.predict((val_data))

    pd =[]
    for i in range(0, len(pred_label)):

        pd.append(np.argmax(pred_label[i], 1))
        print(np.sum(pd[i]==val_label)/len(val_label))
        temp = pd[i] == val_label

        f = open('acc.txt', 'a')
        if model_type==1:
            f.write("TS: %3.0f, model: %2d, BatchSize: %3d, Accuracy: %.2f%%\n"
                    % (model_name[model_num], num_frame, i, batch_size, 100*np.sum(temp)/len(temp)))       
        else:
            f.write("TS: %3.0f, model: %2d, BatchSize: %3d, Accuracy: %.2f%%\n"
                % (num_frame, i, batch_size, 100*np.sum(temp)/len(temp)))
        f.close()

    del t_model
    K.clear_session()
