import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from util.model import model
import numpy as np
from keras import backend as K
from util.model import model_set
from util.data_process import data_set
from util.data_process import get_data_skele_npy, get_data_img_npy

data_path = r'E:\ken\2021-PRLetter - Capturing Causality and Bias in Human Action Recognition\fall2\\HAR-UP\\'

model_list = {0: "test", 1: "test1", 2: "test2", 3: "test3", 4: "test4"}
channel = 2
binary = True
num_frame = 20

def get_data(model_type):
    from readcsv import get_csv
    data, label = get_csv(skip=int(np.ceil(num_frame/2)), num_frame=num_frame, binary=binary)
    batch_size = 1024

    return data, label, batch_size

def model_train_eval(num_frame, val_index,
                        num_classes, model_type,
                        train_data, train_label,
                        val_data, val_label,
                        batch_size, wd, lr, base_epoch, epoch_mult, num_cycle):

        test_model = model(num_classes, model_type=model_type, wd=wd, lr=lr, binary=binary)

        test_model.train(train_data, train_label, num_frame, val_data, val_label, bs=batch_size, base_epoch=base_epoch, epoch_mult=epoch_mult, num_cycle=num_cycle, binary=binary)

        model_path = model_list[model_type] + '-models//cv' + str(cv) + '.hdf5'

        test_model.load(model_path)

        test_data = np.array(val_data)
        pred_label = test_model.predict(test_data)

        if type(pred_label) == np.ndarray:
            if binary == True:
                pred_label = pred_label >0.5
                pred_label = pred_label[:,0]
            else:
                pred_label = np.argmax(pred_label,1)
            temp = pred_label == val_label
            print(np.sum(temp)/len(val_label))
            f = open(model_list[model_type] + '-models//acc.txt', 'a')
            f.write("Type: %s, TS: %3.0f, cv: %2.0f, BatchSize: %3.0f, Accuracy: %.2f%%\n"
                    % (model_list[model_type], num_frame, val_index, batch_size, 100*sum(temp)/len(temp)))
            f.close()
        else:
            pd =[]
            for i in range(0, len(pred_label)):
                if binary == True:
                    temp = pred_label[i]>0.5
                    temp = temp[:,0]
                    pd.append(temp)
                else:
                    pd.append(np.argmax(pred_label[i], 1))
                print(np.sum(pd[i]==val_label)/len(val_label))
                temp = pd[i] == val_label

                f = open(model_list[model_type] + '-models//acc.txt', 'a')
                f.write("Type: %s, TS: %3.0f, cv: %2.0f, BatchSize: %3.0f, Accuracy: %.2f%%\n"
                        % (model_list[model_type], num_frame, val_index, batch_size, 100*sum(temp)/len(temp)))
                f.close()

        del test_model
        K.clear_session()

def test_models(num_frame, val_index, num_classes, model_type, wd, lr, num_cycle):
    test_model = model(num_classes, model_type=model_type, wd=wd, lr=lr)

    for j in range(0, num_cycle):
        model_path = model_list[model_type] + '-models//cv' + str(val_index) + '-' +  str(j) + '.hdf5'
        test_model.load(model_path)
        path_string = (data_path + 'Subject'+ str(val_index+1) +'*')
        test_data, test_label = get_data_skele_npy(path_string, end_index=-1, skip=int(np.ceil(num_frame/2)))
        test_data = np.array(test_data)
        pred_label = test_model.predict(test_data)

        np.save(model_list[model_type] + '-models//cm//pred' + str(val_index) + '-' + str(j), pred_label)
        np.save(model_list[model_type] + '-models//cm//truth' + str(val_index) + '-' + str(j), test_label)


if __name__ == "__main__":
    _argparser = argparse.ArgumentParser(
        description='Gesture Recognition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--timestep', type=int, default=20, metavar='INTEGER',
        help='Time step in network')
    _argparser.add_argument(
        '--cycle', type=int, default=10, metavar='INTEGER',
        help='Number of cycles')
    _argparser.add_argument(
        '--base', type=int, default=100, metavar='INTEGER',
        help='Starting number of epochs')
    _argparser.add_argument(
        '--mult', type=float, default=1., metavar='FLOAT',
        help='Epoch multiplier')
    _argparser.add_argument(
        '--cv', type=int, default=0, metavar='INTEGER',
        help='Cross-Validation Set')
    _argparser.add_argument(
        '--modeltype', type=int, default=0, metavar='INTEGER',
        help='Model Type')
    _argparser.add_argument(
        '--lr', type=float, default=1e-3, metavar='FLOAT',
        help='Learning rate')
    _argparser.add_argument(
        '--wd', type=float, default=0, metavar='FLOAT',
        help='Weight decay')
    _argparser.add_argument(
        '--evaluate', action='store_true', default=False,
        help='Evaluate Model')
    _args = _argparser.parse_args()
    print(_args)
    num_frame = _args.timestep
    model_type = _args.modeltype
    cv = _args.cv
    lr = _args.lr
    wd = _args.wd
    test_flag = _args.evaluate
    num_cycle = _args.cycle
    base_epoch = _args.base
    epoch_mult = _args.mult

    os.makedirs(model_list[model_type] + '-models//', exist_ok=True)
    os.makedirs('temp//', exist_ok=True)

    data_set(num_frame, channel)
    model_set(num_frame, channel)
#    test_flag =True
    if test_flag == True:
        for val_index in range(0,17):
            test_models(num_frame, val_index, 14, model_type, wd, lr, num_cycle)
    else:
        data, label, batch_size = get_data(model_type)
        batch_size = batch_size//(num_frame)
        train_data = [] + data[0]
        train_data = train_data + data[1]

        val_data = [] + data[2]

        train_label = [] + label[0]
        train_label = train_label + label[1]

        val_label = [] + label[2]
        num_classes = len(np.unique(train_label))
        train_data = np.array(train_data)
        val_data = np.array(val_data)
        model_train_eval(num_frame, cv,
                            num_classes, model_type,
                             train_data, train_label,
                             val_data, val_label,
                             batch_size, wd, lr, base_epoch, epoch_mult, num_cycle)

