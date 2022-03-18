from util.model import model
from util.data_process import data_process
import numpy as np
from keras import backend as K
import argparse
from sklearn.model_selection import train_test_split
from read import *
import time

sensor_key = []
sensor_key.append( ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp'])
sensor_key.append(['ACC', 'BVP', 'EDA', 'TEMP'])
sensor_key.append(['Chest', 'Wrist', 'Chest_Wrist'])
fusion_key = ['Chest', 'Wrist','fusion']


import os
if __name__ == "__main__":
    _argparser = argparse.ArgumentParser(
            description='Stress Recognition',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--timestep', type=int, default=240, metavar='INTEGER',
        help='Time step in network')
    _argparser.add_argument(
        '--step', type=int, default=1, metavar='INTEGER',
        help='Time step in network')
    _argparser.add_argument(
        '--cv', type=int, default=0, metavar='INTEGER',
        help='Cross-Validation Set')
    _argparser.add_argument(
        '--type', type=int, default=2, metavar='INTEGER',
        help='model')
    _argparser.add_argument(
        '--cycle', type=int, default=1, metavar='INTEGER',
        help='model')
    _argparser.add_argument(
        '--bs', type=int, default=256, metavar='INTEGER',
        help='model')
    _argparser.add_argument(
        '--epoch', type=int, default=100, metavar='INTEGER',
        help='model')
    _argparser.add_argument(
        '--ident', type=int, default=0, metavar='INTEGER',
        help='model')
    _args = _argparser.parse_args()

    path= "database//WESAD//"
    num_frame= _args.timestep
    step = _args.step
    cv = _args.cv
    type_model = _args.type
    num_cycle = _args.cycle
    epoch = _args.epoch
    bs = _args.bs
    ident = _args.ident==1

    data_model = data_process(path, int(num_frame/4),step, binary=False)
    data, label, subject = data_model.get_data()

    from os import path
    if path.exists("save_personal//rnd.npy"):
        cv_list = (np.load("save_personal//rnd.npy",allow_pickle=True))
    else:
        import copy
        cv_list = copy.deepcopy(label)
        for index in range(subject):
            num_data = len(data[index][0][0])
            cv_numbers = np.random.randint(10, size=num_data)
            for pos in range(0,2):
                for sense in range(0, len(data[0][pos])):
                    cv_list[index][pos][sense]=cv_numbers
        np.save("save_personal//rnd.npy",cv_list)

    for fusion_num in range(0,6):
        if fusion_num <3 :
            fusion = fusion_num
        else:
            fusion = fusion_num-3

        wd = 0.0001
        lr = 1e-5
        os.makedirs('save_personal//save' + str(num_frame) + 'step' + str(step) + '//models' + str(type_model) + '//', exist_ok=True)
        train_data = []
        val_data = []
        train_label = []
        val_label = []
        val_id=[]
        train_id=[]
        for index in range(subject):
            if fusion == 0:
                t_data = np.zeros((len(data[index][fusion][0]),240,8))

            for pos in range(0, 2):
                for sense in range(0, len(data[index][pos])):
                    for t in range(0, len(cv_list[index][pos][sense])):
                        if cv_list[index][pos][sense][t]==cv:
                            val_data.append(data[index][pos][sense][t])
                            val_label.append(label[index][pos][sense][t])
                            val_id.append(index)
                        else:
                            train_data.append(data[index][pos][sense][t])
                            train_label.append(label[index][pos][sense][t])
                            train_id.append(index)

        if ident == True:
            train_label= train_id
            val_label = val_id

        if fusion_num < 3:
            t_model = model(num_classes=len(np.unique(train_label)), model_type=(type_model,0),
                    wd=wd, lr=lr, num_frame=num_frame, feature_size=(len(train_data[0][0]),),independent=0)
            fusion_type = 'SL'
        else:
            t_model = model(num_classes=len(np.unique(train_label)), model_type=(type_model,0),
                            wd=wd, lr=lr, num_frame=num_frame, feature_size=(len(train_data[0][0]),),independent=(fusion_num-2))
            fusion_type = 'FL'

        filepath = 'save_personal//save' + str(num_frame)+ 'step' + str(step) + '//models' + str(type_model) + '//' + str(0)  + '-' + fusion_key[pos] + '-' + sensor_key[pos][sense] + '-cv' + str(cv) + '.hdf5'
        time_a = time.perf_counter()
        save_file = t_model.train(np.array(train_data), list(train_label), cv, filepath, pos, sense,
                                        np.array(val_data), val_label, bs, base_epoch=epoch,
                                        epoch_mult=1, num_cycle=num_cycle)
        time_b = time.perf_counter()

        train_time = time_b-time_a

        t_model.load(save_file)
        time_a = time.perf_counter()
        pred_label = t_model.predict(np.array(val_data))
        time_b = time.perf_counter()

        test_time = time_b-time_a
        sv = 'save_personal//save'+ str(num_frame)+ 'step' + str(step) + '//fr' + str(num_frame) + 'cv' + str(cv)+ 'model' + str(type_model) + fusion_key[pos] + sensor_key[pos][sense]
        np.save(sv, pred_label)

        temp = np.argmax(pred_label,1)
        print(np.sum(temp==val_label)/len(val_label))
        temp = temp == val_label
        f = open('save_personal_acc.txt', 'a')
        f.write("Train: %5.5f, Test: %5.5f, TS: %3.0f, cv:%3d, model: %2d, BatchSize: %3d, Chest/wrist: %s, Sensor: %s, Accuracy: %.2f%%\n"
                % (train_time, test_time, num_frame,cv, type_model, bs, fusion_key[pos], sensor_key[pos][sense], 100*np.sum(temp)/len(temp)))
        f.close()

        del t_model
        K.clear_session()
