from util.model import model
from util.data_process import data_process
import numpy as np
from keras import backend as K
import argparse
from sklearn.model_selection import train_test_split
from read import *

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
        '--type', type=int, default=0, metavar='INTEGER',
        help='model')
    _argparser.add_argument(
        '--cycle', type=int, default=1, metavar='INTEGER',
        help='model')
    _args = _argparser.parse_args()

    path= "database//WESAD//"

    num_frame= _args.timestep
    skip = _args.step
    cv = _args.cv
    type_model = _args.type
    num_cycle = _args.cycle

    data_model = data_process(path, int(num_frame/4),skip=5)
    data, label, subject = data_model.get_data()
    for cv in range(0, 15):
        for pos in range(0,2):
            for sense in range(0, len(data[0][pos])):
                os.makedirs('save' + str(num_frame) + '//models' + str(type_model) + '//', exist_ok=True)

                train_data = []
                val_data = []
                train_label = []
                val_label = []

                for index in range(subject):
                    if index == cv:
                        val_data += data[index][pos][sense]
                        val_label += label[index][pos][sense]
                    else:
                        train_data += data[index][pos][sense]
                        train_label += label[index][pos][sense]

                t_model = model(num_classes=len(np.unique(train_label)), model_type=(type_model,0),
                                wd=0, lr=1e-4, num_frame=num_frame, feature_size=(len(train_data[0][0]),))

                save_file = t_model.train(np.array(train_data), list(train_label), cv, pos, sense,
                                                np.array(val_data), val_label, 32, base_epoch=100,
                                                epoch_mult=1, num_cycle=num_cycle)

                t_model.load(save_file)
                pred_label = t_model.predict(np.array(val_data))
                sv = 'save'+ str(num_frame) + '//fr' + str(num_frame) + 'cv' + str(cv)+ 'model' + str(type_model) + fusion_key[pos] + sensor_key[pos][sense]
                np.save(sv, pred_label)

                temp = np.argmax(pred_label,1)
                print(np.sum(temp==val_label)/len(val_label))
                temp = temp == val_label
                f = open('acc.txt', 'a')
                f.write("TS: %3.0f, cv:%3d, model: %2d, BatchSize: %3d, Chest/wrist: %s, Sensor: %s, Accuracy: %.2f%%\n"
                        % (num_frame,cv, type_model, 32, fusion_key[pos], sensor_key[pos][sense], 100*np.sum(temp)/len(temp)))
                f.close()

                del t_model
                K.clear_session()

        for fusion in range(0,3):
                train_data = []
                val_data = []
                train_label = []
                val_label = []

                for index in range(subject):
                    if index == cv:
                        if fusion == 0:
                            t_data = np.zeros((len(data[index][fusion][0]),240,8))
                            for mode in range(0, len(data[index][fusion])):
                                if mode==0:
                                    t_data[:,:,0:3] = np.array(data[index][fusion][mode])
                                else:
                                    t_data[:,:,mode+2:mode+3] = np.array(data[index][fusion][mode])
                        elif fusion == 1:
                            t_data = np.zeros((len(data[index][fusion][0]),240,6))
                            for mode in range(0, len(data[index][fusion])):
                                if mode==0:
                                    t_data[:,:,0:3] = np.array(data[index][fusion][mode])
                                else:
                                    t_data[:,:,mode+2:mode+3] = np.array(data[index][fusion][mode])
                        else:
                            t_data = np.zeros((len(data[index][0][0]),240,8+6))
                            for mode in range(0, len(data[index][0])):
                                if mode==0:
                                    t_data[:,:,0:3] = np.array(data[index][0][mode])
                                else:
                                    t_data[:,:,mode+2:mode+3] = np.array(data[index][0][mode])

                            for mode in range(0, len(data[index][1])):
                                if mode==0:
                                    t_data[:,:,8:11] = np.array(data[index][1][mode])
                                else:
                                    t_data[:,:,mode+10:mode+11] = np.array(data[index][1][mode])

                        val_data += list(t_data)
                        val_label += label[index][0][0]
                    else:
                        if fusion == 0:
                            t_data = np.zeros((len(data[index][fusion][0]),240,8))
                            for mode in range(0, len(data[index][fusion])):
                                if mode==0:
                                    t_data[:,:,0:3] = np.array(data[index][fusion][mode])
                                else:
                                    t_data[:,:,mode+2:mode+3] = np.array(data[index][fusion][mode])
                        elif fusion == 1:
                            t_data = np.zeros((len(data[index][fusion][0]),240,6))
                            for mode in range(0, len(data[index][fusion])):
                                if mode==0:
                                    t_data[:,:,0:3] = np.array(data[index][fusion][mode])
                                else:
                                    t_data[:,:,mode+2:mode+3] = np.array(data[index][fusion][mode])
                        else:
                            t_data = np.zeros((len(data[index][0][0]),240,8+6))
                            for mode in range(0, len(data[index][0])):
                                if mode==0:
                                    t_data[:,:,0:3] = np.array(data[index][0][mode])
                                else:
                                    t_data[:,:,mode+2:mode+3] = np.array(data[index][0][mode])

                            for mode in range(0, len(data[index][1])):
                                if mode==0:
                                    t_data[:,:,8:11] = np.array(data[index][1][mode])
                                else:
                                    t_data[:,:,mode+10:mode+111] = np.array(data[index][1][mode])

                        train_data += list(t_data)
                        train_label += label[index][1][0]

                t_model = model(num_classes=len(np.unique(train_label)), model_type=(type_model,0),
                                wd=0, lr=1e-4, num_frame=num_frame, feature_size=(len(train_data[0][0]),))

                save_file = t_model.train(np.array(train_data), list(train_label), cv, -1, fusion,
                                                np.array(val_data), val_label, 32, base_epoch=100,
                                                epoch_mult=1, num_cycle=num_cycle)

                t_model.load(save_file)
                pred_label = t_model.predict(np.array(val_data))
                sv = 'save'+ str(num_frame) + '//fr' + str(num_frame) + 'cv' + str(cv)+ 'model' + str(type_model) + 'fusion' + sensor_key[2][fusion]
                np.save(sv, pred_label)

                temp = np.argmax(pred_label,1)
                print(np.sum(temp==val_label)/len(val_label))
                temp = temp == val_label
                f = open('acc.txt', 'a')
                f.write("TS: %3.0f, cv:%3d, model: %2d, BatchSize: %3d, Fusion: %s, Accuracy: %.2f%%\n"
                        % (num_frame,cv, type_model, 32, sensor_key[2][fusion], 100*np.sum(temp)/len(temp)))
                f.close()

                del t_model
                K.clear_session()