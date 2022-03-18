import numpy as np
from glob import glob
import pickle

class data_process(object):
    def __init__(self, root_path, num_frame, skip, binary=False):
        keys = ['label', 'subject', 'signal']
        signal_keys = ['wrist', 'chest']
        chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        wrist_fs = [32, 64, 4, 4]
        subs = [2, 3, 4, 5, 6, 7,8,9,10,11,13,14,15,16,17]
        store_data = [[[] for j in range(2)] for i in range(len(subs))]
        store_labels = [[[] for j in range(2)] for i in range(len(subs))]
        for i in range(0,len(subs)):
            print(i)
            path = root_path + 'S' + str(subs[i]) + '//S' + str(subs[i]) + '.pkl'
            with open(path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')

            signal = data[keys[2]]
            wrist_data = signal[signal_keys[0]]
            chest_data = signal[signal_keys[1]]

            labels = data[keys[0]]

            for modes in range(0, len(chest_sensor_keys)):
                fs = int(700/4)
                temp_data = chest_data[chest_sensor_keys[modes]]
                resamp = [sum(temp_data[t:t+fs]) for t in range(0, len(temp_data),fs)]
                resamp = np.array(resamp)/fs
                resamp_labels= [np.rint(np.mean((labels[t:t+fs]))) for t in range(0, len(labels),fs)]

                mode_data = []
                mode_labels = []
                frame_data = list(resamp[0:num_frame*4])
                mode_data.append(frame_data.copy())
                mode_labels.append(resamp_labels[num_frame*4])

                step=0
                for frames in range(num_frame*4, len(resamp)):
                    frame_data.append(list(resamp[frames]))
                    frame_data.pop(0)

                    step +=1
                    if step >= skip*4:
                        step = 0
                        if resamp_labels[frames]<=0 or resamp_labels[frames] >3:
                            continue;
                        t = np.array(frame_data.copy())
                        mode_data.append(t)
                        if binary==True:
                            mode_labels.append(resamp_labels[frames]==2)
                        else:
                            mode_labels.append(resamp_labels[frames]-1)

                store_data[i][0].append(mode_data.copy())
                store_labels[i][0].append(mode_labels.copy())

            for modes in range(0, len(wrist_sensor_keys)):
                fs = int(wrist_fs[modes]/4)
                temp_data = wrist_data[wrist_sensor_keys[modes]]
                resamp = [sum(temp_data[t:t+fs]) for t in range(0, len(temp_data),fs)]
                resamp = np.array(resamp)/fs

                mode_data = []
                mode_labels = []
                frame_data = list(resamp[0:num_frame*4])
                mode_data.append(frame_data.copy())
                mode_labels.append(resamp_labels[num_frame*4])
                step=0
                for frames in range(num_frame*4, len(resamp)):
                    frame_data.append(list(resamp[frames]))
                    frame_data.pop(0)

                    step +=1
                    if step >= skip*4:
                        step = 0
                        if resamp_labels[frames]<=0 or resamp_labels[frames] >3:
                            continue;
                        t = np.array(frame_data.copy())
                        mode_data.append(t)
                        if binary==True:
                            mode_labels.append(resamp_labels[frames]==2)
                        else:
                            mode_labels.append(resamp_labels[frames]-1)

                store_data[i][1].append(mode_data.copy())
                store_labels[i][1].append(mode_labels.copy())

            self.data = store_data
            self.label = store_labels
            self.subject = len(subs)

    def get_data(self):
        return self.data, self.label, self.subject
