import numpy as np
from glob import glob
import pickle
from tqdm import tqdm

class data_process(object):
    def __init__(self, root_path):
        keys = ['label', 'subject', 'signal']
        signal_keys = ['wrist', 'chest']

        subs = root_path[root_path.rfind('//'):]
        path = root_path + '//S' + subs + '.pkl'
        
        with open(path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')

        signal = data[keys[2]]
        wrist_data = signal[signal_keys[0]]
        chest_data = signal[signal_keys[1]]

        chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        labels = data[keys[0]]
        wrist_fs = [32, 64, 4, 4]
        self.wrist_data = []
        self.chest_data = []
        
        prog = tqdm(range(0, len(chest_sensor_keys)))
        for modes in prog:
        #for modes in range(0, len(chest_sensor_keys)):
            fs = int(700/4)
            temp_data = chest_data[chest_sensor_keys[modes]]
            resamp = [sum(temp_data[t:t+fs]) for t in range(0, len(temp_data),fs)]
            resamp = np.array(resamp)/fs
            resamp_labels= [np.rint(np.mean((labels[t:t+fs]))) for t in range(0, len(labels),fs)]
            
            self.chest_data.append(resamp)
            
        prog = tqdm(range(0, len(wrist_sensor_keys)))
        for modes in prog:           
        #for modes in range(0, len(wrist_sensor_keys)):
            fs = int(wrist_fs[modes]/4)
            temp_data = wrist_data[wrist_sensor_keys[modes]]
            resamp = [sum(temp_data[t:t+fs]) for t in range(0, len(temp_data),fs)]
            resamp = np.array(resamp)/fs
            self.wrist_data.append(resamp)
            
            
            
        self.labels = resamp_labels


    def get_data(self):
        return self.wrist_data, self.chest_data, self.labels

