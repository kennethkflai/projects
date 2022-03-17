import csv
import numpy as np

#def get_csv():
path = r'CompleteDataSet.csv'
path2 = r'CompleteDataSetFeatures1&0.5.csv'
path3 = r'Features_1&0.5_Vision.csv'
def create_sequence_index(data, label, index, data_store, skip, num_frame=20):
    seq = []
#    index_store = []
    for i in range(0, len(data_store)):
        if (i >= num_frame) and (i % skip == 0):
            while(len(seq) > num_frame):
                seq.pop(0)

            data.append(seq.copy())
            label.append(index)

        seq.append(data_store[i])
#        index_store.append(index)

def get_csv(skip, num_frame, binary):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        header = []
        total_features = [[[[] for i in range(3)] for i in range(11)] for i in range(17)]

        for row in csv_reader:
            if line_count == 0:
                line_count +=1
            elif line_count == 1:
                for h in row:
                    header.append(h)
                line_count +=1
            else:
                features = []
                for h in range(1, len(row)-11):
                    if h >=8 and h<= 13:
                        continue
                    if h%7==0:
                        continue
                    features.append(np.float(row[h]))
                features = np.array(features)
                features = np.reshape(features, (4,6))
                subject = np.int(row[-4])-1
                act = np.int(row[-3])-1
                trial = np.int(row[-2])-1
                total_features[subject][act][trial].append(features)

        data = [[] for i in range(3)]
        label = [[] for i in range(3)]
        for sub in range(0, len(total_features)):
            for act in range(0, len(total_features[sub])):
                for trial in range(0, len(total_features[sub][act])):
                    temp = total_features[sub][act][trial]
                    index = act
                    if binary==True:
                        if index >= 5:
                            index = 0
                        else:
                            index = 1

                    create_sequence_index(data[trial], label[trial], index, temp, skip, num_frame)
    return data, label

def get_csv2(skip, num_frame, binary):
    with open(path2) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        header = []
        total_features = [[[[] for i in range(3)] for i in range(11)] for i in range(17)]

        for row in csv_reader:
            if line_count == 0:
                line_count +=1
            elif line_count == 1:
                for h in row:
                    header.append(h)
                line_count +=1
            else:
                features = []
                for h in range(1, len(row)-4-7*18):
                    if header[h].find('illuminance') >=0 or header[h].find('Pocket')>=0 or header[h].find('Infrared')>=0:
                        continue
                    if header[h].find('Brain') >=0:
                        continue
                    features.append(np.float(row[h]))
                features = np.array(features)
                features = np.reshape(features, (18,24))
                subject = np.int(row[-4])-1
                act = np.int(row[-3])-1
                trial = np.int(row[-2])-1
                total_features[subject][act][trial].append(features)

        data = [[] for i in range(3)]
        label = [[] for i in range(3)]
        for sub in range(0, len(total_features)):
            for act in range(0, len(total_features[sub])):
                for trial in range(0, len(total_features[sub][act])):
                    temp = total_features[sub][act][trial]
                    if len(temp)==0:
                        continue
                    temp = np.array(temp)
                    temp = temp[:,np.newaxis,:,:]
                    temp = list(temp)
                    index = act
                    if binary==True:
                        if index >= 5:
                            index = 0
                        else:
                            index = 1
                    indx =[]
                    for ind in range(0, len(temp)):
                        indx.append(index)
                    data[trial] = data[trial] + temp
                    label[trial] = label[trial] + indx
    return data, label

def get_csv3(skip, num_frame, binary):
    with open(path3) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        header = []
        total_features = [[[[] for i in range(3)] for i in range(11)] for i in range(17)]

        for row in csv_reader:
            if line_count == 0:
                line_count +=1
            elif line_count == 1:
                for h in row:
                    header.append(h)
                line_count +=1
            else:
                features = []
                for h in range(1, len(row)-4):
                    features.append(np.float(row[h]))
                features = np.array(features)
                features = np.reshape(features, (40,20))
                subject = np.int(row[-4])-1
                act = np.int(row[-3])-1
                trial = np.int(row[-2])-1
                total_features[subject][act][trial].append(features)

        data = [[] for i in range(3)]
        label = [[] for i in range(3)]
        for sub in range(0, len(total_features)):
            for act in range(0, len(total_features[sub])):
                for trial in range(0, len(total_features[sub][act])):
                    temp = total_features[sub][act][trial]
                    if len(temp)==0:
                        continue
                    temp = np.array(temp)
                    temp = temp[:,np.newaxis,:,:]
                    temp = list(temp)
                    index = act
                    if binary==True:
                        if index >= 5:
                            index = 0
                        else:
                            index = 1
                    indx =[]
                    for ind in range(0, len(temp)):
                        indx.append(index)
                    data[trial] = data[trial] + temp
                    label[trial] = label[trial] + indx
    return data, label
