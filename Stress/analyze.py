from sklearn.metrics import classification_report
import numpy as np
from util.data_process import data_process

path = r'stress_affect\save_binary\save240step1\\'
path = r'.\new\save2\save240step1\/'
sensor_key = []
sensor_key.append( ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp'])
sensor_key.append(['ACC', 'BVP', 'EDA', 'TEMP'])
sensor_key.append(['Chest', 'Wrist', 'Chest_Wrist'])
sensor_key.append(['Chest', 'Wrist', 'Chest_Wrist'])
fusion = ['Chest', 'Wrist','fusionFL-','fusionSL-']

data_model = data_process("database//WESAD//", int(240/4),1,False)
data, label, subject = data_model.get_data()

accuracy =[]
precision =[]
recall =[]
f1 =[]
sensitivity = []
for pos in range(4):
    temp1=[]
    temp2=[]
    temp3=[]
    temp4=[]
    temp5=[]
    for sense in range(len(sensor_key[pos])):
       temp1.append([])
       temp2.append([])
       temp3.append([])
       temp4.append([])
       temp5.append([])
    accuracy.append(temp1)
    precision.append(temp2)
    recall.append(temp3)
    f1.append(temp4)
    sensitivity.append(temp5)

result = np.zeros(())
model=2
for pos in range(0,4):
    for sense in range(0,len(sensor_key[pos])):
        for cv in range(0, 15):
            if pos == 2 or pos==3:
                model = sense
            name = 'fr240cv' + str(cv) + 'model' + str(model) + fusion[pos] + sensor_key[pos][sense] + '.npy'

            data = np.load(path + name)

            predicted = np.argmax(data,1)
            truth_label = []
            for index in range(subject):
                if index == cv:
                    if pos >= 2:
                        truth_label += label[index][0][0]
                    else:
                        truth_label += label[index][pos][sense]

            target_names = ['Baseline', 'Stress', 'Amusement']
#            target_names = ['No Stress', 'Stress']
            a=classification_report(truth_label, predicted, target_names=target_names,output_dict=True)
            accuracy[pos][sense].append(a['accuracy'])
            precision[pos][sense].append(a['weighted avg']['precision'])
            recall[pos][sense].append(a['weighted avg']['recall'])
            f1[pos][sense].append(a['weighted avg']['f1-score'])

            s=[]
            for n in target_names:
                temp_sense = 0
                total = 0
                for k in target_names:
                    if n is not k:
                        temp_sense= temp_sense + a[k]['recall']*a[k]['support']
                        total += a[k]['support']
                s.append(temp_sense/total)


            sensitivity[pos][sense].append(sum(s)/len(s))

        print(fusion[pos], sensor_key[pos][sense] , 'accuracy:',
              np.mean(accuracy[pos][sense]), '+-', np.std(accuracy[pos][sense]),
              np.mean(precision[pos][sense]), '+-', np.std(precision[pos][sense]),
              np.mean(recall[pos][sense]), '+-', np.std(recall[pos][sense]),
              np.mean(f1[pos][sense]), '+-', np.std(f1[pos][sense]),
              np.mean(sensitivity[pos][sense]), '+-', np.std(sensitivity[pos][sense]))
