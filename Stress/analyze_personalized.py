from sklearn.metrics import classification_report
import numpy as np
from util.data_process import data_process
from util.model import model
from keras import backend as K

path = r'stress_affect\personalized\save3-\save240step1\\'
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
for pos in range(2):
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

cv_list = (np.load("personalized//save3-//rnd.npy",allow_pickle=True))

idd=True
result = np.zeros(())
model_name=2
for pos in range(0,2):
    for sense in range(0,len(sensor_key[pos])):
        for cv in range(0, 10):
            if pos == 2 or pos==3:
                model_name = sense
#            name = 'fr240cv' + str(cv) + 'model' + str(model_name) + fusion[pos] + sensor_key[pos][sense] + '.npy'
#
#            data1 = np.load(path + name)
#
#            predicted = np.argmax(data1,1)
            truth_label = []
            train_data = []
            val_data = []
            train_label = []
            val_label = []
            val_id=[]
            train_id=[]
            for index in range(subject):
                for t in range(0, len(cv_list[index][pos][sense])):
                    if cv_list[index][pos][sense][t]==cv:
                        val_data.append(data[index][pos][sense][t])
                        val_label.append(label[index][pos][sense][t])
                        val_id.append(index)
                    else:
                        train_data.append(data[index][pos][sense][t])
                        train_label.append(label[index][pos][sense][t])
                        train_id.append(index)

            if idd==True:
                truth_label = val_id
                target_names = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
            else:
                truth_label = val_label
                target_names = ['Baseline', 'Stress', 'Amusement']

            t_model = model(num_classes=len(np.unique(truth_label)), model_type=(2,0),
                            wd=0, lr=1e-5, num_frame=240, feature_size=(len(train_data[0][0]),),independent=0)
            name = 'models2//0-' +  fusion[pos] + '-' + sensor_key[pos][sense] + '-'+ 'cv' + str(cv) +  '.hdf5'
            t_model.load(path + name)
            pred_label = t_model.predict(np.array(val_data))
            predicted = np.argmax(pred_label,1)


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

            del t_model
            K.clear_session()

        print(fusion[pos], sensor_key[pos][sense] , 'accuracy:',
              np.mean(accuracy[pos][sense]), '+-', np.std(accuracy[pos][sense]),
              np.mean(precision[pos][sense]), '+-', np.std(precision[pos][sense]),
              np.mean(recall[pos][sense]), '+-', np.std(recall[pos][sense]),
              np.mean(f1[pos][sense]), '+-', np.std(f1[pos][sense]),
              np.mean(sensitivity[pos][sense]), '+-', np.std(sensitivity[pos][sense]))
