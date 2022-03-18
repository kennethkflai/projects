from util.model import model
from util.data_process import data_process
import numpy as np
from keras import backend as K

for target in range(0, 17):
    f = open('acc.txt', 'a')

    f.write("Normalize Base Point: %d"
            % (target))
    f.close()
    for fr in range (1,8):
        batch_size = 32
        num_frame = 2**fr
        root_path = r'E:\ken\database\biisc\img'
        data_model = data_process(root_path, num_frame, mode=0, target=target)

        data, label, subject = data_model.get_data()

        un = np.unique(label)
        un = [str(i) for i in un]
        lbl_dict = {un[i]:i  for i in range(0,len(un))}

        train_data = []
        val_data = []
        train_label = []
        val_label = []

        for index in range(subject):
            if index <6 and index >0:
                val_data += data[index]
                val_label += label[index]
            else:
                train_data += data[index]
                train_label += label[index]


        train_label = [lbl_dict[i] for i in train_label]
        val_label = [lbl_dict[i] for i in val_label]

        num_classes = len(np.unique(train_label))

        mt=0
        t_model = model(num_classes, model_type=mt, wd=0, lr=1e-3, num_frame=num_frame)

        save_file = t_model.train(np.array(train_data), train_label, fr,
                                np.array(val_data), val_label, batch_size, 100)

        t_model.load(save_file)
        pred_label = t_model.predict(np.array(val_data))

        pd =[]
        for i in range(0, len(pred_label)):

            pd.append(np.argmax(pred_label[i], 1))
            print(np.sum(pd[i]==val_label)/len(val_label))
            temp = pd[i] == val_label

            f = open('acc.txt', 'a')
    #        print(np.sum(temp==True), len(temp))
            f.write("TS: %3.0f, model: %2d, BatchSize: %3d, Accuracy: %.2f%%\n"
                    % (num_frame, i, batch_size, 100*np.sum(temp)/len(temp)))
            f.close()

        del t_model
        K.clear_session()
