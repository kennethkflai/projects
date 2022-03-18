import numpy as np
import cv2
from glob import glob
num_points = 17
class data_process(object):
    def __init__(self, root_path, num_frame, skeleton=True, target=0):

        path = root_path +'\*'
        num_subject = len(glob(path))
        self.subject = num_subject
        self.data = [[] for i in range(num_subject)]
        self.label = [[] for i in range(num_subject)]

        if skeleton == True:
            path = root_path + '\*\*\*\*\*\*.npy'
            list_file = glob(path)

            for index in range(0, len(list_file)):
                current_file = list_file[index]
                temp = current_file[current_file.find('img')+4:current_file.find('img')+4+3]
                subject = int(temp)-1
                temp = current_file[current_file.find('img')+8:current_file.find('img')+8+4]
                gesture = temp

                file_data = np.load(current_file)
                file_data = file_data[:,:,0:2]

                file_data = cv2.resize(file_data,(num_points, num_frame))

#                file_data[:,:,0] = (file_data[:,:,0]-145)/290
#                file_data[:,:,1] = (file_data[:,:,1]-240)/480

                for i in range(len(file_data)):
                    file_data[i,:,0] = (file_data[i,:,0]-file_data[i,target,0])/290
                    file_data[i,:,1] = (file_data[i,:,1]-file_data[i,target,0])/480
                self.data[subject].append(file_data)
                self.label[subject].append(gesture)

        else:
            path = root_path + '\*\*\*\*\*'
            list_file = glob(path)

            for index in range(0, len(list_file)):
                current_file = list_file[index]
                temp = current_file[current_file.find('img')+4:current_file.find('img')+4+3]
                subject = int(temp)-1

                temp = current_file[current_file.find('img')+8:current_file.find('img')+8+4]
                gesture = temp

                image_dir = glob(current_file + '\*.png')
                file_data = []
                for image_path in image_dir:
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (100,200))
                    file_data.append(img)

                self.data[subject].append(file_data)
                self.label[subject].append(gesture)

        print(num_subject)

    def get_data(self):
        return self.data, self.label, self.subject
