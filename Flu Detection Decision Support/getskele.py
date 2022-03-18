import numpy as np
from glob import glob
import cv2
from keras import backend as K

def get_data(model, m):

    path = r'database\biisc\videos\*'
    sv_path = r'database\biisc\\'
    list_video = glob(path)

    for i in range(0, len(list_video)):
        vid_path = list_video[i]
        temp = vid_path
        pose = temp[temp.rfind('_')+1: temp.rfind('.')]
        temp = temp[:temp.rfind('_')]

        if len(pose) ==2:
            flip = 1
            pose = temp[temp.rfind('_')+1:]
            temp = temp[:temp.rfind('_')]
        else:
            flip = 0

        loco = temp[temp.rfind('_')+1:]
        temp = temp[:temp.rfind('_')]

        action = temp[temp.rfind('_')+1:]
        temp = temp[:temp.rfind('_')]

        gender = temp[temp.rfind('_')+1:]
        temp = temp[:temp.rfind('_')]

        subject = temp[temp.rfind('\\')+2:]

        print(i, pose, loco, action, gender, subject, flip)

        ss = sv_path + 'img\\' + str(subject) \
            + '\\' + str(action) + '\\' + str(pose) \
            + '\\' + str(loco) + '\\' + str(flip) + '\\'

        cap = cv2.VideoCapture(vid_path)

        data=[]
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==False:
                break
            sz=m.layers[0].output_shape[1]
            if sz==None:
                sz=224
            frame = cv2.resize(frame, (sz,sz))
            data.append(frame)

        cap.release()
        features = m.predict(np.array(data))
        ss =  sv_path + 'img\\' + str(subject) \
            + '\\' + str(action) + '\\' + str(pose) \
            + '\\' + str(loco) + '\\' + str(flip) + '\\' + model
        np.save(ss,features)


    root_path = r'database\biisc\img'
    path = root_path +'\*'
    num_subject = len(glob(path))
    subject = num_subject
    data = [[] for i in range(num_subject)]
    label = [[] for i in range(num_subject)]

    path = root_path + '\*\*\*\*\*\\' + model +'.npy'
    list_file = glob(path)

    for index in range(0, len(list_file)):
        current_file = list_file[index]
        temp = current_file[current_file.find('img')+4:current_file.find('img')+4+3]
        subject = int(temp)-1
        temp = current_file[current_file.find('img')+8:current_file.find('img')+8+4]
        gesture = temp

        file_data = np.load(current_file)

        data[subject].append(file_data)
        label[subject].append(gesture)

    svpath = '.\data\\data_' + model
    np.save(svpath, data)
    svpath = '.\data\\label_' + model
    np.save(svpath, label)
    del m
    K.clear_session()

def get_datat(model, m):
    root_path = r'database\biisc\img'
    path = r'database\biisc\img\*\*\*\*\*'
    list_file = glob(path)

    for index in range(0, len(list_file)):
        current_file = list_file[index]
        print(current_file)
        temp = current_file[current_file.find('img')+4:current_file.find('img')+4+3]
        subject = int(temp)-1

        temp = current_file[current_file.find('img')+8:current_file.find('img')+8+4]
        gesture = temp

        image_dir = glob(current_file + '\*.png')
        file_data = []
        for image_path in image_dir:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224,224))

            file_data.append(img)
        result = m.predict(np.array(file_data))
        sv_path = current_file + '\\' + model
        np.save(sv_path, result)


    path = root_path +'\*'
    num_subject = len(glob(path))
    subject = num_subject
    data = [[] for i in range(num_subject)]
    label = [[] for i in range(num_subject)]

    path = root_path + '\*\*\*\*\*\\' + model +'.npy'
    list_file = glob(path)

    for index in range(0, len(list_file)):
        current_file = list_file[index]
        temp = current_file[current_file.find('img')+4:current_file.find('img')+4+3]
        subject = int(temp)-1
        temp = current_file[current_file.find('img')+8:current_file.find('img')+8+4]
        gesture = temp

        file_data = np.load(current_file)

        data[subject].append(file_data)
        label[subject].append(gesture)

    svpath = '.\data\\data_' + model
    np.save(svpath, data)
    svpath = '.\data\\label_' + model
    np.save(svpath, data)
    del m
    K.clear_session()

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge


from keras.models import Model
from keras.layers import GlobalAveragePooling2D


model_name = {0:"VGG16",
              1:"VGG19",
              2:"ResNet50",
              3:"InceptionV3",
              4:"InceptionResNetV2",
              5:"Xception",
              6:"NASNetLarge",
              7:"DenseNet201"}
#
#
#base_model = VGG16(weights='imagenet', include_top=False)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#model = Model(inputs=base_model.input, outputs=x)
#get_data('VGG16', model)
#
#base_model = VGG19(weights='imagenet', include_top=False)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#model = Model(inputs=base_model.input, outputs=x)
#get_data('VGG19', model)
#
#base_model = ResNet50(weights='imagenet', include_top=False)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#model = Model(inputs=base_model.input, outputs=x)
#get_data('ResNet50', model)
#
#base_model = InceptionV3(weights='imagenet', include_top=False)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#model = Model(inputs=base_model.input, outputs=x)
#get_data('InceptionV3', model)
#
#base_model = InceptionResNetV2(weights='imagenet', include_top=False)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#model = Model(inputs=base_model.input, outputs=x)
#get_data('InceptionResNetV2', model)
#
#base_model = Xception(weights='imagenet', include_top=False)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#model = Model(inputs=base_model.input, outputs=x)
#get_data('Xception', model)
#
#base_model = NASNetLarge(weights='imagenet', include_top=False)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#model = Model(inputs=base_model.input, outputs=x)
#get_data('NASNetLarge', model)

#base_model = DenseNet201(weights='imagenet', include_top=False)
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#model = Model(inputs=base_model.input, outputs=x)
#get_data('DenseNet201', model)
