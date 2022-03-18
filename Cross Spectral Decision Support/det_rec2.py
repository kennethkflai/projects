
import cv2
from keras.models import load_model
import argparse

from yolo.yolo import YOLO, detect_video, detect_img


#####################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model-weights/YOLO_Face.h5',
                        help='path to model weights file')
    parser.add_argument('--anchors', type=str, default='cfg/yolo_anchors.txt',
                        help='path to anchor definitions')
    parser.add_argument('--classes', type=str, default='cfg/face_classes.txt',
                        help='path to class definitions')
    parser.add_argument('--score', type=float, default=0.5,
                        help='the score threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='the iou threshold')
    parser.add_argument('--img-size', type=list, action='store',
                        default=(416, 416), help='input image size')
    parser.add_argument('--image', default=False, action="store_true",
                        help='image detection mode')
    parser.add_argument('--video', type=str, default='samples/subway.mp4',
                        help='path to the video')
    parser.add_argument('--output', type=str, default='outputs/',
                        help='image/video output path')
    args = parser.parse_args()
    return args


yolofd = YOLO(get_args())

from glob import glob
from PIL import Image
import os

def face_det(path, typ):
    ss = glob(path)
    for dir_index in range(0, len(ss)):
        dir_path = ss[dir_index]
        if dir_path[-1].isnumeric() is False:
            continue

        subject = dir_path[dir_path.rfind('\\')+1:]
        os.makedirs('data//' + typ + '//' + subject, exist_ok=True)

        img_path = dir_path + '/*'
        img_list = glob(img_path)

        for img_index in range(0, len(img_list)):
            imgp = img_list[img_index]

            sv_path = 'data//' + typ + '//' + subject + '//' + imgp[imgp.rfind('\\')+1:]
            if typ != 'cs':
                img = cv2.imread(imgp)
                pil_image = Image.fromarray(img.copy())
                im, bb = yolofd.detect_image(pil_image)
                for b in bb:
                    crop = img[int(b[0]):int(b[2]), int(b[1]):int(b[3])]
                    break
                cv2.imwrite(sv_path,crop)
                print(imgp)
            else:
                import shutil
                if (imgp.find('.fac') ==-1) and (imgp.find('RGB_E')==-1):
                    shutil.copyfile(imgp, sv_path)

import numpy as np

def get_data(ss):
    data=[]
    label=[]
    subject_list=[]
    for index in range(0,len(ss)):
        img = ss[index]
        temp = img[:img.rfind('\\')]
        subject = int(temp[temp.rfind('\\')+1:])

        if subject not in subject_list:
            subject_list.append(subject)

        im = cv2.imread(img)
        im = cv2.resize(im, (224,224))
        im = np.float64(im)/255.0

        data.append(im)
        label.append(subject_list.index(subject))

    return data, label

from keras.utils.np_utils import to_categorical
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import adam, SGD
from keras_vggface import VGGFace

types = ['rgb', 'nir', 'ir', 'cs']

for train_type in types:
    img_path = r'face_multi_modal\data\\' + train_type + '*\*\*'
    ss = glob(img_path)
    train_data, train_label = get_data(ss)
    train_data = np.array(train_data)
    num_classes = len(np.unique(train_label))
    train_label = to_categorical(train_label, num_classes=None)

    vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(num_classes, activation='softmax', name='classifier')(x)
    model = Model(vgg_model.input, out)

    filepath= train_type + ".hdf5"

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_data, train_label,batch_size=8, verbose=1,
              epochs=10)
    model.save_weights(filepath)
    for test_type in types:
        img_path = r'face_multi_modal\data\\' + test_type + '*\*\*'
        ss = glob(img_path)
        test_data, test_label = get_data(ss)

        test_data = np.array(test_data)
        test_label = to_categorical(test_label, num_classes=None)

        model.load_weights(filepath)
        pred=model.predict(test_data)
        np.save('result//pred-' + test_type + '-' + train_type, pred)
        np.save('result//truth-' + test_type + '-' + train_type, test_label)
        p = np.argmax(pred,1)
        t = np.argmax(test_label,1)
        print(np.sum(p==t)/len(t))

        f=open('acc.txt','a')
        f.write("test: %s, train: %s, Acc: %f\n" % (test_type, train_type, np.sum(p==t)/len(t)))
        f.close()

    del model
    K.clear_session()
