
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


#yolofd = YOLO(get_args())

path = r'database/tufts/TD_RGB_E_Set[1-4]/*'
from glob import glob
from PIL import Image
import os
typ = 'rgb_e'
#ss = glob(path)
#for dir_index in range(0, len(ss)):
#    dir_path = ss[dir_index]
#    if dir_path[-1].isnumeric() is False:
#        continue
#
#    subject = dir_path[dir_path.rfind('\\')+1:]
#    os.makedirs('data//' + typ + '//' + subject, exist_ok=True)
#
#    img_path = dir_path + '/*'
#    img_list = glob(img_path)
#
#    for img_index in range(0, len(img_list)):
#        imgp = img_list[img_index]
#        sv_path = 'data//' + typ + '//' + subject + '//' + imgp[imgp.rfind('\\')+1:]
#        img = cv2.imread(imgp)
#        pil_image = Image.fromarray(img.copy())
#        im, bb = yolofd.detect_image(pil_image)
#        for b in bb:
#            crop = img[int(b[0]):int(b[2]), int(b[1]):int(b[3])]
#        cv2.imwrite(sv_path,crop)
#        print(imgp)

import numpy as np
img_path = r'face_multi_modal\data\\' + typ + '\*\*'

for test_set in range(1, 6):
    for cv in range (1, 6):
        ss = glob(img_path)
        test_data=[]
        train_data=[]
        test_label=[]
        train_label=[]
        val_data=[]
        val_label=[]
        subject_list=[]

        for index in range(0,len(ss)):
            img = ss[index]
            subject = int(img[img.rfind('\\' + typ)+len('\\' + typ)+1:img.rfind('\\')])
            if subject not in subject_list:
                subject_list.append(subject)
        #    i = int(img[img.rfind('A_')+2:img.rfind('_')])
            i = int(img[img.rfind('_')+1:img.rfind('.')])
            im = cv2.imread(img)
            im = cv2.resize(im, (224,224))
            im = np.float64(im)/255.0

            if test_set == i:
                test_data.append(im)
                test_label.append(subject_list.index(subject))
                if test_set==cv:
                    val_data.append(im)
                    val_label.append(subject_list.index(subject))
            elif cv == i:
                val_data.append(im)
                val_label.append(subject_list.index(subject))
            else:
                train_data.append(im)
                train_label.append(subject_list.index(subject))



        from keras.utils.np_utils import to_categorical
        from keras.layers import LSTM, Dense, Activation, Dropout, Flatten, GlobalAveragePooling2D
        from keras.layers.convolutional import Conv2D, MaxPooling2D
        from keras import backend as K
        from keras.models import Sequential, Model, load_model
        from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
        from keras.optimizers import adam, SGD
        from keras_vggface import VGGFace

        num_classes = len(np.unique(train_label))
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        val_data = np.array(val_data)

        train_label = to_categorical(train_label, num_classes=None)
        test_label = to_categorical(test_label, num_classes=None)
        val_label = to_categorical(val_label, num_classes=None)

        vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(num_classes, activation='softmax', name='classifier')(x)
        model = Model(vgg_model.input, out)


        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
        #model.compile( loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        filepath= typ + "-" + str(cv) + ".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='max')
        early=EarlyStopping(monitor='val_acc', patience=10,verbose=0,mode='auto')
        callbacks_list=[]

        model.fit(train_data, train_label,batch_size=8, verbose=1,
                  validation_data=(val_data, val_label),
                  callbacks=callbacks_list, epochs=10)

#        model.load_weights(filepath)
        pred=model.predict(test_data)

        np.save('result//pred-' + test_set + '-' + str(cv), pred)
        np.save('result//truth-' + test_set + '-' + str(cv), test_label)

        p = np.argmax(pred,1)
        t = np.argmax(test_label,1)
        print(np.sum(p==t)/len(t))

        f=open('acc.txt','a')
        f.write("test: %d, val: %d, Acc: %f\n" % (test_set, cv, np.sum(p==t)/len(t)))
        f.close()

        del model
        K.clear_session()
