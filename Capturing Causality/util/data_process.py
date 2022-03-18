import cv2
import numpy as np
from glob import glob
from util.activity_frame import activity_frame_split
from scipy import stats
from PIL import Image

num_channel = 2
num_point = 17
num_frame = 16


def data_set(value, channel):
    """Set number of time-step and number of channels"""
    global num_frame
    num_frame = value

    global num_channel
    num_channel = channel

def normalize(points):
    """Center and Normalize points based on index=1 which is the center of the palm
        num_channel==3 for world coordinates, already normalized
        num_channel==2 for 2d image coordinates, division by 640/480 in x/y dimensions
    """

    for i in range(0, len(points)):
        points[i, :, 0] /= 480.
        points[i, :, 1] /= 640.
        points[i, :, 0] = points[i, :, 0] - points[i, 1, 0]
        points[i, :, 1] = points[i, :, 1] - points[i, 1, 1]

    return np.array(points)


def preprocess(points):
    """Preprocess points"""
    world_points = normalize(points)

    return world_points

def create_sequence_index(data, label, index, data_store, skip):
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


def get_key_points():
    """Obtain the critical frames from the sequence, based on labeled data by Smedt et al."""

    keyPath='E:/ken/database/DHG/DHG_Skele/informations_troncage_sequences.txt'
    keyPoints=np.genfromtxt(keyPath,delimiter=" ")
    #The line format is as follows: #gesture - #finger - #subject - #essai - # frame of the the effective beginning of the gesture - # frame of the the effective end of the gesture.

    kpt=np.zeros((14,2,20,5,2))

    for kp in range(0,len(keyPoints)):
    	for g in range(1,15):
    		if keyPoints[kp][0]==g:
    			for f in range(1,3):
    				if keyPoints[kp][1]==f:
    					for s in range (1,21):
    						if keyPoints[kp][2]==s:
    							for e in range (1, 6):
    								if keyPoints[kp][3]==e:
    									kpt[g-1][f-1][s-1][e-1][0]=keyPoints[kp][4]
    									kpt[g-1][f-1][s-1][e-1][1]=keyPoints[kp][5]
    return kpt

def get_data_skele_json(path_string, end_index=-1, skip=1):
    import json

    ss = glob(path_string)
    data = []
    label = []
    for path_index in range(0, len(ss)):
        if path_index == end_index:
            break
        path = ss[path_index]

        index = int(path[path.find('Activity') + 8:path.find('Trial')-1])-1

        with open(path) as f:
            print(path)
            raw_skel_data = json.load(f)
            temp_points = []
            for i in range(0, len(raw_skel_data)):
                points = np.array(raw_skel_data[i]['poseKeypoints'])

                if points.size == 1:
                    temp_points.append(np.zeros((num_point,num_channel)))
                else:
#                    if i == 0:
#                        skele_index = []
#
#                        diff = np.ones((len(points)))
#                        for p_index in range(0, len(points)):
#                            p = points[p_index][1,:2]
#                            diff[p_index] = sum(np.abs(p - [0.5, 0.5]))/2
#
#                        t = np.argsort(diff)
#                        for p_index in range(0, len(t)):
#                            skele_index.append([t[p_index], points[t[p_index]][1,:2]])
#                    else:
#                        same_person_flag = np.zeros((len(skele_index), len(points)))
#                        for store_skele in range(0, len(skele_index)):
#                            skele_point = skele_index[store_skele][1]
#
#                            for p_index in range(0, len(points)):
#                                test_point = points[p_index][1,:2]
#                                dif = sum(np.abs(test_point-skele_point))/2
#
#                                if dif <= 0.1:
#                                    same_person_flag[store_skele][p_index] = 1
#                                    skele_index[store_skele][1]=test_point
#                                    skele_index[store_skele][0]=p_index
#
#                            if sum(same_person_flag[store_skele][:]) == 0:
#                                skele_index[store_skele][0]=-1
#
#                        sumy = sum(same_person_flag,0)
#                        for person_index in range(0, len(sumy)):
#                            if sumy[person_index] == 0:
#                                skele_index.append([person_index, points[person_index][1,:2]])

                    process_points = preprocess(points)
#
#                    change_index = skele_index[0][0]
#                    if change_index == -1:
#                        continue
                    points = np.reshape(process_points[0][:, 0:num_channel],
                                        (num_point,num_channel))
                    temp_points.append(points)

            create_sequence_index(data, label, index, temp_points, skip)

    return data, label

def get_data_skele_npy(path_string, end_index=-1, skip=1, binary=True):
    ss = glob(path_string)
    data = []
    label = []
    for path_index in range(0, len(ss)):
        if path_index == end_index:
            break
        path = ss[path_index]

        index = int(path[path.find('Activity') + 8:path.find('Trial')-1])-1
        if binary==True:
            if index >= 5:
                index = 0
            else:
                index = 1
        raw_skel_data = np.load(path,allow_pickle=True)
        temp_points = []
        for i in range(0, len(raw_skel_data)):
            points = np.array(raw_skel_data[i])

            if len(points) == 0:
                temp_points.append(np.zeros((num_point,num_channel)))
            else:
                process_points = preprocess(points)

                points = np.reshape(process_points[0][:, 0:num_channel],
                                    (num_point,num_channel))
                temp_points.append(points)

        create_sequence_index(data, label, index, temp_points, skip)

    return data, label

def get_data_img_npy(path_string, end_index=-1, skip=1, binary=True):
    ss = glob(path_string)
    data = []
    label = []
    for path_index in range(0, len(ss)):
        if path_index == end_index:
            break
        path = ss[path_index]

        index = int(path[path.find('Activity') + 8:path.find('Trial')-1])-1
        if binary==True:
            if index >= 5:
                index = 0
            else:
                index = 1

        raw_img_data = np.load(path,allow_pickle=True)
        temp_img = []
        for i in range(0, len(raw_img_data)):
            img = np.array(raw_img_data[i])
#            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.size == 1:
                temp_img.append(np.zeros((48,64,3)))
            else:
                img = img / 255.

                img = np.reshape(img, (48,64,3))
                temp_img.append(img)

        create_sequence_index(data, label, index, temp_img, skip)

    return data, label

def get_data_skele_txt(path_string, end_index=-1, skip=1, verbose=0, use_key_point=True):
    """Creation of the data by loading from text file"""
    import time

    ss = glob(path_string)
    data = []
    label = []

    if use_key_point == True:
        key_point = get_key_points()
    for path_index in range(0, len(ss)):
        if path_index == end_index:
            break
        path = ss[path_index]

        gesture_index = int(path[path.find('gesture_') + len('gesture_'):path.find('finger_')-1])-1
        finger_index = int(path[path.find('finger_') + len('finger_'):path.find('subject_')-1])-1
        subject_index = int(path[path.find('subject_') + len('subject_'):path.find('essai_')-1])-1
        essai_index = int(path[path.find('essai_') + len('essai_'):path.find('essai_') + len('essai_')+1])-1

        if verbose == 1:
            st = time.time()
            print('[' + str(path_index) + '] ' + path)

        raw_skel_data=[];
        raw_skel_data=np.genfromtxt(path,delimiter=" ")
        if use_key_point == True:
            s_index = np.int(key_point[gesture_index][finger_index][subject_index][essai_index][0])
            e_index = np.int(key_point[gesture_index][finger_index][subject_index][essai_index][1])
        else:
            s_index = 0
            e_index = len(raw_skel_data)
        temp_points = []
        for i in range(s_index, e_index):
            points = np.array(raw_skel_data[i])
            points = np.reshape(points, (num_point, num_channel))

            if points.size == 1:
                temp_points.append(np.zeros((num_point,num_channel)))
            else:
                process_points = preprocess(points)
                temp_points.append(process_points)

        create_sequence_index(data, label, gesture_index, temp_points, skip)

        if verbose == 1:
            en = time.time()
            print(str(en-st))
    return data, label
