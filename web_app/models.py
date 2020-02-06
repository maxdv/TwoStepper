# This script takes the user's input movie, and processes it
# video --> frames --> Human Pose Estimation --> csv --> [CNN] --> Dance Move ID
import cv2
import numpy as np
import pandas as pd
import os
from natsort import natsorted
from keras.models import load_model
from keras.models import Sequential
from keras.layers import  *
from keras.optimizers import *
from keras.models import model_from_json
from keras import backend as K


def listdir_nohidden(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def video_to_frames(videoFile_path,frameFile_path):

    videoFile = str(listdir_nohidden(videoFile_path)[0])
    vidcap = cv2.VideoCapture(videoFile_path + '/' + videoFile)
    success,image = vidcap.read()
    est_tot_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)  # Sets an upper bound # of frames in video clip
    desired_frames = np.round(np.arange(0,est_tot_frames,est_tot_frames/30)) #split the video into 30 frames
    
    for i in desired_frames:
        vidcap.set(1,i-1)                      
        success,image = vidcap.read(1)         # image is an array of array of [R,G,B] values
        frameId = vidcap.get(1)                # The 0th frame is often a throw-away
        cv2.imwrite(frameFile_path + '/frame%d.jpg' % frameId, image)
    vidcap.release()
    cv2.destroyAllWindows()
    
def predict(skeletons_path, parent_directory):
    skeleton_name = listdir_nohidden(skeletons_path)[0]
    skeleton_string = skeletons_path + '/' + skeleton_name
    user_input_skeleton_raw = np.genfromtxt(skeleton_string, delimiter=',')
    user_input_skeleton_transformed = 1-np.absolute(user_input_skeleton_raw.reshape(1,36,60,1))
#     input_shape = (36, 60, 1)
    
#     model = Sequential()
#     model.add(Conv2D(filters=60, kernel_size=[36,4],strides=[36,2],padding='same', activation='relu'))
#     model.add(MaxPool2D(pool_size=4, padding='same'))

#     model.add(Conv2D(filters=60, kernel_size=[18,4],strides=[18,2],padding='same', activation='relu'))
#     model.add(MaxPool2D(pool_size=4, padding='same'))

#     model.add(Conv2D(filters=60, kernel_size=[9,4],strides=[9,2],padding='same', activation='relu'))
#     model.add(MaxPool2D(pool_size=4, padding='same'))

#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(512,activation='relu'))
#     model.add(Dropout(rate=0.5))
#     model.add(Dense(3,activation='relu'))

# #     optimizer = Adam(lr=1e-3)

    
#     model.load_weights('model_weights.h5')



#     model.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['accuracy'])

#     pred = model.predict(img_ts)
    
#     index_predict = np.argmax(pred[0])
    
#     # summarize model.
#     model_summ = model.summary()
    
#     #if probabilities are spread out and there's no clear winner, return "unsure"
#     if pred[0][index_predict] <= 0.33:
#         return "unsure"
    K.clear_session()

    model = Sequential()

    model.add(InputLayer(input_shape = [36,60,1]))

    model.add(Conv2D(filters=60, kernel_size=[36,4],strides=[36,2],padding='same', activation='softmax'))
    model.add(MaxPool2D(pool_size=4, padding='same'))

    model.add(Conv2D(filters=60, kernel_size=[18,4],strides=[18,2],padding='same', activation='softmax'))
    model.add(MaxPool2D(pool_size=4, padding='same'))

    model.add(Conv2D(filters=60, kernel_size=[9,4],strides=[9,2],padding='same', activation='softmax'))
    model.add(MaxPool2D(pool_size=4, padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512,activation='softmax'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(3,activation='softmax'))

    model.load_weights('model_weights.h5')
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy',metrics=['accuracy'])

    
    model_out = model.predict(user_input_skeleton_transformed)
    
    index_predict = np.argmax(model_out[0])
    dict_labels = ['turn', 'cuddle', 'shadow']
    
#     if pred[0][index_predict] <= 0.33:
#         return "unsure"
#     else:
#         return dict_labels[index_predict]
#     if np.argmax(model_out) == 0:
#         str_label='Turn'
#     elif np.argmax(model_out) == 1:
#         str_label='Cuddle'
#     elif np.argmax(model_out) == 2:
#         str_label='Shadow'
    # plt.imshow(data)
    
    K.clear_session()

    return dict_labels[index_predict]

    

    
    
    
#     json_file = open('model.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights("model.h5")


