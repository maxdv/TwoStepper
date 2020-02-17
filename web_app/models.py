# This script takes the user's input movie, and processes it
# video --> frames --> Human Pose Estimation --> csv --> [CNN] --> Dance Move ID
import cv2
import numpy as np
import os
from natsort import natsorted
from keras.models import load_model
from keras.models import Sequential
from keras.layers import  *
from keras.optimizers import *
from keras.models import model_from_json
from keras import backend as K

# Generates a list of files in a directory excluding the hidden ones
def listdir_nohidden(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

# Imports a video file and saves it as a series of images
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
    
# Takes a time series array and predicts a related dance move
def predict(skeletons_path, parent_directory):
    skeleton_name = listdir_nohidden(skeletons_path)[0]
    skeleton_string = skeletons_path + '/' + skeleton_name
    user_input_skeleton_raw = np.genfromtxt(skeleton_string, delimiter=',')
    user_input_skeleton_transformed = 1-np.absolute(user_input_skeleton_raw.reshape(1,36,60,1))
    
    K.clear_session()
    
    model = Sequential()

    model.add(InputLayer(input_shape = [36,60,1]))

    model.add(Conv2D(filters=60, kernel_size=[36,4],strides=[36,2],padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=4, padding='same'))

    model.add(Conv2D(filters=60, kernel_size=[18,4],strides=[18,2],padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=4, padding='same'))

    model.add(Conv2D(filters=60, kernel_size=[9,4],strides=[9,2],padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=4, padding='same'))


    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(3,activation='relu'))

    model.load_weights('model_weights.h5')
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy',metrics=['accuracy'])
    
    model_out = model.predict(user_input_skeleton_transformed)
    
    index_predict = np.argmax(model_out[0])
    dict_labels = ['turn', 'cuddle', 'shadow']
    
    if model_out[0][index_predict] <= 0.05: # if the predictions are bad
        move_name_str = "unsure"
        return move_name_str
    else:
        move_name_str = dict_labels[index_predict]
        return move_name_str 
    
    K.clear_session()

    return move_name_str

    
##### THE NUTS AND BOLTS OPENPOSE/HUMAN POSE ESTIMATION MODEL ####

# Find the Keypoints using Non Maximum Suppression on the Confidence Map
def getKeypoints(probMap, threshold=0.1):
    
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    
    #find the blobs
    _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

# Find valid connections between the different joints of a all persons present
def getValidPairs(output, mapIdx, frameWidth, frameHeight, detected_keypoints, POSE_PAIRS):
    
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB 
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid
        
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)
                    
                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:            
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    print(valid_pairs)
    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
# It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, mapIdx, frameWidth, frameHeight, detected_keypoints, POSE_PAIRS, keypoints_list):
    # the last number in each row is the overall score 
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])): 
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score 
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


def frames_to_hpe(pathway_to_frames, pathway_to_skeletons, protoFile, weightsFile):

    frame_tot = 30

    try:
        if not os.path.exists(pathway_to_skeletons):
            os.makedirs(pathway_to_skeletons)
    except OSError:
        print ('Error: Creating directory of data')

    clip_names = listdir_nohidden(pathway_to_frames)


    nPoints = 18
    # COCO Output Format
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                        'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                        'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

    POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                  [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                  [1,0], [0,14], [14,16], [0,15], [15,17],
                  [2,17], [5,16] ]

    # # index of pafs correspoding to the POSE_PAIRS
    # # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
              [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
              [47,48], [49,50], [53,54], [51,52], [55,56], 
              [37,38], [45,46]]

    
    for kk in range(len(clip_names)):
        
        time_series_clip = -1 * np.ones((2*nPoints, 2*frame_tot))
        frame_names = natsorted(listdir_nohidden(pathway_to_frames))
        outputexists = os.path.isfile(pathway_to_skeletons + '/User_input___skeleton.csv')
        
        if int(outputexists) == 0:

            for tt in range(len(frame_names)): 
                fileinfo=os.stat(pathway_to_frames + '/' + frame_names[tt])

                if fileinfo.st_size > 0: # Omits 0 byte images

                    image1 = cv2.imread(pathway_to_frames + '/' + '/' + frame_names[tt])
                    frameWidth = image1.shape[1]
                    frameHeight = image1.shape[0]

                    # Load the network and pass the image through the network
                    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

                    # Fix the input Height and get the width according to the Aspect Ratio
                    inHeight = 368
                    inWidth = int((inHeight/frameHeight)*frameWidth)

                    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                              (0, 0, 0), swapRB=False, crop=False)

                    net.setInput(inpBlob)
                    output = net.forward()

                    i = 0
                    probMap = output[0, i, :, :]
                    probMap = cv2.resize(probMap, (frameWidth, frameHeight))


                    detected_keypoints = []
                    keypoints_list = np.zeros((0,3))
                    keypoint_id = 0
                    threshold = 0.1

                    for part in range(nPoints):
                        probMap = output[0,part,:,:]
                        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
                        keypoints = getKeypoints(probMap, threshold)
                        keypoints_with_id = []
                        for i in range(len(keypoints)):
                            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                            keypoint_id += 1

                        detected_keypoints.append(keypoints_with_id)

                    valid_pairs, invalid_pairs = getValidPairs(output, mapIdx, frameWidth, frameHeight, detected_keypoints, POSE_PAIRS)

                    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, mapIdx, frameWidth, frameHeight, detected_keypoints, POSE_PAIRS, keypoints_list)
                    saver = -1 * np.ones((nPoints*2,2)) # for two humans in x,y coodinates

                    for i in range(17):
                        if len(personwiseKeypoints) <= 2:

                            for n in range(len(personwiseKeypoints)):
                                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                if -1 in index:
                                    continue
                                B = np.int32(keypoints_list[index.astype(int), 0])
                                A = np.int32(keypoints_list[index.astype(int), 1])

                                saver[i+(n*17), 0] = B[0] / frameWidth
                                saver[i+(n*17), 1] = A[0] / frameHeight
                        else:
                            for n in range(0,2): #picks the two most prominent people in the frame
                                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                if -1 in index:
                                    continue
                                B = np.int32(keypoints_list[index.astype(int), 0])
                                A = np.int32(keypoints_list[index.astype(int), 1])

                                saver[i+(n*17), 0] = B[0] / frameWidth
                                saver[i+(n*17), 1] = A[0] / frameHeight
                    time_series_clip[:,2*tt:2*tt+2] = saver
                else: 
                    time_series_clip[:,2*tt:2*tt+2] = time_series_clip[:,2*tt:2*tt+2]
            np.savetxt(pathway_to_skeletons + '/User_input___skeleton.csv', time_series_clip, delimiter=',')
        
        elif outputexists == 1: # true or 1, then skip alll this
            time_series_clip = time_series_clip


