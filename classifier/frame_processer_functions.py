import pandas as pd
import time
from datetime import datetime
import cv2
import numpy as np
import os
import youtube_dl
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from natsort import natsorted

# !pip install q keras==2.3.1
from keras.models import Sequential
from keras.layers import  *
from keras.optimizers import *
from keras.models import load_model




## Generally useful functions
########################################################

# Turns a hh:mm:ss timestamp into seconds
def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# List the files in a directory without the hidden files
def listdir_nohidden(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


## Process functions
########################################################

# Reads the videos from the dataframe, downloads them, splits them into the identified clips, and augments the data with a moving time window.
def videos_from_database(dance_move_database_path, classifier_directory):
    # Read and identify the unique videos to download.
    df  = pd.read_csv(dance_move_database_path)
    url_unique_vals = df.URL.unique()
    title_unique_vals = df.Title.unique()
	
    # Create the directories needed.
    try:
        if not os.path.exists(classifier_directory + '/youtube_videos'):
            os.makedirs(classifier_directory + '/youtube_videos')
        if not os.path.exists(classifier_directory + '/movie_clips'):
            os.makedirs(classifier_directory + '/movie_clips')
        if not os.path.exists(classifier_directory + '/frames'):
            os.makedirs(classifier_directory + '/frames')        
    except OSError:
        print ('Error: Creating directory of data')

    # Download the youtube videos	
    os.chdir(classifier_directory + '/youtube_videos')

    for kk in range(len(title_unique_vals)):
        url_lookup = url_unique_vals[kk]
        title = title_unique_vals[kk]
        dfsub = df[df['Title'].str.match(title_unique_vals[kk])]
        dfsub = dfsub.reset_index()
        
        # call youtube-dl from the command line
        ytdlstring = "youtube-dl -f 'bestvideo[height<=480, ext=mp4]' " + url_lookup + " --id"   
        os.system(ytdlstring)   

        # Identify each clip of interest, split the video apart, and augment with the moving time window
        for jj in range(len(dfsub)):
            # Convert the start and end times into seconds for ffmpeg
            start_time = get_sec(dfsub.start_time[jj])
            end_time = get_sec(dfsub.end_time[jj])

            # Augmented start and end times
            early_start_time = start_time - (0.5 * (end_time - start_time))
            late_end_time = end_time + (0.5 * (end_time - start_time))
            late_start_time = start_time + (0.25 * (end_time - start_time))
            early_end_time = end_time - (0.25 * (end_time - start_time))
            
            # Names for the input and output videos
            movie_name = dfsub.URL[jj][-11:] + '.mp4'
            movie_out_clip = dfsub.move[jj] + '.' + dfsub.URL[jj][-11:] + '__' + str(start_time) + '__' + str(end_time) + '.mov'
            movie_out_clip_long = dfsub.move[jj] + '.' + dfsub.URL[jj][-11:] + '__' + str(early_start_time) + '__' + str(late_end_time) + '.mov'
            movie_out_clip_early = dfsub.move[jj] + '.' + dfsub.URL[jj][-11:] + '__' + str(early_start_time) + '__' + str(early_end_time) + '.mov'
            movie_out_clip_late = dfsub.move[jj] + '.' + dfsub.URL[jj][-11:] + '__' + str(late_start_time) + '__' + str(late_end_time) + '.mov'

            movie_out_name_and_path = classifier_directory + '/movie_clips/' + movie_out_clip
            movie_out_name_and_path_long = classifier_directory + '/movie_clips/' + movie_out_clip_long 
            movie_out_name_and_path_early = classifier_directory + '/movie_clips/' + movie_out_clip_early 
            movie_out_name_and_path_late = classifier_directory + '/movie_clips/' + movie_out_clip_late
            
            # Strings for command line ffmpeg
            moviesplitterstring = 'ffmpeg -ss ' + str(start_time) + ' -to ' + str(end_time) + ' -i ' + movie_name + ' -c copy ' + movie_out_name_and_path
            moviesplitterstring_early = 'ffmpeg -ss ' + str(early_start_time) + ' -to ' + str(early_end_time) + ' -i ' + movie_name + ' -c copy ' + movie_out_name_and_path_early
            moveisplitterstring_long = 'ffmpeg -ss ' + str(early_start_time) + ' -to ' + str(late_end_time) + ' -i ' + movie_name + ' -c copy ' + movie_out_name_and_path_long
            movesplitterstring_late = 'ffmpeg -ss ' + str(late_start_time) + ' -to ' + str(late_end_time) + ' -i ' + movie_name + ' -c copy ' + movie_out_name_and_path_late

            # ffmpeg command line
            os.system(moviesplitterstring)       
            os.system(moviesplitterstring_early)
            os.system(moveisplitterstring_long)
            os.system(movesplitterstring_late)

# Splits clips in directory into individual images
def clips_to_frames(pathway_to_clips, pathway_to_frames):
    os.chdir(pathway_to_clips)
    clip_names = listdir_nohidden(pathway_to_clips)
    
    # For each clip, split it into a directory of images
    for ii in range(len(clip_names)):
        # Use cv2 to read and split clips into images
        clip_name = clip_names[ii]
        vidcap = cv2.VideoCapture(pathway_to_clips + '/' + clip_names[ii])
        success,image = vidcap.read()

        frame_dir = pathway_to_frames + '/' + clip_name[:-4]
        # make directory for frames
        try:
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
        except OSError:
            print ('Error: Creating directory of data')

        # Sets an upper bound # of frames in video clip
        est_tot_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Split the video, regardless of length into a maximum of 30 frames  
        desired_frames = np.round(np.arange(0,est_tot_frames,est_tot_frames/30)) 
        
        # clip splitter
        for i in desired_frames:
            vidcap.set(1,i-1)                      
            success,image = vidcap.read(1)         # image is an array of array of [R,G,B] values
            frameId = vidcap.get(1)                # The 0th frame is often a throw-away
            cv2.imwrite(frame_dir + '/' + clip_name + 'frame%d.jpg' % frameId, image)
        vidcap.release()
        cv2.destroyAllWindows()
 



##Open Pose functions--for more details on this model, go to:
# https://www.learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/ 
# and
# https://github.com/spmallick/learnopencv/tree/master/OpenPose-Multi-Person
######################################################## 
 
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
# It finds the person and index at which the joint should be added. This can be done since we have an id for each joint.
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

# Passes the directories of images into the human pose estimator.
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
        frame_names = natsorted(listdir_nohidden(pathway_to_frames + '/' + clip_names[kk]))
        outputexists = os.path.isfile(pathway_to_skeletons + '/' + clip_names[kk] + '___skeleton.csv')
        
        if int(outputexists) == 0: # Skip this expensive computation if it already exists

            for tt in range(len(frame_names)): 
                fileinfo=os.stat(pathway_to_frames + '/' + clip_names[kk] + '/' + frame_names[tt])
                if fileinfo.st_size > 0: # Avoids any 0 byte images

                    image1 = cv2.imread(pathway_to_frames + '/' + clip_names[kk] + '/' + frame_names[tt])
                    print(pathway_to_frames + '/' + clip_names[kk] + '/' + frame_names[tt])
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

                    for i in range(17): # for each identified keypoint
                        if len(personwiseKeypoints) <= 2: # the model gets confused with many people in the frame 

                            for n in range(len(personwiseKeypoints)): # for the identified keypoints on each person
                                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                if -1 in index:
                                    continue
                                B = np.int32(keypoints_list[index.astype(int), 0])
                                A = np.int32(keypoints_list[index.astype(int), 1])

                                saver[i+(n*17), 0] = B[0] / frameWidth #save the x and y coordinates of the keypoints in relative frame space.
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
            np.savetxt(pathway_to_skeletons + '/' + clip_names[kk] + '___skeleton.csv', time_series_clip, delimiter=',')
        
        elif outputexists == 1: # true or 1, then skip alll this
            time_series_clip = time_series_clip


# CNN on the dataset
###########################################################

# The moves in the model
def move_label(img): 
    label = img.split('.')[0]
    if label == 'turn':
        ohl = np.array([1,0,0])
    elif label == 'cuddle':
        ohl = np.array([0,1,0])
    elif label == 'shadow':
        ohl = np.array([0,0,1])
    return ohl

# The following transform the arrays from HPE into inputs that the CNN can understand.
def train_data_with_label(train_data):
    train_images = []
    for ii in tqdm(listdir_nohidden(train_data)): # for each training array
        path = os.path.join(train_data, ii)
        img = 1-np.absolute(np.genfromtxt(path, delimiter=',')) # transform array
        train_images.append([np.array(img), move_label(ii)]) # tag with label
    shuffle(train_images) # shuffle the order
    return train_images

def test_data_with_label(test_data):
    test_images = []
    for ii in tqdm(listdir_nohidden(test_data)):
        path = os.path.join(test_data, ii)
        img = 1-np.absolute(np.genfromtxt(path, delimiter=',')) 
        test_images.append([np.array(img), move_label(ii)])
    shuffle(test_images)
    return test_images

def validate_data_with_label(validate_data):
    validate_images = []
    for ii in tqdm(listdir_nohidden(validate_data)):
        path = os.path.join(validate_data, ii)
        img = 1-np.absolute(np.genfromtxt(path, delimiter=',')) 
        validate_images.append([np.array(img), move_label(ii)])
    shuffle(validate_images)
    return validate_images

# create a model to identify dance moves from the input arrays
def skeletons_to_nn(train_data, test_data, validate_data, webapp_directory, classifier_directory):
	# tag and id the distinct data types for the model
    training_images = train_data_with_label(train_data)
    testing_images = test_data_with_label(test_data)
    validating_images = validate_data_with_label(validate_data)
    
    # reshape the data into arrays that the model can digest
    
    tr_img_data = np.array([ii[0] for ii in training_images]).reshape(-1,36,60,1)
    tr_lbl_data = np.array([ii[1] for ii in training_images])

    tst_img_data = np.array([ii[0] for ii in testing_images]).reshape(-1,36,60,1)
    tst_lbl_data = np.array([ii[1] for ii in testing_images])

    val_img_data = np.array([ii[0] for ii in validating_images]).reshape(-1,36,60,1)
    val_lbl_data = np.array([ii[1] for ii in validating_images])
    
    # Model architecture
    
    model = Sequential()

    model.add(InputLayer(input_shape = [36,60,1]))
    
    # Layer that looks at the dancers synoptically, and advances in time
    model.add(Conv2D(filters=60, kernel_size=[36,4],strides=[36,2],padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=4, padding='same'))
    
    # Layer that looks at each dancer, and advances in time
    model.add(Conv2D(filters=60, kernel_size=[18,4],strides=[18,2],padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=4, padding='same'))
    
    # Layer that looks at half of the dancer on an approximate transverse plane, and advances in time
    model.add(Conv2D(filters=60, kernel_size=[9,4],strides=[9,2],padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=4, padding='same'))

    
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(3,activation='relu'))
    optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=tr_img_data,y=tr_lbl_data,epochs=50,batch_size=100)
    
    # Save the weights here, and for the web app    
    model.save(webapp_directory + '/model_.h5')
    model.save(classifier_directory + '/model_.h5')

    model.save_weights(webapp_directory + '/model_weights.h5') 
    model.save_weights(classifier_directory + '/model_weights.h5')

    