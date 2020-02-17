# This script generates a database of youtube video clips of dance moves identified in the twostepper_input.csv table. The videos are downloaded and split into clips that have dance moves in them. To maximize the impact of the data, each clip is augmented by making it slightly early or late, or extending the duration of the clip. These clips are then split into a series of frames, which are fed into a Human Pose Estimation model called OpenPose (https://www.learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/). Each directory of frames saves the keypoints into a csv, which is the training data for skeleton_nn.py.

import os
from frame_processer_functions import *

# Identify the main directories of this app
classifier_directory = os.getcwd()
parent_directory = classifier_directory[:-11] #exclude '/classifier'
webapp_directory = parent_directory + '/web_app'

# Main directories that will be passed between functions
dance_move_database_path = classifier_directory + '/twostepper_input.csv'
pathway_to_clips = classifier_directory + '/movie_clips'
pathway_to_frames = classifier_directory + '/frames'
pathway_to_skeletons = classifier_directory + '/skeletons'


# You really only want to download this one time; it's a large, expensive model. Because of the vagaries of Flask, I've opted to download this into the web_app folder from the shell script.
protoFile = webapp_directory + '/pose/coco/pose_deploy_linevec.prototxt'
weightsFile = webapp_directory + '/pose/coco/pose_iter_440000.caffemodel'


# Download the videos listed in the .csv, split them into clips, and augment the data with the moving time-window.
videos_from_database(dance_move_database_path, classifier_directory)

# Take the clips and turn each into a directory of frames.
clips_to_frames(pathway_to_clips, pathway_to_frames)

# Identify all the parts of the body based on the frames.
frames_to_hpe(pathway_to_frames, pathway_to_skeletons, protoFile, weightsFile)
