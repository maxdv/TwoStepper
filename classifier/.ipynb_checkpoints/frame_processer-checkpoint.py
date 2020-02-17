# import pandas as pd
# import time
# from datetime import datetime
# import cv2
# import numpy as np
import os
# import youtube_dl
from frame_processer_functions import *


classifier_directory = os.getcwd()
parent_directory = classifier_directory[:-11] #exclude '/classifier'

dance_move_database_path = classifier_directory + '/twostepper_input_tester.csv'
pathway_to_clips = classifier_directory + '/movie_clips'
pathway_to_frames = classifier_directory + '/frames'
pathway_to_skeletons = classifier_directory + '/skeletons'

webapp_directory = parent_directory + '/web_app'

# You really only want to download this one time
protoFile = webapp_directory + '/pose/coco/pose_deploy_linevec.prototxt'
weightsFile = webapp_directory + '/pose/coco/pose_iter_440000.caffemodel'


# Turn the database of youtube videos into frames to analyze
videos_from_database(dance_move_database_path, classifier_directory)
clips_to_frames(pathway_to_clips, pathway_to_frames)

# Identify all the parts of the body based on the frames
frames_to_hpe(pathway_to_frames, pathway_to_skeletons, protoFile, weightsFile)
