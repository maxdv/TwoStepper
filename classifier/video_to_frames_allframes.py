''' This script will intake a video file, and split it into a series of individual frames,
    save it in a local directory.'''

import cv2
import numpy as np
import os
import youtube_dl

video_file_path = '/Users/max/Documents/Insight/twostepper-master/classifier/video_file/Country 2 Step Basic Under Arm Turn. The Mans Steps-qxqu7p8oK5g__.mp4'
cap = cv2.VideoCapture(video_file_path)

# create a directory for the images
try:
    if not os.path.exists('frames_tmp'):
        os.makedirs('frames_tmp')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Saves image of the current frame in jpg file
    name = './frames_tmp/frame' + str(currentFrame) + '.jpg'
    #print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
