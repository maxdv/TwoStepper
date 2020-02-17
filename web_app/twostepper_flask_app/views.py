from flask import render_template
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory

from twostepper_flask_app import app

from werkzeug.utils import secure_filename

from PIL import Image
from flask import render_template, send_from_directory
from flask import Flask, flash, request, redirect, url_for


import numpy as np
import os
import shutil
from models import *

# Identify working directories
parent_directory = os.getcwd()
videoFile_path = parent_directory + '/input_video/input_video_clip'
frameFile_path = parent_directory + '/input_video/input_video_frames'
skeletons_path = parent_directory + '/input_video/input_skeletons'

# Model files
protoFile = parent_directory + '/pose/coco/pose_deploy_linevec.prototxt'
weightsFile = parent_directory + '/pose/coco/pose_iter_440000.caffemodel'

# Remove existing video, frame, and skeleton
shutil.rmtree(videoFile_path)
shutil.rmtree(frameFile_path)
shutil.rmtree(skeletons_path)

# Create new directories for user inputs 
try:   
    if not os.path.exists(videoFile_path):
        os.makedirs(videoFile_path)  
    if not os.path.exists(frameFile_path):
        os.makedirs(frameFile_path)
    if not os.path.exists(skeletons_path):
        os.makedirs(skeletons_path)
except OSError:
    print ('Error: Creating directory of data')
        
UPLOAD_FOLDER = videoFile_path
ALLOWED_EXTENSIONS = {'mov', 'mp4', 'avi', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Ensure the file is in the correct format              
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload page for user video
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Landing page for user
@app.route('/', methods = ['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        file_secure = secure_filename(file.filename)
        
        if file_secure == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        
        if not allowed_file(file_secure):
            flash('Wrong File Format! Please use .mov, .mp4, or .avi')
            return redirect(request.url)
        
        if file and allowed_file(file_secure):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Turn the video into frames
            user_frames = video_to_frames(videoFile_path,frameFile_path)
            # Input frames into HPE
            user_hpe = frames_to_hpe(frameFile_path, skeletons_path, protoFile, weightsFile)
            # Predict the dance move based on pretrained model
            predicted_class = predict(skeletons_path, parent_directory)

        return render_template('landing_page.html', predicted_class = predicted_class)
        
    if __name__ == "__main__":
        app.run()