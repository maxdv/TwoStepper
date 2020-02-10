from flask import render_template
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory

from twostepper_flask_app import app

from werkzeug.utils import secure_filename

from PIL import Image
from flask import render_template, send_from_directory
from flask import Flask, flash, request, redirect, url_for


import numpy as np
import os
from models import *


parent_directory = os.getcwd()
videoFile_path = parent_directory + '/input_video/input_video_clip'
frameFile_path = parent_directory + '/input_video/input_video_frames'
skeletons_path = parent_directory + '/input_video/input_skeletons'
protoFile = parent_directory + '/pose/coco/pose_deploy_linevec.prototxt'
weightsFile = parent_directory + '/pose/coco/pose_iter_440000.caffemodel'

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



              
def allowed_file(filename):
    """ function used to ensure file is in expected format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file_size(movie_clip):
    """ function used to ensure file is not too big"""
    if os.stat.st_size(movie_clip):
        return ' Your file size is too big! Try timming the clip to < 1 MB'


@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
 
    parent_directory = os.getcwd()
    videoFile_path = parent_directory + '/input_video/input_video_clip'
    frameFile_path = parent_directory + '/input_video/input_video_frames'
    skeletons_path = parent_directory + '/input_video/input_skeletons'
    protoFile = parent_directory + '/pose/coco/pose_deploy_linevec.prototxt'
    weightsFile = parent_directory + '/pose/coco/pose_iter_440000.caffemodel'
    
    try:   
        if not os.path.exists(videoFile_path):
            os.makedirs(videoFile_path)  
        if not os.path.exists(frameFile_path):
            os.makedirs(frameFile_path)
        if not os.path.exists(skeletons_path):
            os.makedirs(skeletons_path)
    except OSError:
        print ('Error: Creating directory of data')   

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


    

            user_frames = video_to_frames(videoFile_path,frameFile_path)

            user_hpe = frames_to_hpe(frameFile_path, skeletons_path, protoFile, weightsFile)

            predicted_class = predict(skeletons_path, parent_directory)

        return render_template('landing_page.html', predicted_class = predicted_class)
        
    if __name__ == "__main__":
        app.run()