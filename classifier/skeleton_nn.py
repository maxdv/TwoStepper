# This script takes the outputs from frame_processor.py and trains a CNN on the arrays of the HPE. The model outputs are saved in this directory, and in the web_app diretory.

import os
from frame_processer_functions import *

# identify the working directories
classifier_directory = os.getcwd()
parent_directory = classifier_directory[:-11] #exclude '/classifier'
webapp_directory = parent_directory + '/web_app'

train_data = classifier_directory + '/train'
test_data = classifier_directory + '/test'
validate_data = classifier_directory + '/validate'

# Train a model to interpret the skeletons
skeletons_to_nn(train_data, test_data, validate_data, webapp_directory, classifier_directory)