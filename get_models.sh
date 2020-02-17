# Body Hands and Face Models
# Downloading body pose (COCO and MPI), face and hand models
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
POSE_FOLDER="web_app/pose/"
FACE_FOLDER="web_app/face/"
HAND_FOLDER="web_app/hand/"

# Pose Models
# Body (COCO)
COCO_FOLDER=${POSE_FOLDER}"coco/"
COCO_MODEL=${COCO_FOLDER}"pose_iter_440000.caffemodel"
wget -c ${OPENPOSE_URL}${COCO_MODEL} -P ${COCO_FOLDER}

