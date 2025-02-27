import cv2
import numpy as np
from display import Display
from pointmap import Map, Point

### Camera intrinsics
# Define principal point offset or optical center coordinates
W, H = 1920 // 2, 1080 // 2

# Define focus length
F = 270

# Define Intrinsic Matrix and inverse of that
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
Kinv = np.linalg.inv(K)

# Image display initialization
display = Display(W, H)

# Initialize a map
mapp = Map()
mapp.create_viewer()


if __name__=="__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            #process_frame(frame)
        else:
            break