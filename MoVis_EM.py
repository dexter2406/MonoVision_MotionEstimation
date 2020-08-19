from __future__ import print_function
import sys
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from useFunc.detectAndTrack import *
from useFunc.utils import *
from useFunc.featMatch import *

if __name__ == '__main__':

    # Params
    intv_EM = 4  # interval to implement
    # - focal length, Camera height
    foc_len, H_cam = 1200, 0.8
    thresh = 3  # threshold angle to avoid outliers

    # settings for read & write video
    prePath = r'C:\ProgamData\global_dataset\img_vid'
    vidName = r'\vid1_4'
    fmt = '.mp4'
    cap = cv.VideoCapture(prePath + vidName + fmt)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps_vid = cap.get(cv.CAP_PROP_FPS)
    sizeW, sizeH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # size = (sizeW*2, int(sizeH*crop))
    size = (sizeW, sizeH)
    c_pnt = (int(sizeW / 2), int(sizeH / 2))
    camMat = np.array([[foc_len, 0, c_pnt[0]],
                       [0, foc_len / 1.6, c_pnt[1]],
                       [0, 0, 1]])

    # init video writer
    write_name = 'output\\' + vidName + '_EM.avi'
    vidWrite = cv.VideoWriter(write_name, fourcc, fps_vid, size)

    # Read first frame, quit if unable to read the video file
    success, _ = cap.read()
    if not success:
        print('Failed to read video')
        sys.exit(1)

    # MAIN
    numFr = 0
    while cap.isOpened():

        # see if it's the end
        t1 = time.time()
        success, frame = cap.read()
        if not success:
            print("Done")
            break
        frameCopy = np.copy(frame)

        # Eigen-motion estimation, independent of detection
        if numFr == 0:  # init
            frame0 = np.copy(frameCopy)
            angs = np.array([0, 0, 0])
            pitch = 0  # orig pose
        elif numFr % intv_EM == 0:  # angle calc
            angs = feat_match(frame0, frameCopy, numFr, size, camMat=camMat, crop=1,
                              foc_len=foc_len, match_pnts=20, thresh=thresh)
            frame0 = np.copy(frameCopy)  # stored for next round
            pitch += angs[0]

        # counter udpate
        numFr += 1
        t = time.time() - t1

        # print info
        if t > 0:
            print_info(frameCopy, t, numFr, pitch, angs[0])

        cv.imshow("Ego-motion", frameCopy)
        vidWrite.write(frameCopy)
        if cv.waitKey(1) & 0xFF == 27:
            break
