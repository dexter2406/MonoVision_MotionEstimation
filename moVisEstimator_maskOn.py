from __future__ import print_function
import sys
import time
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from random import randint
from useFunc.detectAndTrack import *
from useFunc.utils import *
from useFunc.featMatch import *
from useFunc.featMatch_test import *
from utils.maskObj import maskROI

if __name__ == '__main__':

    # testing params
    # - frame interval to estimate distance/  # to calc the relVel
    intv_dist = intv_relVel = 5
    # - focal length, Camera height
    foc_len, H_cam = 1200, 0.8
    thresh = 1

    # settings for read & write video
    prePath = r'C:\ProgamData\global_dataset\img_vid'
    vidName = r'\vid1_2'
    fmt = '.mp4'
    cap = cv.VideoCapture(prePath + vidName + fmt)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps_vid = cap.get(cv.CAP_PROP_FPS)
    sizeW, sizeH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # size = (sizeW*2, int(sizeH*crop))
    size = (sizeW, sizeH)
    c_pnt = (int(sizeW / 2), int(sizeH / 2))
    camMat = np.array([[foc_len, 0, c_pnt[0]],
                       [0, foc_len/1.6, c_pnt[1]],
                       [0, 0, 1]])

    # init video writer
    write_name = 'output\\' + vidName +'maskOn.avi'
    vidWrite = cv.VideoWriter(write_name, fourcc, fps_vid, size)

    # bbox color settings
    # colors = [(255, 255, 255)]
    colors = []
    for i in range(10):
        # colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        colors.append((255, 255, 255))

    # Read first frame, quit if unable to read the video file
    success, _ = cap.read()
    if not success:
        print('Failed to read video')
        sys.exit(1)

    # Process video and track objects
    numFr = 0   # global count
    frmCnt = 0  # local count
    flag_fail = 1       # tracker failure report
    flag_relVel = 0     # for relative velocity calc
    relVel = 0          # init relative velocity
    dist0 = 0           # distances of previous frame for relVel calc
    while cap.isOpened():

        # see if it's the end
        t1 = time.time()
        success, frame = cap.read()
        if not success:
            print("Done")
            break

        frameCopy = np.copy(frame)

        # YOLO detection, re-init under conditions:
        # 0. the 1st-frame initialization
        # 1. after consecutive M frames
        # 2. tracking fails
        if (frmCnt == intv_dist or flag_fail == 1) or (numFr == 0):
            ret_yolo, boxes_yolo = yolov3_det(net, frameCopy)

            # (re)-init tracker only for valid yolo boxes
            if ret_yolo:
                # BUG: if no obj in 1st frame, it fails
                # Ego-motion estimation, independent of detection
                if numFr == 0:  # init
                    frame0 = np.copy(frameCopy)
                    angs = np.array([0, 0, 0])
                    pitch = 0  # orig pose
                else:
                    MaskedFrame = maskROI(frameCopy, boxes_yolo)
                    angs = feat_match(frame0, MaskedFrame, numFr, camMat=camMat, crop=1,
                                      foc_len=foc_len, match_pnts=20, thresh=thresh)
                    frame0 = np.copy(frameCopy)  # stored for next round
                    pitch += angs[0]

                frameOut, dist = calc_distance(boxes_yolo, pitch, frameCopy, c_pnt,
                                               colors, foc_len, H_cam)

                multiTracker = initTrackObj(boxes_yolo, frameCopy)
                # distSet = np.zeros((5, len(boxes_yolo), 2))
                flag_fail = frmCnt = flag_relVel = 0
                meanSet = []
                print("frame:%d" % numFr)
            else:
                frameOut = np.copy(frameCopy)
                print("YOLO failed, skip the frame")

        # After yolo re-init, do normal tracking
        elif ret_yolo:
            # get updated location of objects in subsequent frames
            ret_track, box_tmp = multiTracker.update(frameCopy)

            if ret_track:
                boxes_track = box_tmp
                frameOut, dist = calc_distance(boxes_track, pitch, frameCopy, c_pnt,
                                               colors, foc_len, H_cam)
                # process the distance for relative velocity calc
                meanSet, dist0 = calc_relVel(dist0, dist, meanSet, frmCnt,
                                             flag_fail, fps_vid, intv_relVel)
                frmCnt += 1

            else:
                flag_fail = 1  # flag for re-init
                # if tracker failed, output the original frame
                temp = multiTracker.getObjects()
                frameOut = np.copy(frameCopy)
                print("failed to update frame %d" % numFr)

            if frmCnt == intv_relVel or ret_track is None:
                # output relVel, by calc the mean of the local relVel of 5 frames
                meanSet, _ = calc_relVel(dist0, dist, meanSet, frmCnt,
                                         flag_fail, fps_vid, intv_relVel)
                draw_relVel(boxes_track, meanSet, frameOut, colors)
                meanSet = []

        # counter udpate
        numFr += 1
        t = time.time() - t1

        # print info
        if t > 0:
            print_info(frameOut, t, numFr, pitch, angs[0])

        # plt.imshow(cv.cvtColor(frameOut, cv.COLOR_BGR2RGB)), plt.show()
        # show frame & write

        cv.imshow('MultiTracker', frameOut)
        vidWrite.write(frameOut)

        # quit on ESC button
        if cv.waitKey(1) & 0xFF == 27:
            break
