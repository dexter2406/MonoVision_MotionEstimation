from __future__ import print_function
import sys
import time
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from useFunc.detectAndTrack_keras import *
from useFunc.utils_kerasYolo import *
from useFunc.featMatch import *
from useFunc.yolov3_mod import YOLO
from PIL import Image
from useFunc.utils_kerasYolo import calc_distance_keras

if __name__ == '__main__':

    # testing params
    intv = 5    # frame interval to estimate distance
    # init yolo class
    yolo = YOLO()

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
    foc_len = 1100
    # init video writer
    write_name = 'output\\' + vidName +'_rv_'+'.avi'
    vidWrite = cv.VideoWriter(write_name, fourcc, fps_vid, size)
    font = cv.FONT_HERSHEY_SIMPLEX

    # bbox color settings
    colors = [(255, 255, 255)]
    # for i in range(15):
    #     colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

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

        # Eigen-motion estimation, independent of detection
        if numFr == 0:  # init
            frame0 = np.copy(frameCopy)
            angs = np.array([0, 0, 0])
            pitch = 0  # orig pose
        elif numFr % intv == 0:  # angle calc
            angs = feat_match(frame0, frameCopy, numFr, size, crop=1)
            frame0 = np.copy(frameCopy)  # stored for next round
            pitch += angs[0]

        # print("pitch:%.2f" % pitch)

        # YOLO detection, re-init under conditions:
        # 0. the 1st-frame initialization
        # 1. after consecutive M frames
        # 2. tracking fails
        if (frmCnt == 5 or flag_fail == 1) or (numFr == 0):
            # keras-yolo detect
            framePIL = cv.cvtColor(frameCopy, cv.COLOR_BGR2RGB)
            framePIL = Image.fromarray(np.uint8(framePIL))
            ret_yolo, boxes_yolo = yolo.detect_image(framePIL)

            # (re)-init tracker only for valid yolo boxes
            if ret_yolo:
                # frameOut, dist = calc_distance(boxes_yolo, pitch, frameCopy, c_pnt)
                frameOut, dist = calc_distance_keras(boxes_yolo, pitch, frameCopy, c_pnt)
                multiTracker = initTrackObj_keras(boxes_yolo, frameCopy)
                # distSet = np.zeros((5, len(boxes_yolo), 2))
                flag_fail = 0
                frmCnt = 0
                flag_relVel = 0
                meanSet = []
                print("frame:%d" % numFr)

            # in case yolo fails
            else:
                frameOut = np.copy(frameCopy)
                print("YOLO failed, skip the frame")

        # After yolo re-init, do normal tracking
        elif ret_yolo:
            # get updated location of objects in subsequent frames
            ret_track, box_tmp = multiTracker.update(frameCopy)

            if ret_track:
                boxes_track = box_tmp
                frameOut, dist = calc_distance_keras(boxes_track, pitch, frameCopy, c_pnt)
                # process the distance for relative velocity calc
                meanSet, dist0 = calc_relVel(dist0, dist, meanSet, frmCnt, flag_fail, fps_vid)
                frmCnt += 1

            else:
                flag_fail = 1  # flag for re-init
                # if tracker failed, output the original frame
                temp = multiTracker.getObjects()
                frameOut = np.copy(frameCopy)
                print("failed to update frame %d" % numFr)

            if frmCnt == 5 or (ret_track is None):
                # output relVel, by calc the mean of the local relVel of 5 frames
                meanSet, _ = calc_relVel(dist0, dist, meanSet, frmCnt, flag_fail, fps_vid)
                draw_relVel_keras(boxes_track, meanSet, frameOut)

        numFr += 1
        t = time.time() - t1

        if t > 0:
            fps = "FPS: %.1f" % (1 / t)
            print(fps)
            numInfo = "Frame: %d" % numFr
            if angs[0] > 0:
                pitchInfo = "Pitch: %.2f(down)" % pitch
            elif angs[0] < 0:
                pitchInfo = "Pitch: %.2f(up)" % pitch
            else:
                pitchInfo = "Pitch: %.2f" % pitch

            cv.putText(frameOut, fps, (50, 50), font, 0.7, (230, 230, 230))
            cv.putText(frameOut, numInfo, (50, 80), font, 0.7, (230, 230, 230))
            cv.putText(frameOut, pitchInfo, (50, 110), font, 0.7, (230, 230, 230))
        # #
        # plt.imshow(cv.cvtColor(frameOut, cv.COLOR_BGR2RGB)), plt.show()
        # a=1
        # show frame & write

        cv.imshow('MultiTracker', frameOut)
        vidWrite.write(frameOut)

        # quit on ESC button
        if cv.waitKey(1) & 0xFF == 27:
            break
