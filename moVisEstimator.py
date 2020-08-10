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

if __name__ == '__main__':

    # testing params
    intv = 5

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
    write_name = ('output\\' + vidName + 'intv_%d'+'.avi') % intv
    # vidWrite = cv.VideoWriter(write_name, fourcc, fps_vid, size)

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
    dist0 = 0
    while cap.isOpened():

        # see if it's the end
        t1 = time.time()
        success, frame = cap.read()
        if not success:
            print("Done")
            break

        frameCopy = np.copy(frame)

        # ego-motoin, independent of detection
        if numFr == 0:  # init
            frame0 = np.copy(frameCopy)
            angs = np.array([0, 0, 0])
            pitch = 0  # orig pose
        elif numFr % intv == 0:  # angle calc
            angs = feat_match(frame0, frameCopy, numFr, size, crop=1)
            frame0 = np.copy(frameCopy)  # stored for next round
            pitch += angs[0]

        print("pitch:%.2f" % pitch)
        numFr += 1

        # condition for YOLO re-init:
        # 1. after consecutive M frames
        # 2. tracking fails
        if (frmCnt % 5 == 0 and numFr > 0 and frmCnt > 0) or flag_fail == 1:
            ret_yolo, bboxes = yolov3_det(net, frameCopy, angs)
            frameOut = np.copy(frameCopy)

            # (re)-init tracker only for valid boxes
            if ret_yolo:
                frameOut, dist = calc_distance(bboxes, pitch, frameCopy, c_pnt)
                multiTracker = initTrackObj(bboxes, frameCopy)
                flag_fail = 0
                frmCnt = 0
                flag_relVel = 0
            else:
                print("YOLO failed, skip the frame")

        # otherwise, do normal tracking
        elif ret_yolo:
            # get updated location of objects in subsequent frames
            ret_track, boxes = multiTracker.update(frameCopy)
            # if tracker failed, output the original frame
            if ret_track:
                frameOut, dist = calc_distance(boxes, pitch, frameCopy, c_pnt)
                if frmCnt % 2 == 0:
                    relVel, dist0, flag_relVel = calc_relVel(dist0, dist, relVel, flag_relVel, fps=24)
                if flag_relVel == 0:
                    draw_relVel(boxes, relVel, frameOut)
                frmCnt += 1
            else:
                print("failed to update frame %d" % numFr)
                flag_fail = 1  # flag for re-init
                frameOut = np.copy(frameCopy)

        t = time.time() - t1

        # plt.imshow(cv.cvtColor(frameOut, cv.COLOR_BGR2RGB)), plt.show()
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

            cv.putText(frameOut, fps, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (230, 230, 230))
            cv.putText(frameOut, numInfo, (50, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (230, 230, 230))
            cv.putText(frameOut, pitchInfo, (50, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (230, 230, 230))

        # # show frame & write
        cv.imshow('MultiTracker', frameOut)
        # vidWrite.write(frameOut)

        # quit on ESC button
        if cv.waitKey(1) & 0xFF == 27:
            break
