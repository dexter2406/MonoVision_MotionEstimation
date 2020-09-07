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

    # testing params
    # - frame interval to estimate distance/  # to calc the relVel
    intv_dist = intv_RV = 5
    # - focal length, Camera height
    foc_len, H_cam = 1200, 1.8
    thresh = 1
    orig_pitch = 8
    # -----------------------------
    # foc_len_scale factor of fy, due to resize of the original video
    # manually set for testing
    foc_len_scale = 14 / 9
    # -----------------------------
    # Re-calibration settings
    thresh_cnt_RC = 4       # threshold to count
    crit_RC = 2*intv_RV     # criteria for do re-cal

    # settings for read & write video
    prePath = r'C:\ProgamData\global_dataset\img_vid'
    vidName = r'\vid9_2'
    fmt = '.mp4'
    cap = cv.VideoCapture(prePath + vidName + fmt)
    # cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps_vid = cap.get(cv.CAP_PROP_FPS)
    sizeW, sizeH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # size = (sizeW*2, int(sizeH*crop))
    size = (sizeW, sizeH)
    c_pnt = (int(sizeW / 2), int(sizeH / 2))
    camMat = np.array([[foc_len, 0, c_pnt[0]],
                       [0, foc_len / foc_len_scale, c_pnt[1]],
                       [0, 0, 1]])

    # init video writer
    # write_name = 'output\\' + vidName + '.avi'
    # vidWrite = cv.VideoWriter(write_name, fourcc, fps_vid, size)

    # bbox color settings
    # colors = [(255, 255, 255)]
    colors = []
    for i in range(20):
        # colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        colors.append((255, 255, 255))

    # Read first frame, quit if unable to read the video file
    success, _ = cap.read()
    if not success:
        print('Failed to read video')
        sys.exit(1)

    # Process video and track objects
    numFr = 0       # global count
    frmCnt = 0      # local count
    flag_fail = 1   # tracker failure report
    flag_RV = 0     # for relative velocity calc
    relVel = 0      # init relative velocity
    cnt_RC = 0      # re-calibration counter
    angs = np.array([0, 0, 0])
    pitch = orig_pitch  # orig pose
    egomotion = True    # turn on ego-motion compensation
    while cap.isOpened():

        # see if it's the end
        t1 = time.time()
        success, frame = cap.read()
        if not success:
            print("Done")
            break

        frameCopy = np.copy(frame)

        # Ego-motion estimation, independent of detection
        if egomotion:
            if numFr != 0:
                frame0 = np.copy(frameCopy)
            elif numFr % intv_dist == 0:
                angs = feat_match(frame0, frameCopy, numFr, camMat=camMat, crop=1,
                                  foc_len=foc_len, match_pnts=20, thresh=thresh)
                frame0 = np.copy(frameCopy)
                pitch += angs[0]    # pitch angle calc

            # re-cali of EM
            if abs(pitch - orig_pitch) > thresh_cnt_RC:
                cnt_RC += 1
                if cnt_RC > crit_RC:
                    pitch = orig_pitch  # orig pose
                    print("do RE-CALI")
            else:
                cntRC = 0

        # YOLO detection, re-init under conditions:
        # 0. the 1st-frame initialization
        # 1. after consecutive M frames
        # 2. tracking fails
        if (frmCnt == intv_dist or flag_fail == 1) or (numFr == 0):
            ret_yolo, boxes_yolo = yolov3_det(net, frameCopy)

            # (re)-init tracker only for valid yolo boxes
            if ret_yolo:
                flag_fail = frmCnt = 0
                flag_RV = False
                RVtmp_List = []  # tempRV
                dist0 = 0  # distances of previous frame for relVel calc
                frameOut, dist = calc_distance(boxes_yolo, pitch, frameCopy, c_pnt,
                                               foc_len_scale, colors, foc_len, H_cam)
                RVtmp_List, dist0, flag_RV = calc_relVel(dist0, dist, RVtmp_List, frmCnt,
                                                         flag_fail, fps_vid, intv_RV, flag_RV)

                multiTracker = initTrackObj(boxes_yolo, frameCopy)
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
                                               foc_len_scale, colors, foc_len, H_cam)
                # process the distance for relative velocity calc
                RVtmp_List, dist0, flag_RV = calc_relVel(dist0, dist, RVtmp_List, frmCnt,
                                                         flag_fail, fps_vid, intv_RV, flag_RV)
                frmCnt += 1

            else:
                flag_fail = 1  # flag for re-init
                # if tracker failed, output the original frame
                temp = multiTracker.getObjects()
                frameOut = np.copy(frameCopy)
                print("failed to update frame %d" % numFr)

            if frmCnt == intv_RV or ret_track is None:
                # output relVel, by calc the mean of the local relVel of 5 frames
                RVtmp_List, _, _ = calc_relVel(dist0, dist, RVtmp_List, frmCnt,
                                               flag_fail, fps_vid, intv_RV, flag_RV)
                draw_relVel(boxes_track, RVtmp_List, frameOut, colors)
                RVtmp_List = []

        # counter udpate
        numFr += 1
        t = time.time() - t1

        # print info
        if t > 0:
            print_info(frameOut, t, numFr, pitch, angs[0])

        # plt.imshow(cv.cvtColor(frameOut, cv.COLOR_BGR2RGB)), plt.show()
        # show frame & write

        cv.imshow('MultiTracker', frameOut)
        # vidWrite.write(frameOut)

        # quit on ESC button
        if cv.waitKey(1) & 0xFF == 27:
            break
