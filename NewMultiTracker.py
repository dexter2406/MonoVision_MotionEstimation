import cv2 as cv
import numpy as np
import time
from useFunc.testFunc import *
import matplotlib.pyplot as plt
import sys
from useFunc.featMatch import *
from useFunc.detectAndTrack import *
# from useFunc.detectAndTrack_gpu_test import *
from useFunc.utils import print_info


if __name__ == '__main__':

    # -----------testing params--------------
    intv_reint = 5      # interval to re-init detector
    intv_EM = 3         # interval for ego-motion estmation
    intv_RV = 5         # interval for relative velocity
    # foc_len, H_cam = 1300, 1.8       # foc_len, camera height
    foc_len, H_cam = 1200, 0.8
    thresh = 1
    # orig_pitch = 6
    orig_pitch = 0
    # - foc_len_scale factor of fy, due to resize of the original video
    #    manually set for testing
    foc_len_scale = 16/9          # width / height
    # foc_len_scale = 5/3
    # - Re-calibration settings
    thresh_cnt_RC = 4               # threshold to count
    crit_RC = 3 * intv_RV           # criteria for do re-cal

    # -----------video prop--------------
    pathPre = "C:\\ProgamData\\global_dataset\\img_vid\\"
    vidName = 'vid1_4'
    vidFormat = '.mp4'
    cap = cv.VideoCapture(pathPre + vidName + vidFormat)
    fps_vid = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps_vid = cap.get(cv.CAP_PROP_FPS)
    sizeW, sizeH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (sizeW, sizeH)
    c_pnt = (int(sizeW / 2), int(sizeH / 2))
    camMat = np.array([[foc_len, 0, c_pnt[0]],
                       [0, foc_len / foc_len_scale, c_pnt[1]],
                       [0, 0, 1]])
    # write_name = 'output\\' + 'NewMulTracker_' + vidName + '.avi'
    # vidWrite = cv.VideoWriter(write_name, fourcc, fps_vid, size)
    # boxes = [(500, 260, 130, 90), (100, 100, 100, 100)]

    # ----------- other default params -------------
    time2getFps = 1     # 1-sec average fps
    frmCnt_timer = 0    # cnt frame to calc fps
    start_time = time.time()
    numFr = 0           # global count
    frmCnt_reint = 0    # tmp frame count for re-init
    cnt_RC = 0          # re-calibration counter
    angs = np.array([0, 0, 0])
    pitch = orig_pitch  # orig pose
    egomotion = True   # turn on ego-motion compensation
    # ---------- verify source ---------------------
    success, _ = cap.read()     # read 1st frame
    if not success:
        print('Failed to read video')
        sys.exit(1)

    # ---------- Main ------------------------------
    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Done")
            break

        frameCopy = np.copy(frame)

        # ------------------- Ego-Motion estimation ----------------------
        # Ego-motion (EM) estimation, independent of others
        if egomotion:
            if numFr == 0:                          # 1st-frame initialization
                frame_for_EM = np.copy(frameCopy)
            elif numFr % intv_EM == 0:           # find matched features calc find rotation angles(adjacent frames)
                angs = feat_match(frame_for_EM, frameCopy, numFr, camMat=camMat, crop=1,
                                  foc_len=foc_len, match_pnts=20, thresh=thresh)
                frame_for_EM = np.copy(frameCopy)   # frame to be stored, independent of detection process
                pitch += angs[0]                    # accumulated pitch angle

            # re-cali of EM: if the pitch continuously exceeds original static pose for crit_RC frames,
            # reset the pitch to the default pose
            if abs(pitch - orig_pitch) > thresh_cnt_RC:
                cnt_RC += 1
                if cnt_RC > crit_RC:
                    pitch = orig_pitch  # orig pose
                    print("do RE-CALI")
            else:
                cntRC = 0

        # -------------- (Re)init for each M frames ------------
        # 0 - Re-Initialzation
        if frmCnt_reint == intv_reint or numFr == 0:
            # 0.0 - yolo deteted boxes
            ret_yolo, boxes_yolo = yolov3_det(net, frameCopy)
            # ret_yolo, boxes_yolo = yolov3_detect(net, frameCopy, size)
            # 0.1 - if no object, skip to next frame
            if not ret_yolo:
                frame_to_disp = np.copy(frameCopy)
                print("No obj detected by Yolo, skip frame %d" % numFr)

            # 0.2 - if objects exist, init multi-tracker
            else:
                frmCnt_reint = 0                # frame counter for reinitialization
                MultiTracker = {}               # Trackers with ID
                Boxes = {}                      # BBoxes with ID
                RelVels = {}                    # relative velocity container
                RelVels_to_disp = {}
                idSeed = 0                      # ID for objects, reset to 0 for each intv_reint frames
                frame0 = np.copy(frameCopy)     # base frame (will be stored recursively)
                frame_to_disp = np.copy(frameCopy)      # output frame, where info/box is printed/drawn on
                for box in boxes_yolo:
                    # temp_tracker init
                    box = tuple(box)
                    tracker = cv.TrackerMOSSE_create()
                    ret = tracker.init(frame0, box)
                    if ret:                             # store boxes & trackers with respective IDs
                        Boxes[idSeed] = box
                        MultiTracker[idSeed] = tracker
                        RelVels[idSeed] = []
                        idSeed += 1
                        # 0.2 - calc & disp distances
                        dist_new = calc_dist_test(box, pitch, c_pnt, foc_len, foc_len_scale, H_cam)  # (long, lat)
                        frame_to_disp = draw_dist(dist_new, box, frame_to_disp)
                    else:
                        print('there\'s object cannot be initialzed with tracker')
                        break

        # -------------- Detection and Tracking ---------------
        # 1 - normal process (after successful detection)
        elif ret_yolo:
            frame1 = np.copy(frameCopy)
            frmCnt_reint += 1                       # for re-init
            frame_to_disp = np.copy(frameCopy)      # frame for output

            # 1.0 - update for each frame, each box, calc RV
            for BoxId in list(MultiTracker):
                # 1.0.0 - if update fails, delete info in both MultiTrackers
                ret, box_new = MultiTracker[BoxId].update(frame1)
                if not ret:                         # delete objects if the tracker cannot update
                    del Boxes[BoxId]
                    del MultiTracker[BoxId]
                    del RelVels[BoxId]
                    if bool(MultiTracker):          # if empty, move on to next frame
                        continue

                # -------------- Calc relative distances and velocities ---------------
                # 1.0.1 - otherwise,calc temp distance_difference between individual trakcers
                else:
                    # 1.0.1.0 update Boxes, pop out previous box to calc RV
                    box_prev = Boxes[BoxId]    # pop prev box
                    Boxes[BoxId] = box_new     # push new box
                    # 1.0.1.1 - calc & disp distance in current frame
                    dist_new = calc_dist_test(box_new, pitch, c_pnt, foc_len, foc_len_scale, H_cam)  # (long, lat)
                    frame_to_disp = draw_dist(dist_new, box_new, frame_to_disp)
                    # 1.0.1.2 - calc prev_dist: longitudinal & lateral
                    dist_prev = calc_dist_test(box_prev, pitch, c_pnt, foc_len, foc_len_scale, H_cam)

                    # 1.0.1.3 - calc and display relative_velocity
                    RV = np.array([(dist_new[i] - dist_prev[i]) * fps_vid for i in range(2)])
                    RelVels[BoxId].append(RV)
                    RelVels_to_disp[BoxId] = sum(RelVels[BoxId]) / len(RelVels[BoxId])
                    frame_to_disp = draw_RV(RelVels_to_disp[BoxId], box_new, frame_to_disp)

        # -------------- Other stuff  ---------------
        # 2 - timer for average FPS
        frmCnt_timer += 1
        numFr += 1
        if (time.time() - start_time) > time2getFps:
            meanFPS = frmCnt_timer / (time.time() - start_time)
            frmCnt_timer = 0
            start_time = time.time()
            print("average FPS:%.2f" % meanFPS)
        # 3 - print info
        print_info_test(frame_to_disp, meanFPS, numFr, pitch, angs[0])

        # plt.imshow(frame_to_disp), plt.show()
        cv.imshow("new multitracker", frame_to_disp)
        # vidWrite.write(frame_to_disp)
        if cv.waitKey(1) & 0xFF == 27:
            cap.release()
            break
