import cv2 as cv
import numpy as np
import time
from useFunc.testFunc import *
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':

    # -----------testing params--------------
    intv_reint = intv_RV = 5    # interval to re-init detector & relative velocity
    foc_len, H_cam = 1200, 1.8  # foc_len, camera height
    thresh = 1
    orig_pitch = 0
    # - foc_len_scale factor of fy, due to resize of the original video
    #    manually set for testing
    foc_len_scale = 5/3         # width / height
    # - Re-calibration settings
    thresh_cnt_RC = 4           # threshold to count
    crit_RC = 2 * intv_RV       # criteria for do re-cal

    # -----------video prop--------------
    cap = cv.VideoCapture(r"C:\ProgamData\global_dataset\img_vid\vid3_5.mp4")
    fps_vid = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps_vid = cap.get(cv.CAP_PROP_FPS)
    sizeW, sizeH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # size = (sizeW*2, int(sizeH*crop))
    size = (sizeW, sizeH)
    c_pnt = (int(sizeW / 2), int(sizeH / 2))
    camMat = np.array([[foc_len, 0, c_pnt[0]],
                       [0, foc_len / foc_len_scale, c_pnt[1]],
                       [0, 0, 1]])
    # boxes = [(500, 260, 130, 90), (100, 100, 100, 100)]

    # ----------- other default params -------------
    time2getFps = 1     # 1-sec average fps
    frmCnt_timer = 0    # cnt frame to calc fps
    start_time = time.time()
    numFr = 0           # global count
    frmCnt = 0          # local count
    cnt_RC = 0          # re-calibration counter
    angs = np.array([0, 0, 0])
    pitch = orig_pitch  # orig pose
    egomotion = True    # turn on ego-motion compensation

    # ---------- verify source ---------------------
    success, _ = cap.read()
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

        # 0 - (Re)init only for each M frames
        if frmCnt_timer == 0 or frmCnt_timer % 5 == 0:
            MultiTracker0 = {}
            idSeed = 0
            frame0 = np.copy(frameCopy)
            frame_for_disp = frame0
            for box in boxes:
                # temp_tracker init
                tracker = cv.TrackerMOSSE_create()
                ret = tracker.init(frame0, box)
                if ret:
                    MultiTracker0[str(idSeed)] = tracker
                    idSeed += 1
                else:
                    break

        # 1 - normal process
        else:
            frame1 = np.copy(frameCopy)
            MultiTracker1 = MultiTracker0.copy()
            RelVels = {}

            # 1.1 - update for each frame, each box, calc RV
            for id in list(MultiTracker1):
                # 1.1.1 - if update fails, delete info in both MultiTrackers
                ret, box_new = MultiTracker1[id].update(frame1)
                if not ret:
                    del MultiTracker1[id]
                    del MultiTracker0[id]
                    if bool(MultiTracker1):     # if empty then move on to next frame
                        continue

                # 1.1.2 - otherwise,calc temp distance_difference between individual trakcers
                else:
                    # 1.1.1.1 - calc & disp distance in current frame
                    dist_new = calc_dist_test(box_new, pitch, c_pnt, foc_len_scale)  # (long, lat)
                    frame_for_disp = draw_dist(dist_new, box_new, frame1)
                    # 1.1.1.2 - calc prev_dist: longitudinal & lateral
                    ret, box_prev = MultiTracker0[id].update(frame0)
                    dist_prev = calc_dist_test(box_prev, pitch, c_pnt, foc_len_scale)

                    # 1.1.2.1 - calc and display relative_velocity
                    # dim: RelVels = {id: [D,L]}
                    RelVels[id] = [(dist_prev[i] - dist_new[i]) * fps_vid for i in range(2)]
                    frame_for_disp = draw_RV(RelVels[id], box_new, frame_for_disp)
                    if ret:
                        print(box_prev)
                        print(box_new)
                    else:
                        print("something wrong with frame0")

            # 2 - recursively store for next round, unless the next one needs re-init
            if (frmCnt_timer + 1) % 5 != 0:
                MultiTracker0 = MultiTracker1.copy()
                frame0 = frameCopy

            # # 3 - output bboxes and RV
            # if bool(MultiTracker1) is True:     # if not empty
            #     for id, box_tracker in MultiTracker1.items():
            #         # draw bboxes
            #         _, box = box_tracker.update(frame1)
            #         l, t, w, h = box
            #         cv.rectangle(frame0, (int(l), int(t)), (int(l + w), int(t + h)), (255, 255, 255))
            #         print('D:%.1f; L:%.1f' % (RelVels[id][0], RelVels[id][1]))
            # else:
            #     print("no object exists")

        # 4 - timer for average FPS
        frmCnt_timer += 1
        if (time.time() - start_time) > time2getFps:
            fps = frmCnt_timer / (time.time() - start_time)
            frmCnt_timer = 0
            start_time = time.time()
            print("average FPS:%.2f" % fps)

        # plt.imshow(frame0), plt.show()
        cv.imshow("new multitracker", frame_for_disp)
        if cv.waitKey(1) & 0xFF == 27:
            cap.release()
            break
