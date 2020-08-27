import math
import numpy as np
import cv2 as cv
import time


def print_info(frameOut, t, numFr, pitch, delt_p):

    font = cv.FONT_HERSHEY_SIMPLEX
    fps = 1/t
    fps = "FPS: %.1f" % (1 / t)
    numInfo = "Frame: %d" % numFr

    if delt_p > 0.01:
        pitchInfo = "Pitch: %.2f(down)" % pitch
    elif delt_p < -0.01:
        pitchInfo = "Pitch: %.2f(up)" % pitch
    else:
        pitchInfo = "Pitch: %.2f" % pitch

    cv.putText(frameOut, fps, (50, 50), font, 0.6, (80, 80, 230))
    cv.putText(frameOut, numInfo, (50, 80), font, 0.6, (80, 80, 230))
    cv.putText(frameOut, pitchInfo, (50, 110), font, 0.6, (80, 80, 230))


def calc_relVel(dist0, dist1, RVtmp_List, frmCnt, flag_fail, fps,
                intv_RV, flag_RV):

    # flip the flag
    flag_RV = not flag_RV

    # dafault: calc the relVel for every 5 frames
    # if the tracker works (in normal situ)
    if flag_fail == 0:

        # - the 1st frame (of each loop) is for init
        if frmCnt == 0:
            dist0 = dist1
        # - then, store the instant relVel, but it's unstable, therefore
        #   not for output
        elif 0 < frmCnt < intv_RV:
            RVtmp_List.append(dist1 - dist0)
            dist0 = dist1   # recursive
        # - at the 5th frame, output the mean of the instant relVel
        #   to smooth the value for better stability.
        #   But this introduces delay.
        elif frmCnt == intv_RV:
            RVtmp_List = sum(RVtmp_List) / len(RVtmp_List) * fps

    # if the tracker fails, then output the mean relVel in advance
    else:
        RVtmp_List = sum(RVtmp_List) / len(RVtmp_List) * fps

    # at 1st~4th frames, the results are just for update
    # at 5th frame/ the trakcer fails, the results are for output
    return RVtmp_List, dist0, flag_RV


def draw_relVel(boxes, relVel, frame_in, colors):
    num = len(boxes)
    if type(relVel) == int:
        pass

    else:
        for i in range(num):
            box = boxes[i]
            v = relVel[i]
            label = "%.1f:m/s, %.1f:m/s" % (v[0], v[1])
            # calc box/text positions
            textSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, thickness=1)
            if box[1] - 7*baseLine > 0:
                org_text = (int(box[0]), int(box[1] - 3.5*baseLine))
                pos1_rec = (int(box[0]), int(box[1] - 7*baseLine))
                pos2_rec = (int(box[0] + textSize[0]), int(box[1]-3.5*baseLine))
            else:
                org_text = (int(box[0]), int(box[1] + 7*baseLine))
                pos1_rec = (int(box[0]), int(box[1] + 4.5*baseLine))
                pos2_rec = (int(box[0] + textSize[0]), int(box[1]+7.5*baseLine))
            # draw
            cv.rectangle(frame_in, pos1_rec, pos2_rec, (100, 100, 255), cv.FILLED)
            cv.putText(frame_in, label, org_text, cv.FONT_HERSHEY_SIMPLEX,
                       0.5, colors[i], thickness=1)


# unpack the bboxes, calc the distance
def calc_distance(boxes, pitch, frame_in, c_pnt, scaling, colors, foc_len=1200, H_cam=0.8):
    # original position in degrees
    yaw = 0

    # # TO BE MODIFIED!
    # pitch_d, _, _ = angs        # pitch change
    # pitch = pitch + pitch_d     # accumulated pitch
    # draw tracked objects
    distancecs = np.empty((len(boxes), 2))

    for i, newbox in enumerate(boxes):
        # 0-left, 1-top; 2-width, 3-height
        left, top, width, height = newbox
        p1 = (int(left), int(top))
        p2 = (int(left + width), int(top + height))
        bot_center = np.array([int(left + width / 2), int(top + height)])
        bot_center = bot_center - c_pnt
        # calc longitude & lateral distance
        lat, h = bot_center[0], bot_center[1]
        angH = math.atan2(h, foc_len)
        angL = math.atan2(lat, foc_len/scaling)
        D = H_cam / math.tan(angH + math.radians(pitch))
        L = D * math.sin(angL + math.radians(yaw)) / math.cos(angL)
        # D, L = calc_distance(bot_center[0], bot_center[1], angs)

        # store ditances, dim:N*2
        distancecs[i] = np.array([D, L])

        # output info
        dis_label = '%.1fm, %.1fm' % (D, L)

        # draw box for obj
        cv.rectangle(frame_in, p1, p2, (255, 255, 255), 2)

        # draw text
        # - calc box/text positions
        textSize, baseLine = cv.getTextSize(dis_label, cv.FONT_HERSHEY_SIMPLEX,
                                            0.5, thickness=1)
        if top - 3.5*baseLine > 0:
            pos1_rec = (int(left), int(top))
            pos2_rec = (int(left + textSize[0]), int(top - 3.5*baseLine))
            pos_text = (int(left), int(top - 0.5*baseLine))
        else:
            pos1_rec = (int(left), int(top + 0.5*baseLine))
            pos2_rec = (int(left + textSize[0]), int(top + 4.5*baseLine))
            pos_text = (int(left), int(top + 4*baseLine))

        cv.rectangle(frame_in, pos1_rec, pos2_rec, (100, 100, 255), cv.FILLED)
        # - put on text
        cv.putText(frame_in, dis_label, pos_text, cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), thickness=1)

    return frame_in, distancecs


# Checks if box matrix is box valid rotation matrix.
def isRotMat(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotMat2EulAng(R):
    assert (isRotMat(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    x = math.degrees(x)
    y = math.degrees(y)
    z = math.degrees(z)
    return np.array([x, y, z])


def sel_ang(ang1, ang2, R1, R2, thresh_change=2):
    # select the more reasonable angles (two sets in total)
    if np.sum(abs(ang2)) > np.sum(abs(ang1)):
        ang = ang1
        R = R1
    else:
        ang = ang2
        R = R2
    # filter out the unstable value (empirical)
    for i in range(len(ang)):
        if abs(ang[i]) > thresh_change:
            ang[i] = 0

    return ang, R