import math
import numpy as np
import cv2 as cv


def draw_relVel(boxes, relVel, frame_in):
    num = len(boxes)
    if type(relVel) == int:
        pass

    else:
        for i in range(num):
            box = boxes[i]
            v = relVel[i]
            org = (int(box[0]), int(box[1]-25))
            label = "%.1f:m/s, %.1f:m/s" % (v[0], v[1])
            cv.putText(frame_in, label, org, cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (250, 250, 250), thickness=1)


def calc_relVel(dist0, dist1, relVel, flag, fps=24):
    # store current distances
    # - dist0 is the object, reVel unchanged
    if flag == 0:
        dist0 = np.copy(dist1)
        flag = 1
        return relVel, dist0, flag
    # use stored distances to calc the relative velocities
    else:
        reVel = (dist1 - dist0) * fps
        dist0 = np.copy(dist1)
        flag = 0
        return reVel, dist0, flag


# unpack the bboxes, calc the distance
def calc_distance(boxes, pitch, frame_in, c_pnt, foc_len=1200, H_cam=0.8):
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
        angL = math.atan2(lat, foc_len)
        D = H_cam / math.tan(angH + math.radians(pitch))
        L = D * math.sin(angL + math.radians(yaw)) / math.cos(angL)
        # D, L = calc_distance(bot_center[0], bot_center[1], angs)

        # store ditances, dim:N*2
        distancecs[i] = np.array([D, L])

        # output info
        dis_label = '%.1fm, %.1fm' % (D, L)
        p_text = (int(left), int(top) - 5)
        cv.rectangle(frame_in, p1, p2, (255, 255, 255), 2, 1)
        cv.putText(frame_in, dis_label, p_text, cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (50, 220, 220), thickness=1, lineType=4)
    return frame_in, distancecs


# Checks if a matrix is a valid rotation matrix.
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


def sel_ang(ang1, ang2, R1, R2, thresh_change=1.5):
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
