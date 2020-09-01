import cv2 as cv
import numpy as np
import math


def print_info_test(frameOut, pitch, yaw, row):
    font = cv.FONT_HERSHEY_SIMPLEX
    pitchInfo = 'Pitch: %.1f' % pitch
    yawInfo = 'Yaw: %.1f' % yaw
    rowInfo = 'Row: %.1f' % row
    # fps = 1/t
    # fps = "FPS: %.1f" % (1 / t)
    # numInfo = "Frame: %d" % numFr

    # if delt_p > 0.01:
    #     pitchInfo = "Pitch: %.2f(down)" % pitch
    # elif delt_p < -0.01:
    #     pitchInfo = "Pitch: %.2f(up)" % pitch
    # else:
    #     pitchInfo = "Pitch: %.2f" % pitch

    # cv.putText(frameOut, fps, (50, 50), font, 0.6, (80, 80, 230))
    # cv.putText(frameOut, numInfo, (50, 80), font, 0.6, (80, 80, 230))
    cv.putText(frameOut, pitchInfo, (50, 50), font, 1, (80, 80, 230))
    cv.putText(frameOut, yawInfo, (50, 90), font, 1, (80, 80, 230))
    cv.putText(frameOut, rowInfo, (50, 130), font, 1, (80, 80, 230))


# unpack the bboxes, calc the distance
def calc_dist_test(box, pitch, c_pnt, scaling, foc_len=1200, H_cam=0.8):
    # original position in degrees
    yaw = 0

    # 0-left, 1-top; 2-width, 3-height
    left, top, width, height = box
    bot_center = np.array([int(left + width / 2), int(top + height)])
    bot_center = bot_center - c_pnt
    # calc longitude & lateral distance
    lat, h = bot_center[0], bot_center[1]
    angH = math.atan2(h, foc_len)
    angL = math.atan2(lat, foc_len / scaling)
    D = H_cam / math.tan(angH + math.radians(pitch))
    L = D * math.sin(angL + math.radians(yaw)) / math.cos(angL)

    return D, L


def draw_dist(dist, box, frame):
    frame_out = np.copy(frame)

    # output info
    dis_label = '%.1fm, %.1fm' % dist  # (D, L)

    # draw box for obj
    left, top, width, height = box
    rect_p1 = (int(left), int(top))
    rect_p2 = (int(left + width), int(top + height))
    cv.rectangle(frame_out, rect_p1, rect_p2, (255, 255, 255), 2)

    # draw text
    # - calc box/text positions
    textSize, baseLine = cv.getTextSize(dis_label, cv.FONT_HERSHEY_SIMPLEX,
                                        0.5, thickness=1)
    if top - 3.5 * baseLine > 0:
        pos1_rec = (int(left), int(top))
        pos2_rec = (int(left + textSize[0]), int(top - 3.5 * baseLine))
        pos_text = (int(left), int(top - 0.5 * baseLine))
    else:
        pos1_rec = (int(left), int(top + 0.5 * baseLine))
        pos2_rec = (int(left + textSize[0]), int(top + 4.5 * baseLine))
        pos_text = (int(left), int(top + 4 * baseLine))

    cv.rectangle(frame_out, pos1_rec, pos2_rec, (100, 100, 255), cv.FILLED)
    # - put on text
    cv.putText(frame_out, dis_label, pos_text, cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), thickness=1)

    return frame_out


def draw_RV(RV, box, frame):
    frame_out = np.copy(frame)

    # output info: relative velocity [d, l]
    RV_label = '%.1fm/s, %.1fm/s' % (RV[0], RV[1])

    # draw text
    # - box info
    left, top = box[0], box[1]
    # - calc box/text positions
    textSize, baseLine = cv.getTextSize(RV_label, cv.FONT_HERSHEY_SIMPLEX, 0.5, thickness=1)
    if box[1] - 7 * baseLine > 0:
        org_text = (int(left), int(top - 3.5 * baseLine))
        pos1_rec = (int(left), int(top - 7 * baseLine))
        pos2_rec = (int(left + textSize[0]), int(top - 3.5 * baseLine))
    else:
        org_text = (int(left), int(top + 7 * baseLine))
        pos1_rec = (int(left), int(top + 4.5 * baseLine))
        pos2_rec = (int(left + textSize[0]), int(top + 7.5 * baseLine))
    # - draw
    cv.rectangle(frame_out, pos1_rec, pos2_rec, (100, 100, 255), cv.FILLED)
    cv.putText(frame_out, RV_label, org_text, cv.FONT_HERSHEY_SIMPLEX,
               0.5, (255, 255, 255), thickness=1)

    return frame_out
