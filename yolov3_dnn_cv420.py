import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# test settings
thresh_change = 0.5
match_pnts = 20
crop = 1
interv = 3

# default settings
foc_len = 1300
inSize = (416, 416)
pathPre = 'C:\ProgamData\Yolo_v4\darknet\\build\darknet\\x64\\'
dataPre = 'C:\\ProgamData\\global_dataset\\img_vid\\'
vid_name = "vid1_2"
names_traffic = [0, 1, 2, 3, 5, 7]  # person,bicycle,car,motorbike,bus,truck
num_skip_fr = 5  # skip some 5 consecutive no-object frames
num_min_match = 8  # looking for at least 8 matches
fmt = '.mp4'
fourcc = cv.VideoWriter_fourcc(*'XVID')


# --------------Functions----------------
# # Calc the longitude & lateral distance of the object
# # in an image
# def calc_distance(lat, h, pit, H_cam=0.8):
#     # default: original camera is in neutral position
#     # pitch, yaw = 0, 0   # in degrees, original position
#     yaw = 0
#
#     # calc longitude & lateral distance
#     ang1 = math.atan2(h, foc_len)
#     ang2 = math.atan2(lat, foc_len)
#     D = H_cam / math.tan(ang1 - math.radians(pit))
#     L = D * math.sin(ang2 + math.radians(yaw)) / math.cos(ang2)
#     return D, L


# # Checks if a matrix is a valid rotation matrix.
# def isRotMat(R):
#     Rt = np.transpose(R)
#     shouldBeIdentity = np.dot(Rt, R)
#     I = np.identity(3, dtype=R.dtype)
#     n = np.linalg.norm(I - shouldBeIdentity)
#     return n < 1e-6
#
#
# # Calculates rotation matrix to euler angles
# def rotMat2EulAng(R):
#     # assert (isRotMat(R))
#     assert (R[2, 0] != 1)
#     y = -math.asin(R[2, 0])
#     y = min(abs(y), abs(math.pi + y))
#     x = math.atan2(R[2, 1] / math.cos(y), R[2, 2] / math.cos(y))
#     z = math.atan2(R[1, 0] / math.cos(y), R[0, 0] / math.cos(y))
#     # sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#     # singular = sy < 1e-6
#     # if not singular:
#     #     x = math.atan2(R[2, 1], R[2, 2])
#     #     y = math.atan2(-R[2, 0], sy)
#     #     z = math.atan2(R[1, 0], R[0, 0])
#     # else:
#     #     x = math.atan2(-R[1, 2], R[1, 1])
#     #     y = math.atan2(-R[2, 0], sy)
#     #     z = 0
#     x = math.degrees(x)
#     y = math.degrees(y)
#     z = math.degrees(z)
#     return np.array([x, y, z])
#
#
# def sel_ang(ang1, ang2, R1, R2):
#     # select the more reasonable angles (two sets in total)
#     if np.sum(abs(ang2)) > np.sum(abs(ang1)):
#         ang = ang1
#         R = R1
#     else:
#         ang = ang2
#         R = R2
#     # filter out the unstable value (empirical)
#     for i in range(len(ang)):
#         if abs(ang[i]) > thresh_change:
#             ang[i] = 0
#
#     return ang, R

# def relVelEsti(relVelPrev, DList, fps_vid):
#     DList = [i for i in DList if abs(i) < 20]  # [1, 2, 5, -5, -3]
#     listLen = len(DList)  # 5
#     DListP = [i for i in DList if i > 0]
#     DListN = [i for i in DList if i < 0]
#     critP = len(DListP)  # increasing, 3
#     critN = listLen - critP
#
#     if critP > critN:
#         DOutput = DListP
#         relVel = np.mean(DOutput) * fps_vid
#     elif critP > critN:
#         DOutput = DListN
#         relVel = np.mean(DOutput) * fps_vid
#     else:
#         relVel = relVelPrev
#
#     print(relVel)

# # ORB-feature matching between consecutive frames to find
# # the Essential matrix -> pitch angle to compensate go-motion
# # - img_train: previous frame; img_query: current frame
# # PS: it's independent of BBox
# def feat_match(img_train, img_query, num_fr):
#     img_query = img_query[0:int(crop * sizeH), 0:sizeW]
#     img_train = img_train[0:int(crop * sizeH), 0:sizeW]
#     # img_query = img_query[int(crop*sizeH):sizeH, 0:sizeW]
#     # img_train = img_train[int(crop*sizeH):sizeH, 0:sizeW]
#     kp1, des1 = orb.detectAndCompute(img_train, None)  # query
#     kp2, des2 = orb.detectAndCompute(img_query, None)  # train
#     # frm = np.copy(img_query)
#
#     if len(des1) < 8 or len(des2) < 8:
#         print("cannot compensate ego motion: not enough feature pnts to be matched")
#         return None
#
#     matches = bf.match(des2, des1)  # pos1 for query
#     if len(matches) < 8:
#         print("not enough matches in %d frame, pitch unchanged" % num_fr)
#         return None
#
#     matches = sorted(matches, key=lambda x: x.distance)[:match_pnts]
#
#     # img_out = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
#     kpts1 = np.array([kp1[m.trainIdx].pt for m in matches], dtype=np.int)
#     kpts2 = np.array([kp2[m.queryIdx].pt for m in matches], dtype=np.int)
#
#     essMat, mask = cv.findEssentialMat(kpts1, kpts2, focal=foc_len, prob=0.9999, threshold=0.1)  # focal length to be revised
#     # essMat, mask = cv.findEssentialMat(kpts1, kpts2, focal=foc_len, method=8, prob=0.9999)
#     matches = [matches[i] for i in range(len(mask)) if mask[i] == 1]
#     img_out = cv.drawMatches(img_query, kp2, img_train, kp1, matches, None,
#                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     R1, R2, t = cv.decomposeEssentialMat(essMat)
#     rotAngs1 = rotMat2EulAng(R1)
#     rotAngs2 = rotMat2EulAng(R2)
#     rotAngs, RMat = sel_ang(rotAngs1, rotAngs2, R1, R2)
#     # for i in range(len(rotAngs)):
#     #     if abs(rotAngs[i]) > 3:
#     #         print("the %sth param: %.3f, frame %s" %(i, rotAngs[i], num_fr))
#     #         print(rotAngs1)
#     #         print(rotAngs2)
#     t_ang = "pitch:%.3f; yaw:%.3f; row:%.3f" % (rotAngs[0], rotAngs[1], rotAngs[2])
#     t_fr = "%s" % num_fr
#     # img_out = cv.putText(img_out, t_ang, (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
#     img_out = cv.putText(img_out, t_fr, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
#     # return pitch & yaw angle
#     # return rotAngs[0], rotAngs[2]
#     return img_out, rotAngs


# YOLOv3 detect for:
# - draw BBOX with label and confidence
# - extract bottom-center coordinates
def yolov3_det(net, frame, pit):
    # pitch: camera pitch angle changes between two frames
    # set to 0 for testing

    img_in = np.copy(frame)
    # YOLOv3 inference
    classes, confidences, boxes = net.detect(img_in, confThreshold=0.1, nmsThreshold=0.4)

    # if no object detected, return directly
    if len(confidences) == 0:
        return img_in
        # return bbox, frame, class_set

    # processing each bbox
    stat = 0
    DList, LList = [], []
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):

        # filter out weak detection
        if confidence < 0.8 or classId not in names_traffic:
            continue

        else:
            # unpack
            stat += 1
            label = '%.2f' % confidence
            label = '%s: %s' % (names[classId], label)
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            left, top, width, height = box
            box_large = np.array([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2])

            # ------------------------------
            # localize (bottom-center) -> distance estimation
            bot_center = np.array([int(left + width / 2), int(top + height)])
            bot_center = bot_center - c_pnt
            D, L = calc_distance(bot_center[0], bot_center[1], pit)
            dis_label = '%.1f, %.1f' % (D, L)
            foc_info = 'focal: %s px' % foc_len
            fr_info = "frame: %s" % num_fr
            # DList = DList.append(D)
            # LList = LList.append(L)
            # ------------------------------

            # draw bbox
            top = max(top, labelSize[1])
            img_out = cv.rectangle(img_in, box_large, color=(0, 255, 0), thickness=3)
            img_out = cv.rectangle(img_out, (left, top - labelSize[1]), (left + labelSize[0], top - baseLine),
                                   (255, 255, 255), cv.FILLED)
            img_out = cv.putText(img_out, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            # show estimation
            img_out = cv.rectangle(img_out, (left, top + height + 2 * baseLine), (left + int(1 * labelSize[0]),
                                                                                  top + height), (200, 100, 200),
                                   cv.FILLED)
            img_out = cv.putText(img_out, dis_label, (left, top + height + 2 * baseLine), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                 (0, 0, 0))
            img_out = cv.putText(img_out, foc_info, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220))
            img_out = cv.putText(img_out, fr_info, (50, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220))

    if stat == 0:
        img_out = img_in.copy()

    return img_out


# ----------Initialization---------------
# Initiate ORB detector and BF matcher
orb = cv.ORB_create()
bf = cv.BFMatcher()
fps = 0.0  # recursively averaged FPS
num_mark = 0
flag = 1
pitch = 0
pitch_list = []

# read input img/vid with its properties
# frame = cv.imread(pathPre + 'dog.jpg')
cap = cv.VideoCapture(dataPre + vid_name + fmt)
fps_vid = cap.get(cv.CAP_PROP_FPS)
sizeW, sizeH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# size = (sizeW*2, int(sizeH*crop))
size = (sizeW, sizeH)
c_pnt = (int(sizeW / 2), int(sizeH / 2))  # center point
# init video writer
# vidWrite = cv.VideoWriter(vid_name + '_out.avi', fourcc, fps_vid, size)
# write_name = (vid_name + "_%spnt.avi") % match_pnts
write_name = vid_name + "_test_.avi"
vidWrite = cv.VideoWriter(write_name, fourcc, fps_vid, size)

# DNN-based model with YOLOv3 params
# currently CV2(4.2.0) hasn't fixed the problem: yolov4.cfg cause problem.
# - could try 3.4.11-pre (need manually install)
net = cv.dnn_DetectionModel(pathPre + 'cfg\\yolov3.cfg', pathPre + 'yolov3.weights')
net.setInputSize(inSize)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)
# read class names for detection
with open(pathPre + 'data\\coco.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')

# ----------Main Part---------------
if __name__ == '__main__':

    # process frame-wise
    num_fr = 0  # count frame number for init
    acc_ang = 0
    cnt_ang = np.array([[0, 0, 0]])
    angs = np.array([0, 0, 0])
    while cap.isOpened():
        t1 = time.time()
        # get current frame
        ret, frame_orig = cap.read()
        if not ret:
            print("No frame left, exit.")
            break
        frame_copy = frame_orig.copy()

        # Feature matching between consecutive frames
        # Init 1st frame, direct in to next frame without matching
        if num_fr == 0:
            num_fr += 1
            frame1 = frame_copy
            frame_out = frame1
            print("initialized")
            vidWrite.write(frame_out)
            continue

        elif num_fr == 1:
            print("running...")

        # MAIN
        # ORB feature extracted inside (each) bbox
        num_fr += 1
        frame2 = frame_copy
        # match between two frames to calc the pitch & yaw angles
        # by essential matrix
        if num_fr % interv == 0:
            if flag == 1:
                frame_store = frame2
                flag = 0    # 0-hold
            elif flag == 0:
                _, angs = feat_match(frame_store, frame2, num_fr)
                pitch = pitch + angs[0]
                print(angs[0])
                flag = 1    # 1-updated

        # if num_fr % 8 == 0 and flag == 1:
        #     frame_store = frame2
        #     num_mark = num_fr + interv
        #     flag = 0
        # elif num_fr == num_mark:
        #     _, angs = feat_match(frame_store, frame2, num_fr)
        #     flag = 1
        # print("frame:%s; angle:%s" % (num_fr, angs))

        # (test) show angle changes
        # acc_ang = np.add(acc_ang, angs)
        # acc_t = acc_ang[np.newaxis, :]
        # cnt_ang = np.vstack((cnt_ang, acc_t))
        # print("frame:%s; acc_angle:%s" % (num_fr, acc_ang))

        # frame_out = feat_match(frame1, frame2, num_fr)
        # print("%s" % pitch_ang)
        # release the var_name for next frame
        frame1 = frame2
        # YOLOv3 detection with Bbox coordinates
        frame_out = yolov3_det(net, frame2, pitch)
        pitch_list.append(pitch)
        pitch_info = "pitch: %.3f" % pitch
        print("acc_pit:%.3f, flag=%d, frame=%d" % (pitch, flag, num_fr))
        cv.putText(frame_out, pitch_info, (50, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120))
        # display FPS
        # fps = (fps + (1. / (time.time() - t1))) / 2
        # print(time.time() - t1)
        # print("fps= %.1f" % fps)

        # display
        cv.imshow('output', frame_out)
        num_info = "%d" % num_fr
        # filepath = "C:\\Users\\IAV_2_GPU_WS\\Desktop\\debug_match\\frame"+num_info+".jpg"
        # cv.imwrite(filepath, frame_copy)
        vidWrite.write(frame_out)

        if 0xFF & cv.waitKey(30) == 27:
            print('done')
            break

    print("Done: %d frames processed" % num_fr)
    # show image
    cap.release()
    cv.destroyAllWindows()

    # show pitch changes
    plt.figure()
    plt.plot(pitch_list)
    plt.legend(["pitch"])
    # plt.title("%s pnts" % match_pnts)
    plt.show()
