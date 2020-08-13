import numpy as np
import cv2 as cv


names_traffic = [0, 1, 2, 3, 5, 7]  # person,bicycle,car,motorbike,bus,truck
# DNN-based model with YOLOv3 params
# currently CV2(4.2.0) hasn't fixed the problem: yolov4.cfg cause problem.
# - could try 3.4.11-pre (need manually install)
pathPre = 'C:\ProgamData\Yolo_v4\darknet\\build\darknet\\x64\\'
inSize = (416, 416)
net = cv.dnn_DetectionModel(pathPre + 'cfg\\yolov3.cfg', pathPre + 'yolov3.weights')
net.setInputSize(inSize)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)
# read class names for detection
with open(pathPre + 'data\\coco.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')


def yolov3_det(net, frame_in):
    # pitch: camera pitch angle changes between two frames
    # set to 0 for testing
    ret = None
    img_in = np.copy(frame_in)
    # YOLOv3 inference
    classes, confidences, boxes = net.detect(img_in, confThreshold=0.5, nmsThreshold=0.4)

    # 1 if no object detected, return directly
    if len(confidences) == 0:
        return ret, None

    # 2.1 if there is, processing each bbox
    stat = 0
    boxSet = np.empty((4,))
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):

        # filter out weak detection
        left, top, width, height = box
        # if size too small or classId not in names_traffic:
        if classId not in names_traffic or (width < 50 and height < 50):
            continue

        else:
            # unpack
            stat += 1
            boxSet = np.vstack((boxSet, box))
            # label = '%.2f' % confidence
            # label = '%s: %s' % (names[classId], label)
            # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # left, top, width, height = box
            # # if width < 50 and height < 50:
            # #     continue
            # # else:
            # box_large = np.array([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2])
            #
            # # draw bbox
            # top = max(top, labelSize[1])
            # cv.rectangle(frame_in, box_large, color=(255, 255, 255), thickness=2)
            # # cv.rectangle(img_in, (left, top - labelSize[1]), (left + labelSize[0], top - baseLine),
            # #                        (255, 255, 255), cv.FILLED)
            # # cv.putText(img_in, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            # # # show estimation
            # # cv.rectangle(img_in, (left, top+height+2*baseLine), (left + int(1*labelSize[0]),
            # #                                                                 top+height), (200,100,200), cv.FILLED)

    # 2.2 if there's no valid box, also return None
    if stat == 0:
        # img_out = np.copy(img_in)
        return ret, None
    else:
        # slice out the first empty box
        boxSet = boxSet[1:, :]
        ret = True
        return ret, boxSet


# init tracker
def initTrackObj(boxes, frame_in):

    multiTracker = cv.MultiTracker_create()
    img_in = np.copy(frame_in)
    num = len(boxes)
    boxes = list(map(tuple, boxes.reshape(num, 4)))

    # for i in range(len(num)):
    for i in range(num):
        _, _, w, h = boxes[i]
        if w > 50 and h > 50:
            multiTracker.add(cv.TrackerMOSSE_create(), img_in, boxes[i])
        # t = time.time() - t1
    # print('Selected bounding boxes {}'.format(bboxes)).
    return multiTracker


# init tracker
def initTrackObj_test(boxes, frame_in):
    multiTracker = cv.MultiTracker_create()
    img_in = np.copy(frame_in)
    num = len(boxes)
    newbox = boxes[:, [1, 0, 2, 3]]
    newbox[:, [2]] = boxes[:, [2]] - boxes[:, [0]]
    newbox[:, [3]] = boxes[:, [3]] - boxes[:, [1]]
    boxes = list(map(tuple, newbox.reshape(num, 4)))

    # for i in range(len(num)):
    for i in range(num):
        _, _, w, h = boxes[i]
        if w > 50 and h > 50:
            multiTracker.add(cv.TrackerMOSSE_create(), img_in, boxes[i])
        # t = time.time() - t1
    # print('Selected bounding boxes {}'.format(bboxes)).
    return multiTracker
