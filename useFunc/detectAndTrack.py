import numpy as np
import cv2 as cv

# Object of interest for COCO classes -> for VOC need 2b modified
names_traffic = [1, 2, 3, 5, 7]  # bicycle,car,motorbike,bus,truck
obj_small = [1, 3]
obj_large = [2, 5, 7]
# names_traffic = [0,1,2]       #CITYSCAPES: vehicle, pedestrian, bike
# obj_small = [1,2]             # pedestrian, bike
# obj_large = [0]            # vehicle
# DNN-based model with YOLOv3 params
# currently CV2(4.2.0) hasn't fixed the problem: yolov4.cfg cause problem.
# - could try 3.4.11-pre (need manually install)
pathPre = 'C:\ProgamData\Yolo_v4\darknet\\build\darknet\\x64\\'
inSize = (416, 416)
net = cv.dnn_DetectionModel(pathPre + 'cfg\\yolov3.cfg', pathPre + 'yolov3.weights')
# net = cv.dnn_DetectionModel(pathPre + 'cfg\\yolov3-voc-CITYSCAPES.cfg', pathPre + '\\backup\\yolov3-voc-CITYSCAPES_6000.weights')
net.setInputSize(inSize)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)
print(net)
# read class names for detection
# with open(pathPre + 'data\\CITYSCAPES.names', 'rt') as f:
with open(pathPre + 'data\\coco.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')


def yolov3_det(net, frame_in):
    # pitch: camera pitch angle changes between two frames
    # set to 0 for testing
    ret = None
    img_in = np.copy(frame_in)
    # YOLOv3 inference
    classes, confidences, boxes = net.detect(img_in, confThreshold=0.6, nmsThreshold=0.8)

    # 1 if no object detected, return directly
    if len(confidences) == 0:
        return ret, None

    # 2.1 if there is, processing each bbox
    stat = 0
    boxSet = np.empty((4,))

    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):

        # filter out weak detection
        left, top, width, height = box
        # filter out some too-far objects
        # if classId not in names_traffic or (width < 50 and height < 50):
        #     continue
        if classId not in names_traffic:
            continue
        elif (classId in obj_large) and (width < 40 and height < 40):
            continue    # if it's vehicle
        elif (classId in obj_small) and (width < 20 or height < 20):
            continue    # if it's person or bicycle
        else:
            # unpack
            stat += 1
            boxSet = np.vstack((boxSet, box))

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
