import numpy as np
import cv2 as cv

# Object of interest for COCO classes -> for VOC need 2b modified
names_traffic = [1, 2, 3, 5, 7]  # bicycle,car,motorbike,bus,truck
obj_small = [1, 3]
obj_large = [2, 5, 7]
# names_traffic = [1, 5, 6, 13, 14]#VOC:bus,car
# obj_small = [1, 13, 14]     # bike, motorbike, person
# obj_large = [5, 6]          # bus, car
# DNN-based model with YOLOv3 params
# currently CV2(4.2.0) hasn't fixed the problem: yolov4.cfg cause problem.
# - could try 3.4.11-pre (need manually install)
conf_thresh = 0.7
nms_thresh = 0.3
pathPre = 'C:\ProgamData\Yolo_v4\darknet\\build\darknet\\x64\\'
inSize = (416, 416)
net = cv.dnn.readNetFromDarknet(pathPre + 'cfg\\yolov3.cfg', pathPre + 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# read class names for detection
with open(pathPre + 'data\\coco.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')


def yolov3_detect(net, frame_in, dim_frame):
    # pitch: camera pitch angle changes between two frames
    # set to 0 for testing
    ret = None
    W, H = dim_frame
    img_in = np.copy(frame_in)
    # YOLOv3 inference
    blob = cv.dnn.blobFromImage(img_in, 1 / 255.0, inSize, swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # if no objects exit, output None
    if len(layerOutputs) == 0:
        return ret, None

    # classes, confidences, boxes = net.detect(img_in, confThreshold=0.5, nmsThreshold=0.4)
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections

        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # threshold by confidence
            if confidence > conf_thresh:
                # scale the Bbox coordinates back relative to the size of the image,
                # Note: YOLO actually returns the center (x, y)-coordinates of
                # the Bbox followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    # ensure at least one detection exists

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        color = (255, 255, 255)
        stat = 0
        boxSet = np.empty((4,))
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # filter out weak detection
            box = [x, y, w, h]
            # filter out some too-far objects
            if confidences[i] not in names_traffic:
                continue
            elif (confidences[i] in obj_large) and (w < 40 and h < 40):
                continue  # if it's vehicle
            elif (confidences[i] in obj_small) and (w < 20 or h < 20):
                continue  # if it's person or bicycle
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
