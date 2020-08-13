import os
import numpy as np
import copy
import colorsys
from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
# from PIL import Image, ImageFont, ImageDraw
from nets.yolo3 import yolo_body, yolo_eval
from utils.utils import letterbox_image
from useFunc.utils_kerasYolo import *
import math
import cv2 as cv

# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
# --------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_weights.h5',
        # "model_path": 'logs/ep072-loss9.096-val_loss9.517.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes_orig.txt',
        "score": 0.7,
        "iou": 0.3,
        "model_image_size": (416, 416)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化yolo
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        # self.c_pnt = c_pnt      # img center
        # self.foc_len = foc_len  # focal length in pixel
        # self.psi_x = 0
        # self.psi_y = 0
    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be box .h5 file.'

        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # # ---------------------------------------------------#
    # #   估算距离
    # # ---------------------------------------------------#
    # def calc_distance(self, bot_center, h_cam=0.8):
    #
    #     # Longitude & lateral distance of the object
    #     pitch, yaw = self.angs
    #     # print("angle:%.2f, %.2f" % self.angs)
    #
    #     # print(pitch)
    #     # print(self.psi_x)
    #     # self.psi_x = self.psi_x + pitch
    #     # self.psi_y = self.psi_y + yaw
    #     # print(self.psi_x)
    #     # calc the lateral and longitudinal offset
    #
    #     pos = bot_center - self.c_pnt
    #     lat, h = pos[0], pos[1]
    #     # print("center:%s,%s" % self.c_pnt)
    #     # print("bot_center:%.2f, %.2f" % tuple(bot_center))
    #     # print("lat:%.2f, h:%.2f" % (lat, h))
    #     # calc longitude & lateral distance
    #     angh = math.atan2(h, self.foc_len)
    #     angl = math.atan2(lat, self.foc_len)
    #     # print("angh:%.2f, angl: %.2f" % (angh, angl))
    #     # angh = math.radians(6)
    #     # D = h_cam / math.tan(angh)
    #     D = h_cam / math.tan(angh + self.psi_x + pitch)
    #     L = D * math.sin(angl + self.psi_y + yaw) / math.cos(angl)
    #     # L = D * math.tan(angl)
    #     return D, L, (angh + self.psi_x + pitch, angl + self.psi_y + yaw)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # start = timer()

        ret = None

        # 调整图片
        new_image_size = (self.model_image_size[0], self.model_image_size[1])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # # 设置字体
        # font_size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
        # font = ImageFont.truetype(font='font/simhei.ttf', size=font_size)
        #
        # thickness = (image.size[0] + image.size[1]) // 300

        # 1 if no object detected, return directly
        if len(out_scores) == 0:
            return ret, None

        # 2.1 if there is, processing each bbox
        stat = 0
        boxSet = np.empty((4,))
        for i, c in list(enumerate(out_classes)):
            box = out_boxes[i]
            top, left, bottom, right = box
            width = right - left
            height = bottom - top
            if c not in [0, 1, 2, 3, 5, 7] or (width < 50 or height < 50):
                continue
            else:
                # unpack
                stat += 1
                boxSet = np.vstack((boxSet, box))

        # 2.2 if there's no valid box, also return None
        if stat == 0:
            # img_out = np.copy(img_in)
            return ret, None
        else:
            # slice out the first empty box, convert it into format of
            # left, top, width, height
            boxSet = convert_box(boxSet[1:, :])
            ret = True
            return ret, boxSet

    # @staticmethod
    # # Checks if box matrix is box valid rotation matrix.
    # def isRotMat(R):
    #     Rt = np.transpose(R)
    #     shouldBeIdentity = np.dot(Rt, R)
    #     I = np.identity(3, dtype=R.dtype)
    #     n = np.linalg.norm(I - shouldBeIdentity)
    #     return n < 1e-6
    #
    # # Calculates rotation matrix to euler angles
    # def rotMat2EulAng(self, R):
    #     assert (self.isRotMat(R))
    #     sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    #     singular = sy < 1e-6
    #     if not singular:
    #         x = math.atan2(R[2, 1], R[2, 2])
    #         y = math.atan2(-R[2, 0], sy)
    #         z = math.atan2(R[1, 0], R[0, 0])
    #     else:
    #         x = math.atan2(-R[1, 2], R[1, 1])
    #         y = math.atan2(-R[2, 0], sy)
    #         z = 0
    #     box = [x, y, z]
    #     if False in [abs(box[i])<0.6 for i in range(len(box))]:
    #         x, y, z = 0, 0, 0
    #     return np.array([x, y, z])
    #
    # def ego_motion(self, img_train, img_query, num_fr):
    #     orb = cv.ORB_create()
    #     bf = cv.BFMatcher()
    #     kp1, des1 = orb.detectAndCompute(img_query, None)  # query
    #     kp2, des2 = orb.detectAndCompute(img_train, None)  # train
    #     # frm = np.copy(img_query)
    #
    #     if len(des1) < 8 or len(des2) < 8:
    #         print("cannot compensate ego motion: not enough feature pnts to be matched")
    #         return None
    #
    #     matches = bf.match(des1, des2)
    #     if len(matches) < 8:
    #         print("not enough matches in %d frame, pitch unchanged" % num_fr)
    #         return None
    #
    #     matches = sorted(matches, key=lambda x: x.distance)[:10]
    #     # img_out = cv.drawMatches(img_query, kp1, img_train, kp2, matches, None,
    #     #                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #
    #     # img_out = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
    #     kpts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.int)
    #     kpts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.int)
    #
    #     essMat, ret = cv.findEssentialMat(kpts1, kpts2, focal=self.foc_len)  # focal length to be revised
    #     R1, _, _ = cv.decomposeEssentialMat(essMat)
    #     # retval, R1, t, mask = cv.recoverPose(essMat, kpts1, kpts2)
    #     rotAngs = self.rotMat2EulAng(R1)
    #     # self.rotAng = [rotAngs[0], rotAngs[2]]
    #     # t = "%.2f" % rotAngs[0]
    #     # cv.putText(frm, t, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    #     # return pitch & yaw angle
    #     # print('ang_x:%.1f; ang_z:%.1f' % (self.rotAng[0], self.rotAng[1]))
    #     return rotAngs[1], rotAngs[2]

    def close_session(self):
        self.sess.close()
