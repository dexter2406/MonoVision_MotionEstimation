import cv2 as cv
import numpy as np
from useFunc.utils import rotMat2EulAng, sel_ang

# ORB-feature matching between consecutive frames to find
# the Essential matrix -> pitch angle to compensate go-motion
# - img_train: previous frame; img_query: current frame
# PS: it's independent of BBox
orb = cv.ORB_create()
bf = cv.BFMatcher()


def feat_match(img_train, img_query, num_fr, camMat, crop=1,
               foc_len=1200, match_pnts=20, thresh=1):
    # in practice for original video, foc_len can be applied,
    # in testing for resized video, camMat is applied

    kp1, des1 = orb.detectAndCompute(img_train, None)
    kp2, des2 = orb.detectAndCompute(img_query, None)
    # frm = np.copy(img_query)

    if len(des1) < 8 or len(des2) < 8:
        print("not enough feature pnts to be matched")
        return np.zeros((3,))

    matches = bf.match(des2, des1)  # pos1 for query
    if len(matches) < 8:
        print("not enough matches in %d frame, pitch unchanged" % num_fr)
        return np.zeros((3,))

    # sort matches according to score
    matches = sorted(matches, key=lambda x: x.distance)[:match_pnts]
    # extract matched-feature coorinates
    kpts1 = np.array([kp1[m.trainIdx].pt for m in matches], dtype=np.int)
    kpts2 = np.array([kp2[m.queryIdx].pt for m in matches], dtype=np.int)

    # essMat, mask = cv.findEssentialMat(kpts1, kpts2, focal=foc_len, prob=0.9999, threshold=0.1)  # focal length to be revised
    essMat, mask = cv.findEssentialMat(kpts1, kpts2, camMat, prob=0.9999, threshold=0.1)
    # matches = [matches[i] for i in range(len(mask)) if mask[i] == 1]
    # img_out = cv.drawMatches(img_query, kp2, img_train, kp1, matches, None,
    #                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    R1, R2, t = cv.decomposeEssentialMat(essMat)
    rotAngs1 = rotMat2EulAng(R1)
    rotAngs2 = rotMat2EulAng(R2)
    rotAngs, RMat = sel_ang(rotAngs1, rotAngs2, R1, R2, thresh)
    # for i in range(len(rotAngs)):
    #     if abs(rotAngs[i]) > 3:
    #         print("the %sth param: %.3f, frame %s" %(i, rotAngs[i], num_fr))
    #         print(rotAngs1)
    #         print(rotAngs2)
    # t_ang = "pitch:%.3f; yaw:%.3f; row:%.3f" % (rotAngs[0], rotAngs[1], rotAngs[2])
    # t_fr = "%s" % num_fr
    # img_out = cv.putText(img_out, t_ang, (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    # img_out = cv.putText(img_out, t_fr, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    # return pitch & yaw angle
    # return rotAngs[0], rotAngs[2]
    # return rotAngs, img_out
    return rotAngs