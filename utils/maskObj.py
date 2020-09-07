import numpy as np


def maskROI(imgIn, boxes):

    imgOut = np.copy(imgIn)

    for box in boxes:
        left, top, width, height = box
        # a mean value for all frame pxl as mask
        fill = int(imgOut.mean())
        # mask on
        imgOut[int(top):int(top+height), int(left):int(left+width), :] = fill

    return imgOut

