import cv2 as cv
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image


def letterbox_image_np(image, size):
    iw, ih = image.shape[1], image.shape[0]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # resize image according to (416,416) & orig size
    image = cv.resize(imgNP, dsize=(nw, nh), interpolation=cv.INTER_CUBIC)
    # plt.imshow(image), plt.show()
    new_image = 128 * np.ones((h, w, 3), dtype=np.uint8)
    # plt.imshow(new_image), plt.show()
    n = np.array(new_image)
    offset_h = (h - nh) // 2
    offset_w = (w - nw) // 2
    new_image[offset_h:offset_h + nh, offset_w:offset_w + nw] = image
    return new_image


def letterbox_image_pil(image, size):
    iw, ih = image.size  # as "Image" object
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # resize image according to (416,416) & orig size
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    # plt.imshow(new_image), plt.show()
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


if __name__ == '__main__':
    # np_frame
    imgNP = cv.imread(r"C:\ProgamData\global_dataset\img_vid\down.jpg")
    # pil_frame
    imgPIL = cv.cvtColor(imgNP, cv.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(np.uint8(imgPIL))

    size = (416, 416)
    # methodNP
    newImg1 = letterbox_image_np(imgNP, size)
    newImg1 = cv.cvtColor(newImg1, cv.COLOR_BGR2RGB)
    # methodPIL
    newImg2 = letterbox_image_pil(imgPIL, size)

    plt.imshow(newImg1), plt.show()
    plt.imshow(newImg2), plt.show()
    pass
