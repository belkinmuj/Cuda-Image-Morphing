import numpy as np
import cv2
from matplotlib import pyplot as plt

import os


path = ""
imgs = os.walk(path)
imgs = list(imgs)[0][2]
imnames = [i for i in imgs if i[-4:] == "jpeg"]

for i in imnames:
    try:
        img_path = os.path.join(path,i)
        img = cv2.imread(img_path)
        mask = cv2.imread(img_path[:-5]+"_IUV.png")
        mask.shape
        img[mask == 0] = 0
        cv2.imwrite(os.path.join(path,"mask_result",i), img)
    except AttributeError:
        pass
    