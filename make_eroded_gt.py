# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import glob
import numpy as np
import PIL.Image as pil
import cv2

eval_cls = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
void_cls = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

class_map = dict(zip(eval_cls, range(len(eval_cls))))
seg_class_map = dict(zip(range(len(eval_cls)), range(len(eval_cls))))


def load_data():
    gt_path = "/content/data_semantics/training/semantic_eroded/*.png"
    gts = []
    gt_paths = sorted(glob.glob(gt_path))

        
    for image_path in gt_paths:
        image = pil.open(image_path)
        mask = np.array(image,dtype=np.uint8)
        for c in range(0,33):
            gt = (mask == c).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,30))
            eroded_gt = cv2.erode(gt,kernel)
            eroded_part = (gt!=eroded_gt)
            mask[eroded_part] = 255
        seg_img = pil.fromarray(mask)
        seg_img.save(image_path)
    

load_data()