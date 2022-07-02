# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
import matplotlib.pyplot as plt
from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}


    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

class KITTISemanticDataset(KITTIDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTISemanticDataset, self).__init__(*args, **kwargs)
        self.eval_cls = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
        self.void_cls = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.class_map = dict(zip(self.eval_cls, range(len(self.eval_cls))))
        self.decode_class_map = dict(zip(range(len(self.eval_cls)), self.eval_cls))

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_semantic(self, frame_index, do_flip):
        f_str = "{:06d}_10{}".format(2*frame_index, self.img_ext)
        #print(f_str)
        color = self.loader(os.path.join(self.semantic_path, "training/image_2", f_str))
        gt = pil.open(os.path.join(self.semantic_path, "training/semantic", f_str))
        #att = np.load(os.path.join(self.semantic_path, "training/distance", "{:05d}_dis.npy".format(frame_index)))

        #print(np.array(gt).max())
        gt = pil.fromarray(self.encode_segmap(np.array(gt)))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
            gt = gt.transpose(pil.FLIP_LEFT_RIGHT)
        #print(np.array(gt).max())
        return color, gt

    def get_distance(self, frame_index, do_flip):
        f_str = "{:06d}_10__dis{}".format(2*frame_index, ".npy")
        dis = np.load(os.path.join(self.distance_path, f_str))
        
        #print(dis.shape)
        if do_flip:
            dis = dis

        return dis

    def encode_segmap(self, mask):
        # Put all void classes to zero
        #print((mask==0))
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
        #dst = cv2.erode(mask,kernel)
        for c in self.void_cls:
            mask[mask == c] = 255
        for c in self.eval_cls:
            mask[mask == c] = self.class_map[c]

        return mask


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path



class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
