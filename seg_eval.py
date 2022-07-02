# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import PIL.ImageOps
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


from skimage.segmentation import mark_boundaries
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
#from fullcrf import fullcrf

#labels = ['unlabeled','ego vehicle','rectification border','out of roi','static','dynamic','ground','road','sidewalk','parking','rail track',
#    'building','wall','fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain',
#    'sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','license plate']

labels = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain',
    'sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
    
eval_cls = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
void_cls = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

class_map = dict(zip(eval_cls, range(len(eval_cls))))
seg_class_map = dict(zip(range(len(eval_cls)), range(len(eval_cls))))

color_seg = [[128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],
        [153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],
        [0, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70],[0, 60, 100],
        [0, 80, 100],[0, 0, 230],[119, 11, 32]]
color_seg = np.array(color_seg).astype(np.uint8)

def load_data(config):

    inputs = []
    gts = []

    output_dirs = {"input":[],"seg":[],"depth":[]}

    input_paths = sorted(glob.glob(os.path.join(config.input_path, '*.png')))[1::2]
    if config.gt_path != None:
        gt_paths = sorted(glob.glob(os.path.join(config.gt_path, '*.png')))[1::2]
    else:
        gt_paths = []

    model_name = config.model_name
    output_dir_base = os.path.join(config.output_dir, model_name)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    print("-> Predicting on {:d} eval images".format(len(input_paths)))
    
    print("-> {:d} gt images".format(len(gt_paths)))
    
    for image_path in input_paths:
        image = pil.open(image_path).convert('RGB')
        inputs.append(image)
        
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir_input = os.path.join(output_dir_base, "{}_input.png".format(output_name))
        output_dir_seg = os.path.join(output_dir_base, "{}_seg.png".format(output_name))
        output_dir_depth = os.path.join(output_dir_base, "{}_zepth.png".format(output_name))
        output_dirs["input"].append(output_dir_input)
        output_dirs["seg"].append(output_dir_seg)
        output_dirs["depth"].append(output_dir_depth)
        
    for image_path in gt_paths:
        image = pil.open(image_path)
        original_width, original_height = image.size
        gt = np.array(image,dtype=np.uint8)
        for c in void_cls:
            gt[gt==c] = 255
        for c in eval_cls:
            gt[gt==c] = class_map[c]
        gts.append(gt)
    
    return inputs, gts, output_dirs


def predict_seg(models, inputs, output_dirs, config):
    """Function to predict for a single image or folder of images
    """
    
    for key, model in models.items():
        model.eval()
        model.to('cuda')
    
    segs = []

    with torch.no_grad():

        for i, image in enumerate(inputs):

            original_width, original_height = image.size
            image.save(output_dirs["input"][i])
            # Load image and preprocess
            
            input_image = image.resize((config.width, config.height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to('cuda')

            if config.architecture == "share":
                seg_features = models["encoder"](input_image)
                seg = models["seg"](seg_features)[("output", 0)]
                depth = models["depth"](seg_features)[("output", 0)]

            if config.architecture == "byol":
                features = models["encoder"](input_image)
                representation = models["decoder"](features)
                representation = F.normalize(representation, dim=1, p=2)
                seg = F.softmax(models["head"](representation.detach(), features)[("output", 0)],dim=1)
                depth = None

            seg = torch.nn.functional.interpolate(
                seg, (original_height, original_width), mode="bilinear", align_corners=False)

            seg = seg.squeeze()
            seg = torch.argmax(seg, dim=0).cpu().numpy()

            segs.append(seg)

            if depth != None:
                disp_resized = torch.nn.functional.interpolate(depth, (original_height, original_width), mode="bilinear", align_corners=False)
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)
                im.save(output_dirs["depth"][i])

    return segs

def compute_metrics(segs, gts, output_dirs, h_match = False):
    num_cls = len(eval_cls)
    intersection = np.zeros((num_cls, num_cls))
    for seg, gt in zip(segs, gts):
        mask = (gt >= 0) & (gt < num_cls)
        hist = np.bincount(
            num_cls * seg[mask] + gt[mask], minlength=num_cls ** 2
        ).reshape(num_cls, num_cls)
        intersection += hist
    i = j = range(num_cls)
    if h_match:
        union = -intersection + intersection.sum(axis=1,keepdims=True) + intersection.sum(axis=0,keepdims=True)
        cost = -intersection/union
        cost[np.isnan(cost)] = -1
        i, j = linear_sum_assignment(cost)
    
    hist = intersection
    print(j)
    
    acc = hist[i,j].sum() / hist.sum()
    acc_cls = hist[i,j] / hist.sum(axis=0)[j]
    acc_cls = np.nanmean(acc_cls)
    iu = hist[i,j] / (hist.sum(axis=1) + hist.sum(axis=0)[j] - hist[i,j])
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=0)[j] / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    gt_labels = [labels[index] for index in j]
    cls_iu = dict(zip(gt_labels, iu))

    metrics = {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu
        }
    metrics.update(cls_iu)

    for s, seg in enumerate(segs):
        seg_img = color_seg[j][seg]
        seg_img = pil.fromarray(seg_img)
        seg_img.save(output_dirs["seg"][s])

    return metrics


def build_models(config):
    models = {}
    if config.architecture == "share":
        models["encoder"] = networks.ResnetEncoder(
            num_layers = config.ResX,
            pretrained = True)
        models["seg"] = networks.Decoder(
            out_channels = config.cls_num,
            scales = config.seg_scales,
            activate = 'softmax',
            in_channels = models["encoder"].num_ch_enc)
        models["depth"] = networks.Decoder(
            out_channels = 1,
            scales = config.seg_scales,
            activate = 'sigmoid',
            in_channels = models["encoder"].num_ch_enc)
            
        models_to_load = ["encoder","seg","depth"]

    if config.architecture == "byol":
        models["encoder"] = networks.ResnetEncoder(
            num_layers = config.ResX,
            pretrained = False)
        models["decoder"] = networks.Decoder_BYOL(
            in_channels = models["encoder"].num_ch_enc)
        models["head"] = networks.FullHead(
            out_channels=19, 
            activate = 'softmax', 
        in_channels=models["encoder"].num_ch_enc)
        models_to_load = ["encoder","decoder","head"]
        
    config.load_weights_folder = os.path.expanduser(config.load_weights_folder)
    assert os.path.isdir(config.load_weights_folder), \
        "Cannot find folder {}".format(config.load_weights_folder)
    print("loading model from folder {}".format(config.load_weights_folder))

    for n in models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(config.load_weights_folder, "{}.pth".format(n))
        model_dict = models[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[n].load_state_dict(model_dict)
        
    return models


def model_eval(config, models = None):
    if models == None:
        models = build_models(config)
    inputs, gts, output_dirs = load_data(config)
    segs = predict_seg(models, inputs, output_dirs, config)
    metrics = compute_metrics(segs, gts, output_dirs, config.h_match)
    
    return metrics


