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
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


from skimage.segmentation import mark_boundaries
from scipy.optimize import linear_sum_assignment

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

#labels = ['unlabeled','ego vehicle','rectification border','out of roi','static','dynamic','ground','road','sidewalk','parking','rail track',
#    'building','wall','fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain',
#    'sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','license plate']

labels = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain',
    'sky','person','rider','car','truck','bus','train','motorcycle','bicycle']

eval_cls = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]

color_seg = [[128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],
        [153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],
        [0, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70],[0, 60, 100],
        [0, 80, 100],[0, 0, 230],[119, 11, 32]]
color_seg = np.array(color_seg).astype(np.uint8)

def load_data(config):

    inputs = []
    output_dirs = {"seg":[],"depth":[],"input":[]}

    input_paths = glob.glob(os.path.join(config.input_path, '*.png'))
    
    print("-> Predicting on {:d} eval images".format(len(input_paths)))
    
    
    model_name = config.model_name + str("_") + str(config.depth_w) + str("_") + str(config.coordinate_w) + str("_") + str(config.rgb_w)  + str("_")  + str(config.ncut_w)
    log_path = os.path.join(config.log_dir, model_name)
    output_dir_base = os.path.join(log_path, "pred_seg")
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
            
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
        
        
    return inputs,  output_dirs


def predict_seg(seg_encoder, seg_decoder, depth_encoder, depth_decoder, inputs, output_dirs, config):
    """Function to predict for a single image or folder of images
    """
    seg_encoder.eval()
    seg_decoder.eval()
    depth_encoder.eval()
    depth_decoder.eval()
    
    with torch.no_grad():
        for i, image in enumerate(inputs):

            image.save(output_dirs["input"][i])
            # Load image and preprocess
            original_width, original_height = image.size
            input_image = image.resize((config.width, config.height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to('cuda')

            if config.share:
                features = seg_encoder(input_image)
                seg = seg_decoder(features)[("output", 0)]
                depth = depth_decoder(features)[("output", 0)]
            else:
                seg_features = seg_encoder(input_image)
                seg = seg_decoder(seg_features)[("output", 0)]
                depth_features = depth_encoder(seg)
                depth = depth_decoder(depth_features)[("output", 0)]

            seg = torch.nn.functional.interpolate(
                seg, (original_height, original_width), mode="bilinear", align_corners=False)

            seg_raw = seg.squeeze()
            seg = torch.argmax(seg_raw, dim=0, keepdim=True)
            seg = seg.squeeze(0).cpu().numpy()
            seg_image = color_seg[seg]
            im_seg = pil.fromarray(seg_image)
            im_seg.save(output_dirs["seg"][i])


            disp_resized = torch.nn.functional.interpolate(depth, (original_height, original_width), mode="bilinear", align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            im.save(output_dirs["depth"][i])
            
    return
    
        
def pred(seg_encoder, seg_decoder, depth_encoder, depth_decoder, config):
    inputs, output_dirs = load_data(config)
    predict_seg(seg_encoder, seg_decoder, depth_encoder, depth_decoder, inputs, output_dirs, config)
    
    return 
    
