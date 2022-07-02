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

color_seg = [[203,222,94],[141,212,116],[10,124,29],[28,7,166],[249,186,190],[217,97,191],[244,168,57],[11,196,187],[43,59,164],[244,75,75],
[135,54,33],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[50,50,50],[168,149,213]]
color_seg = np.array(color_seg).astype(np.uint8)

def load_data(config):

    inputs = []
    output_dirs = {"seg":[],"reconstruct":[],"input":[]}

    input_paths = glob.glob(os.path.join(config.input_path, '*0.png'))
    
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
        output_dir_reconstruct = os.path.join(output_dir_base, "{}_reconstruct.png".format(output_name))
        output_dirs["input"].append(output_dir_input)
        output_dirs["seg"].append(output_dir_seg)
        output_dirs["reconstruct"].append(output_dir_reconstruct)
        
        
    return inputs,  output_dirs


def predict_seg(seg_model, reconstruct_decoder, inputs, output_dirs, config):
    """Function to predict for a single image or folder of images
    """
    seg_model.eval()
    
    with torch.no_grad():
        for i, image in enumerate(inputs):

            image.save(output_dirs["input"][i])
            # Load image and preprocess
            original_width, original_height = image.size
            input_image = image.resize((config.width, config.height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to('cuda')
            features = seg_model(input_image)
            reconstruct = reconstruct_decoder(features)[("disp", 0)]

            seg = torch.nn.functional.interpolate(
                features, (original_height, original_width), mode="bilinear", align_corners=False)
            #print(seg.shape)

            seg_raw = seg.squeeze()
            #print(seg_raw.shape)
            
            seg = torch.argmax(seg_raw, dim=0, keepdim=True)
            #print(seg.shape)
            
            #print(seg.shape)
            seg = seg.squeeze(0).cpu().numpy()
            #print(seg.shape)
            seg_image = color_seg[seg]
            #print(seg_image.shape)
            im_seg = pil.fromarray(seg_image)
            #name_dest_im = os.path.join(output_directory, "{}_seg.png".format(output_name))
            im_seg.save(output_dirs["seg"][i])


            reconstruct_resized = torch.nn.functional.interpolate(reconstruct, (original_height, original_width), mode="bilinear", align_corners=False)
            reconstruct_resized_np = (reconstruct_resized.squeeze().permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            #print(reconstruct_resized_np.shape)
            #vmax = np.percentile(disp_resized_np, 95)
            #normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            #mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            #colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(reconstruct_resized_np)

            im.save(output_dirs["reconstruct"][i])
            
    return
    
        
def pred(model, reconstruct_decoder, config):
    inputs, output_dirs = load_data(config)
    predict_seg(model, reconstruct_decoder, inputs, output_dirs, config)
    
    return 
    
