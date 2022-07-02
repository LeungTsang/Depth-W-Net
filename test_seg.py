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

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_path', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument('--K', type=int,
                        help='num_classes', required=True)

    return parser.parse_args()

color_seg = [[203,222,94],[141,212,116],[10,124,29],[28,7,166],[249,186,190],[217,97,191],[244,168,57],[11,196,187],[43,59,164],[244,75,75],
[135,54,33],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[50,50,50],[168,149,213]]
color_seg = np.array(color_seg).astype(np.uint8)

def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join(args.model_path)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.UNet_e(3,args.K)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    encoder.load_state_dict(loaded_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.UNet_d(args.K,1)

    loaded_dict_dec = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict_dec)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    feed_width = 640
    feed_height = 192
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            image = pil.open(image_path).convert('RGB')
            original_width, original_height = image.size
            input_image = image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            seg = torch.nn.functional.interpolate(
                features, (original_height, original_width), mode="bilinear", align_corners=False)

            #plt.imshow(seg_image)
            #plt.savefig()



            # Saving numpy file
            #gb = Glasbey(base_palette="palettes/set1.txt", overwrite_base_palette=True, lightness_range=(10,100), hue_range=(10,100), chroma_range=(10,100), no_black=True)  # complicated example (demonstrate syntax)
            #gb = Glasbey(base_palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])  # base_palette can also be rgb-list
            #color_seg = gb.generate_palette(size=100)
            #color_seg = np.random.rand(100, 3)
            #color_seg  = (color_seg*255).astype(np.uint8)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            seg_map = seg.squeeze().cpu().numpy()
            seg = torch.argmax(seg.squeeze(), dim=0).cpu().numpy()
            seg_image = color_seg[seg]
            #plt.savefig("seg.png")
            #print(seg)
            #seg_image = (mark_boundaries(image,seg)* 255).astype(np.uint8)
            #print(seg_image)
            im_seg = pil.fromarray(seg_image)
            name_dest_im = os.path.join(output_directory, "{}_seg.jpeg".format(output_name))
            im_seg.save(name_dest_im)

            for i in range(seg_map.shape[0]):
              #heatmap = pil.fromarray(seg_map[i])
              plt.imshow(seg_map[i])
              name_dest_im = os.path.join(output_directory, ("{}_heatmap"+str(i)+".png").format(output_name))
              plt.savefig(name_dest_im)
              plt.close()

            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
