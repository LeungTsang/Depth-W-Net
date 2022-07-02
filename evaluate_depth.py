from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def build_models(config):
    models = {}
    if config.architecture == "single_UNet":
        models["seg_encoder"] = networks.UNet_left(3)
        models["seg"] = networks.UNet_right(config.cls_num,config.seg_scales)
        models_to_load = ["seg_encoder","seg"]
            
    if config.architecture == "double_UNet":
        models["encoder"] = networks.UNet_e(3,config.cls_num)
        models["depth"] = networks.UNet_d(config.cls_num,1)
        models_to_load = ["encoder","depth"]

    if config.architecture == "UNet_ResNet":
        models["encoder"] = networks.UNet_e(3,config.cls_num)
        models["depth_encoder"] = networks.ResnetEncoder(18, False, config.cls_num)
        models["depth"] = networks.DepthDecoder(models["depth_encoder"].num_ch_enc)
        models_to_load = ["encoder","depth_encoder","depth"]

    if config.architecture == "share_UNet":
        models["encoder"] = networks.UNet_left(3)
        models["depth"] = networks.UNet_right(1,config.scales,'sigmoid')
        models["seg"] = networks.UNet_right(config.cls_num,config.seg_scales)
        models_to_load = ["encoder","depth","seg"]
        
    if config.architecture == "sfcn_UNet":
        network_data = torch.load("./semi_monoseg/sfcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar")
        models["encoder"] = sfcn.models.__dict__[network_data['arch']](data = network_data).cuda()
        models["depth"] = networks.UNet_d(config.cls_num,1)
        models_to_load = ["depth"]
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
        
    for key, model in models.items():
        model.eval()
        model.to('cuda')
    return models

def evaluate(config):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    config.load_weights_folder = os.path.expanduser(config.load_weights_folder)

    assert os.path.isdir(config.load_weights_folder), \
        "Cannot find a folder at {}".format(config.load_weights_folder)

    print("-> Loading weights from {}".format(config.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, config.eval_split, "test_files.txt"))
    encoder_path = os.path.join(config.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(config.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(config.data_path, filenames,
                                       192, 640,
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=config.num_workers,
                            pin_memory=False, drop_last=False)


    architecture= "UNet_ResNet"

    models = build_models(config)

    pred_disps = []

        #print("-> Computing predictions with size {}x{}".format(
        #    encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in dataloader:
            input_image = data[("color", 0, 0)].cuda()
            print(input_image.shape)

            #if config.post_process:
                # Post-processed results require each image to have two forward passes
            input_image = torch.cat((input_image, torch.flip(input_image, [3])), 0)

            if config.architecture == "double_UNet":
                seg = models["encoder"](input_image)[("output", 0)]
                depth = models["depth"](seg)[("output", 0)]
                #depth = None

            if config.architecture == "UNet_ResNet":
                seg = models["encoder"](input_image)[("output", 0)]
                depth = models["depth"](models["depth_encoder"](seg))[("output", 0)]
                
            if config.architecture == "share_UNet":
                seg_features = models["encoder"](input_image)
                seg = models["seg"](seg_features)[("output", 0)]
                depth = models["depth"](seg_features)[("output", 0)]

            if config.architecture == "sfcn_UNet":
                seg = models["encoder"](input_image)
                depth = models["depth"](seg)[("output", 0)]

            pred_disp, _ = disp_to_depth(depth, MIN_DEPTH, MAX_DEPTH)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    if config.eval_split == 'benchmark':
        save_dir = os.path.join(config.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, config.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if config.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= 1
        #if not opt.disable_median_scaling:
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    #if not config.disable_median_scaling:
    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    return mean_errors

