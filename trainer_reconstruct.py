# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import wandb

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.model_name + str("_") + str(self.config.depth_w) + str("_") + str(self.config.coordinate_w) + str("_") + str(self.config.rgb_w) + str("_")  + str(self.config.ncut_w)
        self.log_path = os.path.join(self.config.log_dir, self.model_name)
        self.log_key = ["reconstruct","seg","seg_coord","seg_rgb","seg_seg","loss"]

        # checking height and width are multiples of 32
        assert self.config.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.config.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda")
        
        self.frame_ids = [0,-1,1]
        self.num_scales = len(self.config.scales)
        self.num_input_frames = len(self.frame_ids)
        self.num_pose_frames = 2


        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

			
        #self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"] = networks.UNet_e(3,self.config.cls_num)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        #self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["reconstruct"] = networks.UNet_d(self.config.cls_num,3)
        self.models["reconstruct"].to(self.device)
        self.parameters_to_train += list(self.models["reconstruct"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.config.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.config.scheduler_step_size, 0.3)

        if self.config.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.config.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.config.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.config.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        img_ext = '.png'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.config.batch_size * self.config.num_epochs

        train_dataset = self.dataset(
            self.config.data_path, train_filenames, self.config.height, self.config.width,
            self.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.config.batch_size, True,
            num_workers=self.config.num_workers, pin_memory=False, drop_last=True)

        self.writers = SummaryWriter(os.path.join(self.log_path, "train"))

        self.ssim = SSIM()
        self.ssim.to(self.device)


        print("Using split:\n  ", self.config.split)
        print("There are {:d} training items\n".format(
            len(train_dataset)))

        #self.save_config()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.config.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.config.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        
        loss_avg = {}
        for k in self.log_key:
          loss_avg[k] = torch.tensor(0)

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            
            for k in self.log_key:
              loss_avg[k] = loss_avg[k]+losses[k]
            
            
            time_to_log = batch_idx % self.config.log_frequency == 0 and batch_idx > 0 

            if time_to_log:
                self.log_time(batch_idx, duration, loss_avg)
                self.log("train", inputs, outputs, loss_avg)
                for k in self.log_key:
                  loss_avg[k] = torch.tensor(0)

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        outputs={}
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["reconstruct"](features)
        outputs[("seg", 0, 0)] = features

        #self.generate_images_pred(inputs, outputs)

        #for key in outputs.keys():
          #print(key)
          #print(outputs[key].shape)

        #print(asx)


        losses = {"reconstruct":torch.tensor(0),"seg_coord":torch.tensor(0),"seg_rgb":torch.tensor(0),"seg_seg":torch.tensor(0),"loss":torch.tensor(0)}

        losses["reconstruct"] = self.compute_reconstruction_loss(inputs, outputs)
        #losses = self.compute_reconstruction_losses(inputs,outputs)
        losses["seg_coord"], losses["seg_rgb"], losses["seg_seg"] = self.compute_loss_seg(inputs, outputs)
        losses["seg"] = self.config.coordinate_w*losses["seg_coord"]+self.config.feature_w*losses["seg_seg"]+self.config.rgb_w*losses["seg_rgb"]
        losses["seg"] = self.config.ncut_w*losses["seg"]/(self.config.coordinate_w+self.config.rgb_w+self.config.feature_w)
        #losses["contrastive"] = self.compute_loss_contrastive(inputs, outputs)
        #print(kk)
        losses["loss"] = losses["reconstruct"] + losses["seg"]# + losses["contrastive"]
        return outputs, losses


                        
    def compute_reconstruction_loss(self, inputs, outputs):
        total_loss = 0
        for scale in self.config.scales:
            loss = self.compute_reprojection_loss(outputs[("disp", scale)], inputs[("color_aug", 0, scale)])
            total_loss += loss.mean()
        total_loss /= self.num_scales
        #losses["loss"] = total_loss
        return total_loss
        
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_loss_seg(self, inputs, outputs):

        loss_seg_rgb = torch.tensor(0)
        loss_seg_coord = torch.tensor(0)
        loss_seg_seg = torch.tensor(0)
        seg = outputs[("seg", 0, 0)]

        pool = nn.AdaptiveAvgPool2d((48,160))
        seg = pool(seg)
        seg_sub = seg.flatten(2).unsqueeze(-1)
        #avg_p = seg_sub.mean(axis=(2,3))
        seg_sub_t = seg_sub.transpose(2,3)
        #seg_sub_matrix = torch.matmul(seg_sub, seg_sub_t)

        if self.config.feature_w >= 0:
          seg_feat = seg_sub.flatten(2).transpose(1,2)
          distance_seg = torch.cdist(seg_feat,seg_feat).unsqueeze(1)
          denominator = torch.matmul(distance_seg, seg_sub)
          numerator = torch.matmul(seg_sub_t, denominator).sum(dim=[2,3])
          denominator = denominator.sum(dim=[2,3])
          loss_seg_seg = (numerator/denominator).mean()

        if self.config.rgb_w >= 0:
          rgb = inputs["color_aug", 0, 0]
          rgb_feat = pool(rgb)
          rgb_feat = rgb_feat.flatten(2).transpose(1,2)
          distance_rgb = torch.cdist(rgb_feat,rgb_feat).unsqueeze(1)
          denominator = torch.matmul(distance_rgb, seg_sub)
          numerator = torch.matmul(seg_sub_t, denominator).sum(dim=[2,3])
          denominator = denominator.sum(dim=[2,3])
          loss_seg_rgb = (numerator/denominator).mean()

        if self.config.coordinate_w >= 0:
          x_range = torch.linspace(1, seg.shape[-1], seg.shape[-1], device=seg.device)
          y_range = torch.linspace(1, seg.shape[-2], seg.shape[-2], device=seg.device)
          y, x = torch.meshgrid(y_range, x_range)
          y = y.expand([seg_sub.shape[0], 1, -1, -1])
          x = x.expand([seg_sub.shape[0], 1, -1, -1])
          coord_feat = torch.cat([x, y], 1)
          coord_feat = coord_feat.flatten(2).transpose(1,2)
          distance_coord = torch.cdist(coord_feat,coord_feat).unsqueeze(1)
          denominator = torch.matmul(distance_coord, seg_sub)
          numerator = torch.matmul(seg_sub_t, denominator).sum(dim=[2,3])
          denominator = denominator.sum(dim=[2,3])
          loss_seg_coord = (numerator/denominator).mean()

        
        return loss_seg_coord, loss_seg_rgb, loss_seg_seg
        #return self.config.depth_w*loss_seg_depth+self.config.coordinate_w*loss_seg_coord+self.config.feature_w*loss_seg_seg+self.config.rgb_w*loss_seg_rgb


    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """


        for k in self.log_key:
          losses[k] = losses[k].cpu().data/self.config.log_frequency

        samples_per_sec = self.config.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss"],
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

        print_string = "reconstruct: {:.5f} | seg_coord: {:.10f} | seg_rgb: {:.10f} | seg_seg: {:.10f} |"
        print(print_string.format(losses["reconstruct"], losses["seg_coord"], losses["seg_rgb"], losses["seg_seg"]))
        wandb.log(losses)
        wandb.log({"epoch":self.epoch})

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        for l, v in losses.items():
            self.writers.add_scalar("{}".format(l), v, self.step)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.config.load_weights_folder = os.path.expanduser(self.config.load_weights_folder)

        assert os.path.isdir(self.config.load_weights_folder), \
            "Cannot find folder {}".format(self.config.load_weights_folder)
        print("loading model from folder {}".format(self.config.load_weights_folder))

        for n in self.config.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.config.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        #optimizer_load_path = os.path.join(self.config.load_weights_folder, "adam.pth")
        #if os.path.isfile(optimizer_load_path):
        #    print("Loading Adam weights")
        #    optimizer_dict = torch.load(optimizer_load_path)
        #    self.model_optimizer.load_state_dict(optimizer_dict)
        #else:
        #    print("Cannot find Adam weights so Adam is randomly initialized