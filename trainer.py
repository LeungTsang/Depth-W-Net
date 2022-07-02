# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import wandb
from seg_eval import model_eval

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import json

from utils import *
from kitti_utils import *
from layers import *
from seg_eval import model_eval

import datasets
import networks
from IPython import embed

color_seg = [[128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],
        [153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],
        [0, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70],[0, 60, 100],
        [0, 80, 100],[0, 0, 230],[119, 11, 32],[0,0,0]]
color_seg = np.array(color_seg).astype(np.uint8)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.model_name# + str("_") + str(self.config.depth_w) + str("_") + str(self.config.coordinate_w) + str("_") + str(self.config.rgb_w) + str("_")  + str(self.config.ncut_w)
        self.log_path = os.path.join(self.config.log_dir, self.model_name)
        self.log_key = ["depth","seg","cross_img","seg_nc","seg_depth","seg_coord","seg_rgb","seg_angle","loss"]
        self.loss_avg = {}
        for k in self.log_key:
          self.loss_avg[k] = torch.tensor(0)

        # checking height and width are multiples of 32
        assert self.config.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.config.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_t = {} 
        self.parameters_to_train = []

        self.device = torch.device("cuda")

        self.eval_cls = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
        
        self.frame_ids = [0,-1,1]
        self.num_scales = len(self.config.depth_scales)
        self.num_input_frames = len(self.frame_ids)
        self.num_pose_frames = 2


        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

        self.segloss = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.klloss = torch.nn.KLDivLoss()

        if self.config.kd:
            self.models_t["encoder"] = networks.ResnetEncoder(
                18, False)
            
            self.models_t["depth"] = networks.DepthDecoder(
                self.models_t["encoder"].num_ch_enc)

            encoder_path = os.path.join(self.config.load_weights_folder_t, "encoder.pth")
            decoder_path = os.path.join(self.config.load_weights_folder_t, "depth.pth")

            encoder_dict = torch.load(encoder_path)
            model_dict = self.models_t["encoder"].state_dict()
            self.models_t["encoder"].load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            self.models_t["depth"].load_state_dict(torch.load(decoder_path))

            self.models_t["encoder"].to(self.device)
            self.models_t["depth"].to(self.device)

            self.models_t["encoder"].eval()
            self.models_t["depth"].eval()

        if self.config.architecture == "share":

            self.models["encoder"] = networks.ResnetEncoder(
                num_layers = self.config.ResX,
                pretrained = False)
            self.models["seg"] = networks.Decoder(
                out_channels = self.config.cls_num,
                scales = self.config.seg_scales,
                activate = 'softmax',
                in_channels = self.models["encoder"].num_ch_enc)
            self.models["depth"] = networks.Decoder(
                out_channels = 1,
                scales = self.config.depth_scales,
                activate = 'sigmoid',
                in_channels = self.models["encoder"].num_ch_enc)
                
        if self.config.architecture == "concat":
            self.models["seg_encoder"] = networks.ResnetEncoder(
                num_layers = self.config.ResX,
                pretrained = False)
            self.models["seg"] = networks.Decoder(
                out_channels = self.config.cls_num,
                scales = self.config.seg_scales,
                activate = 'softmax',
                in_channels = self.models["seg_encoder"].num_ch_enc)
            self.models["depth_encoder"] = networks.ResnetEncoder(
                num_layers = self.config.ResX,
                pretrained = False,
                num_input_ch = self.config.cls_num)
            self.models["depth"] = networks.Decoder(
                out_channels = 1,
                scales = self.config.depth_scales,
                activate = 'sigmoid',
                in_channels = self.models["depth_encoder"].num_ch_enc)
        
        if self.config.architecture == "byol":
            self.models["encoder"] = networks.ResnetEncoder(
                num_layers = self.config.ResX,
                pretrained = False)
            self.models["decoder"] = networks.Decoder_BYOL(
                in_channels = self.models["encoder"].num_ch_enc)

            self.models["projector"] = networks.Block(
                in_channels = 128)

            self.models["predictor"] = networks.Block(
                in_channels = 128)
            #self.models["target_encoder"] = networks.ResnetEncoder(
            #    num_layers = self.config.ResX,
            #    pretrained = False)
            #self.models["target_decoder"] = networks.Decoder(
            #    name = "target",
            #    in_channels = self.models["depth_encoder"].num_ch_enc)

        if "fine_tune" or "linear_eval" in self.config.step:

            self.models["head"] = networks.FullHead(
                out_channels=19, 
                activate = 'softmax', 
                in_channels=self.models["encoder"].num_ch_enc)

        if "depth" in self.config.step and self.config.kd == False:
            
            self.models["pose_encoder"] = networks.ResnetEncoder(
                num_layers = 18,
                pretrained = True,
                num_input_ch=6)

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

        for key, model in self.models.items():
            model.to(self.device)

        if "contrastive" in self.config.step:
            models_to_train = ["encoder","decoder","projector"]
            for key, model in self.models.items():
                if key in models_to_train:
                    self.parameters_to_train += list(model.parameters())
                else:
                    model.eval()
            params = [{"params":self.parameters_to_train},{"params":self.models["predictor"].parameters(),"lr":10*self.config.learning_rate}]
        if "fine_tune" in self.config.step:
            models_to_train = ["encoder","decoder","head"]
            for key, model in self.models.items():
                if key in models_to_train:
                    self.parameters_to_train += list(model.parameters())
                else:
                    model.eval()
            params = [{"params":self.parameters_to_train}]
        if "linear_eval" in self.config.step:
            models_to_train = ["head"]
            for key, model in self.models.items():
                if key in models_to_train:
                    self.parameters_to_train += list(model.parameters())
                else:
                    model.eval()
            params = [{"params":self.parameters_to_train}]

        self.model_optimizer = optim.Adam(params, self.config.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.config.scheduler_step_size, 0.5)

        if self.config.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.config.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                  "kitti_odom": datasets.KITTIOdomDataset,
                  "kitti_semantic": datasets.KITTISemanticDataset}
        print(self.config.dataset)
        self.dataset = datasets_dict[self.config.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.config.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        img_ext = '.png'

        #num_train_samples = len(train_filenames)
        #self.num_total_steps = num_train_samples // self.config.batch_size * self.config.num_epochs

        train_dataset = self.dataset(
            self.config.data_path, train_filenames, self.config.height, self.config.width,
            self.frame_ids, 4, semantic_path=self.config.semantic_path, distance_path=self.config.distance_path, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.config.batch_size, True,
            num_workers=self.config.num_workers, pin_memory=False, drop_last=True)

        num_train_samples = train_dataset.__len__()
        self.num_total_steps = num_train_samples // self.config.batch_size * self.config.num_epochs

        self.writers = SummaryWriter(os.path.join(self.log_path, "train"))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.config.depth_scales:
            h = self.config.height // (2 ** scale)
            w = self.config.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.config.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.config.batch_size, h, w)
            self.project_3d[scale].to(self.device)

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
                self.save_model(self.epoch)

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        #self.set_eval()
        #metrics = model_eval(self.config, self.models)
        #print(metrics)
        #wandb.log(metrics)

        self.model_lr_scheduler.step()

        #print("Training")

        
        #loss_avg = {}
        #for k in self.log_key:
        #  loss_avg[k] = torch.tensor(0)

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            losses = self.process_batch(inputs)

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            
            for k in self.log_key:
              self.loss_avg[k] = self.loss_avg[k]+losses[k]
            
            time_to_log = (self.step+1) % self.config.log_frequency == 0
            if time_to_log:
                self.log_time(batch_idx, duration, self.loss_avg)
                self.log("train", inputs, self.loss_avg)
                for k in self.log_key:
                  self.loss_avg[k] = torch.tensor(0)
                #if self.step%1000 == 0:
                #self.save_model(self.step)
                #metrics = model_eval(self.config, self.models)
                #print(metrics)
                #wandb.log(metrics)
                #self.set_train()
            self.step += 1

        

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        #outputs_depth={}

        losses = {"depth":torch.tensor(0),"seg":torch.tensor(0),"seg_nc":torch.tensor(0),"cross_img":torch.tensor(0),"seg_depth":torch.tensor(0),"seg_coord":torch.tensor(0),"seg_rgb":torch.tensor(0),"seg_angle":torch.tensor(0),"loss":torch.tensor(0)}

        if "depth" in self.config.step:

            if self.config.architecture == "share":
                seg_features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs_depth = self.models["depth"](seg_features)
                outputs_seg = self.models["seg"](seg_features)

            if self.config.architecture == "concat":
                seg_features = self.models["seg_encoder"](inputs["color_aug", 0, 0])
                outputs_seg = self.models["seg"](seg_features)
                depth_features = self.models["depth_encoder"](outputs_seg["output", 0])
                outputs_depth = self.models["depth"](seg_features)


            if self.config.kd:
                outputs_t = self.models_t["depth"](self.models_t["encoder"](inputs["color_aug", 0, 0]))
                for s in self.config.depth_scales:
                    p = torch.cat((outputs_depth[("output", s)],1-outputs_depth[("output", s)]),dim=1)
                    t = torch.cat((outputs_t[("output", s)],1-outputs_t[("output", s)]),dim=1)
                    losses["depth"] = losses["depth"] + self.klloss(p.log(),t)

            else:
                outputs_depth.update(self.predict_poses(inputs))    
                self.generate_images_pred(inputs, outputs_depth)
                losses["depth"] = self.compute_losses(inputs, outputs_depth)
                
            #losses["seg_depth"], losses["seg_coord"], losses["seg_rgb"], losses["seg_angle"] = self.compute_loss_seg_nc(inputs, outputs_seg[("output", 0)] ,outputs_depth[("output", 0)])
            #losses["seg_nc"] = self.config.depth_w*losses["seg_depth"]+self.config.coordinate_w*losses["seg_coord"]+self.config.angle_w*losses["seg_angle"]+self.config.rgb_w*losses["seg_rgb"]
            #losses["seg_nc"] = self.config.ncut_w*losses["seg_nc"]/(self.config.depth_w+self.config.coordinate_w+self.config.rgb_w+self.config.angle_w)
            losses["seg_nc"] = 0.002*self.compute_loss_seg_nc_with_dis(outputs_seg ,inputs[("distance")])

            loss_total = losses["depth"] + losses["seg_nc"]
            self.model_optimizer.zero_grad()
            loss_total.backward()
            self.model_optimizer.step()

            if self.config.train_output != None and self.step%100 == 0:
                seg = outputs_seg[("output", 0)][0]
                seg = torch.argmax(seg, dim=0).cpu().numpy()
                seg_img = color_seg[seg]
                seg_img = pil.fromarray(seg_img)
                seg_img.save(self.config.train_output+str(self.step)+"seg.png")


        if "seg" in self.config.step:

            if self.config.architecture == "share":
                seg_features = self.models["encoder"](inputs["semantic_aug", 0, 0])
                outputs_seg = self.models["seg"](seg_features)
                #outputs_depth = self.models["seg"](seg_features)
                

            losses["seg"] = self.compute_loss_seg(inputs["semantic_gt", 0, 0].squeeze(1), outputs_seg)
            self.model_optimizer.zero_grad()
            losses["seg"].backward()
            self.model_optimizer.step()

            #if self.config.architecture = "duplex_UNet":

        if "contrastive" in self.config.step:
            if self.config.architecture == "byol":
                representation = self.models["decoder"](self.models["encoder"](inputs["semantic_aug", 0, 0]))
                projection = self.models["projector"](representation)
                prediction = self.models["predictor"](projection)
                #target_projection = online_prediction
                with torch.no_grad():
                    representation = self.models["decoder"](self.models["encoder"](inputs["semantic_aug", 0, 0]))
                    projection = self.models["projector"](representation)
                    #_,target_projection,_ = self.models["target_decoder"](self.models["target_encoder"](inputs["semantic_aug", 0, 0]))
                #print(online_prediction)
                #print(target_projection)
                #print("----")
            losses["seg"] = self.compute_loss_contrastive(prediction,projection.detach(),inputs[("distance")])
            self.model_optimizer.zero_grad()
            losses["seg"].backward()
            self.model_optimizer.step()
            #print(losses["seg"])
            if self.config.train_output != None and self.step%10 == 0:
                representation = representation.detach()
                representation = F.normalize(representation, dim=1, p=2)
                np.save(self.config.train_output+str(self.step)+"rep.npy",representation.cpu().numpy())
                img = (inputs["semantic_aug", 0, 0][0].cpu().transpose(0,1).transpose(1,2).numpy()*255).astype(np.uint8)
                img = pil.fromarray(img)
                img.save(self.config.train_output+str(self.step)+"img.png")
        if "linear_eval" in self.config.step:
            if self.config.architecture == "byol":
                with torch.no_grad():
                    features = self.models["encoder"](inputs["semantic_aug", 0, 0])
                    representation = self.models["decoder"](features)
                    representation = F.normalize(representation, dim=1, p=2)
                outputs_seg = self.models["head"](representation.detach(), features)

            losses["seg"] = self.compute_loss_seg(inputs["semantic_gt", 0, 0].squeeze(1), outputs_seg)
            self.model_optimizer.zero_grad()
            losses["seg"].backward()
            self.model_optimizer.step()
            if self.config.train_output != None and self.step%100 == 0:
                seg = outputs_seg[("output", 0)][0]
                seg = torch.argmax(seg, dim=0).cpu().numpy()
                seg_img = color_seg[seg]
                seg_img = pil.fromarray(seg_img)
                seg_img.save(self.config.train_output+str(self.step)+"seg.png")

                img = (inputs["semantic_aug", 0, 0][0].cpu().transpose(0,1).transpose(1,2).numpy()*255).astype(np.uint8)
                img = pil.fromarray(img)
                img.save(self.config.train_output+str(self.step)+"img.png")

        if "fine_tune" in self.config.step:
            if self.config.architecture == "byol":
                features = self.models["encoder"](inputs["semantic_aug", 0, 0])
                representation = self.models["decoder"](features)
                representation = F.normalize(representation, dim=1, p=2)
                outputs_seg = self.models["head"](representation.detach(),features)

            losses["seg"] = self.compute_loss_seg(inputs["semantic_gt", 0, 0].squeeze(1), outputs_seg)
            self.model_optimizer.zero_grad()
            losses["seg"].backward()
            self.model_optimizer.step()
            if self.config.train_output != None and self.step%100 == 0:
                seg = outputs_seg[("output", 0)][0]
                seg = torch.argmax(seg, dim=0).cpu().numpy()
                seg_img = color_seg[seg]
                seg_img = pil.fromarray(seg_img)
                seg_img.save(self.config.train_output+str(self.step)+"seg.png")

                img = (inputs["semantic_aug", 0, 0][0].cpu().transpose(0,1).transpose(1,2).numpy()*255).astype(np.uint8)
                img = pil.fromarray(img)
                img.save(self.config.train_output+str(self.step)+"img.png")
                #np.save(self.config.train_output+str(self.step)+"rep.npy",representation.detach().cpu().numpy())

        #losses["loss"] = losses["depth"] + losses["seg"] + losses["seg_nc"]+losses["cross_img"]# + losses["contrastive"]

        return losses


    def compute_loss_contrastive(self, prediction, projection, distance):
        #pool = torch.nn.AdaptiveAvgPool2d((24,80))
        prediction = prediction.flatten(2)
        prediction = F.normalize(prediction, dim=1, p=2)
        prediction = prediction.unsqueeze(-1).expand(prediction.shape[0],prediction.shape[1],prediction.shape[2],prediction.shape[2])
        projection = projection.flatten(2)
        projection = F.normalize(projection, dim=1, p=2)
        projection = projection.unsqueeze(-1).expand(projection.shape[0],projection.shape[1],projection.shape[2],projection.shape[2]).transpose(2,3)
        #print(prediction.shape)
        #print(projection.shape)
        #print(distance.shape)
        #print(torch.norm(prediction,dim=1, keepdim=True).shape)
        d = 1-(prediction*projection).sum(1,keepdim=True)
        #print(d.shape)
        #print(distance.shape)
        
        #distance[distance<5*distance.mean()]=0
        #distance[distance==0] = -1
        #d = distance.sum(-1,keepdim=True)
        d_norm = distance.sum(-1,keepdim=True)
        d_norm[d_norm==0] = 1
        loss = d*distance/d_norm

        
        #print(distance.shape)
        #distance = distance

        return loss.sum()
        

    def chamfer_distance(self,f1,f2,s1,s2):
        pool = torch.nn.AdaptiveAvgPool2d((32,120))
        f1 = pool(f1).flatten(2).transpose(1,2)
        f2 = pool(f2).flatten(2).transpose(1,2)
        s1 = pool(s1).flatten(2).unsqueeze(-1)>0.5
        s2 = pool(s2).flatten(2).unsqueeze(-1).transpose(2,3)>0.5
        #print(f1.shape)
        #print(s1.shape)
        distance_seg = (torch.cdist(f1,f2).unsqueeze(1)/s1)/s2
        #print(distance_seg)
        d = torch.min(distance_seg,dim=2).values/s2.sum() + torch.min(distance_seg,dim=3).values/s1.sum()
        #print(d.sum())
        #print(D.shape)
        d = torch.where(torch.isinf(d), torch.full_like(d, 0), d)
        d = torch.where(torch.isnan(d), torch.full_like(d, 0), d)
        return d.sum()

    def compute_loss_seg(self, gt, outputs):
        loss = 0
        for scale in self.config.seg_scales:
          divisor = (2 ** scale)
          #seg = outputs[("output", scale)]
          seg = F.interpolate(
                outputs[("output", scale)], [self.config.height, self.config.width], mode="bilinear", align_corners=False)
          #gt = gt[:,::8,::8]
          loss += self.segloss(seg, gt)
          
        return self.segloss(seg, gt)

    def predict_poses(self, inputs):
        outputs_pose = {}

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

        for f_i in self.frame_ids[1:]:
            if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    

                axisangle, translation = self.models["pose"](pose_inputs)
                outputs_pose[("axisangle", 0, f_i)] = axisangle
                outputs_pose[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                outputs_pose[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs_pose
    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        #seg = torch.argmax(features, dim=1).unsqueeze(1)

        for scale in self.config.depth_scales:
            divisor = (2 ** scale)

            disp = F.interpolate(
                    outputs[("output", scale)], [self.config.height, self.config.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, 0.1, 100)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                        

        
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_reconstruction_loss(self, inputs, outputs):
        #losses = {}
        total_loss = 0
        for scale in self.config.depth_scales:
            loss = self.compute_reprojection_loss(outputs[("output", scale)], inputs[("color_aug", 0, scale)]) 
            total_loss = total_loss + loss.mean()
        total_loss = total_loss/self.num_scales
        #losses["loss"] = total_loss
        return total_loss

    def compute_loss_seg_nc(self, inputs, seg, depth):

        loss_seg_depth = torch.tensor(0)
        loss_seg_rgb = torch.tensor(0)
        loss_seg_coord = torch.tensor(0)
        loss_seg_angle = torch.tensor(0)

        pool = nn.AdaptiveAvgPool2d((24,80))
        seg = pool(seg)
        seg_sub = seg.flatten(2).unsqueeze(-1)
        seg_sub_t = seg_sub.transpose(2,3)

        if self.config.rgb_w > 0:
          rgb = inputs["color_aug", 0, 0]
          rgb_feat = pool(rgb)
          rgb_feat = rgb_feat.flatten(2).transpose(1,2)
          distance_rgb = torch.cdist(rgb_feat,rgb_feat).unsqueeze(1)
          denominator = torch.matmul(distance_rgb, seg_sub)
          numerator = torch.matmul(seg_sub_t, denominator).sum(dim=[2,3])
          denominator = denominator.sum(dim=[2,3])
          loss_seg_rgb = (numerator/denominator).mean()

        if self.config.coordinate_w > 0:
          x_range = torch.linspace(1, seg.shape[-1], seg.shape[-1], device=seg.device)
          y_range = torch.linspace(1, seg.shape[-2], seg.shape[-2], device=seg.device)
          y, x = torch.meshgrid(y_range, x_range)
          y = y.expand([seg_sub.shape[0], 1, -1, -1])
          x = x.expand([seg_sub.shape[0], 1, -1, -1])
          #coord_feat = torch.cat([x, y], 1)
          coord_feat = torch.atan(y/torch.abs(x))
          coord_feat = coord_feat.flatten(2).transpose(1,2)
          distance_coord = torch.cdist(coord_feat,coord_feat).unsqueeze(1)
          denominator = torch.matmul(distance_coord, seg_sub)
          numerator = torch.matmul(seg_sub_t, denominator).sum(dim=[2,3])
          denominator = denominator.sum(dim=[2,3])
          loss_seg_coord = (numerator/denominator).mean()

        if self.config.angle_w > 0:
          x_range = torch.linspace(-1, 1, seg.shape[-1], device=seg.device)
          y_range = torch.linspace(-1, 1, seg.shape[-2], device=seg.device)
          y, x = torch.meshgrid(y_range, x_range)
          y = y.expand([seg_sub.shape[0], 1, -1, -1])
          x = x.expand([seg_sub.shape[0], 1, -1, -1])
          #coord_feat = torch.cat([x, y], 1)
          angle_feat = torch.atan(y/torch.abs(x))
          angle_feat = angle_feat.flatten(2).transpose(1,2)
          distance_angle = torch.cdist(angle_feat,angle_feat).unsqueeze(1)
          denominator = torch.matmul(distance_angle, seg_sub)
          numerator = torch.matmul(seg_sub_t, denominator).sum(dim=[2,3])
          denominator = denominator.sum(dim=[2,3])
          loss_seg_angle = (numerator/denominator).mean()

        if self.config.depth_w > 0:
          disp = depth
          depth_feat = pool(disp)
          depth_feat = depth_feat.flatten(2).transpose(1,2)
          distance_depth = torch.cdist(depth_feat,depth_feat).unsqueeze(1)

          denominator = torch.matmul(distance_depth, seg_sub)
          numerator = torch.matmul(seg_sub_t, denominator).sum(dim=[2,3])
          denominator = denominator.sum(dim=[2,3])
          loss_seg_depth = (numerator/denominator).mean()
        
        return loss_seg_depth, loss_seg_coord, loss_seg_rgb, loss_seg_angle

    def compute_loss_seg_nc_with_dis(self, seg, distance):
        seg = seg[("output", 0)]
        pool = nn.AdaptiveAvgPool2d((24,80))
        seg = pool(seg)
        seg_sub = seg.flatten(2).unsqueeze(-1)
        seg_sub_t = seg_sub.transpose(2,3)
        #print(distance.shape)
        #print(seg_sub.shape)
        denominator = torch.matmul(distance, seg_sub)
        numerator = torch.matmul(seg_sub_t, denominator).sum(dim=[2,3])
        denominator = denominator.sum(dim=[2,3])
        loss = (numerator/denominator).mean()

        return loss


    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        #losses = {}
        total_loss = 0

        for scale in self.config.depth_scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("output", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]

                #if self.config.train_output != None:
                    #pe = (torch.abs(pred-target).detach().mean(1,False).cpu().numpy()[0]*255).astype(np.uint8)
                    #print(pe.shape)
                    #pe = pil.fromarray(pe,'P')
                #plt.imshow(pe)
                    #pe.save(self.config.train_output+str(self.step)+"pe.png")


                repro_loss = self.compute_reprojection_loss(pred, target)
                reprojection_losses.append(repro_loss)

            reprojection_losses = torch.cat(reprojection_losses, 1)


            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                repro_loss = self.compute_reprojection_loss(pred, target)
                identity_reprojection_losses.append(repro_loss)

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            identity_reprojection_loss = identity_reprojection_losses


            reprojection_loss = reprojection_losses

            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += 1e-3 * smooth_loss / (2 ** scale)
            total_loss += loss
            #losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        #losses["loss"] = total_loss
        return total_loss

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

        print_string = "depth: {:.8f} | seg: {:.8f} | seg_nc: {:.8f} | cross_img: {:.8f} | seg_depth: {:.8f} | seg_coord: {:.8f} | seg_rgb: {:.8f} | seg_angle: {:.8f} |"
        print(print_string.format(losses["depth"], losses["seg"], losses["seg_nc"],losses["cross_img"], losses["seg_depth"], losses["seg_coord"], losses["seg_rgb"], losses["seg_angle"]))
        wandb.log(losses)
        wandb.log({"epoch":self.epoch})
        wandb.log({"step":self.epoch})

    def log(self, mode, inputs, losses):
        """Write an event to the tensorboard events file
        """
        for l, v in losses.items():
            self.writers.add_scalar("{}".format(l), v, self.step)

    def save_model(self,n):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(n))
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