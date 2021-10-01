from __future__ import absolute_import, division, print_function

import numpy as np
import time
import sys

import torch
import torchgeometry as tgm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import camera
import datasets
import networks
from model_sal import SODModel
from salience_loss import EdgeSaliencyLoss
import attention_net
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.opt.model_name = self.opt.model_name + str(self.opt.defocused_scale) \
                              + "_" + str(self.opt.wac_scale) 
            
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        

        self.models = {}
        self.depth_params = []
        self.att_params = []
        
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.depth_params += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.depth_params += list(self.models["depth"].parameters())


        if self.opt.deformable:

            '''
            self.models["saliency"] = SODModel()
            chkpt = torch.load("../best_sal.pth")
            self.models["saliency"].load_state_dict(chkpt["model"])
            self.models["saliency"].to(self.device)
            self.att_params += list(self.models["saliency"].parameters())
            self.deformable_criterion = EdgeSaliencyLoss(device=self.device)
            '''

            self.models["saliency"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, [0],
                num_output_channels=1)
            self.models["saliency"].to(self.device)
            self.att_params += list(self.models["saliency"].parameters())

            path = "~/tmp/attention/oracleC_weight_old_scratch_"+str(self.opt.defocused_scale)+ \
                   "_"+str(self.opt.wac_scale)+"/models/weights_12/"
            custom = {'path':path,
                      'model_names':['encoder', 'depth'],
                      'models_to_load':['encoder', 'depth']}
            self.load_model(custom)
            

        saccade_total = camera.get_resolutions(self.opt)["saccade_total"]
        self.num  = torch.tensor(saccade_total[0] * saccade_total[1]).int()

        if self.opt.oracleC:

            '''
            # fixed wac depth 
            path = "~/tmp/wac_"+str(self.opt.defocused_scale)+ \
                   "_"+str(self.opt.wac_scale)+"/models/weights_19/"
            self.models["encoder_wac"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder_wac"].to(self.device)

            self.models["depth_wac"] = networks.DepthDecoder(
                self.models["encoder_wac"].num_ch_enc, self.opt.scales)
            self.models["depth_wac"].to(self.device)
            custom = {'path':path,
                      'model_names':['encoder', 'depth'],
                      'models_to_load':['encoder_wac', 'depth_wac']}
            self.load_model(custom)

            # fixed focus depth
            path = "~/tmp/focused/models/weights_19/"
            self.models["encoder_focused"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder_focused"].to(self.device)

            self.models["depth_focused"] = networks.DepthDecoder(
                self.models["encoder_focused"].num_ch_enc, self.opt.scales)
            self.models["depth_focused"].to(self.device)
            custom = {'path':path,
                      'model_names':['encoder', 'depth'],
                      'models_to_load':['encoder_focused', 'depth_focused']}
            self.load_model(custom)
            '''                
                
        if self.opt.e2e:

            # trainable oracle depth 
            path = "~/tmp/attention/oracleC0_"+ \
                   str(self.opt.defocused_scale)+ \
                   "_"+str(self.opt.wac_scale)+"_5_LR0.0001/models/weights_4/"
            custom = {'path':path,
                      'models_to_load':['encoder', 'depth'],
                      'model_names':['encoder', 'depth']}
            self.load_model(custom)
            
            # fixed wac depth 
            path = "~/tmp/wac_"+str(self.opt.defocused_scale)+ \
                   "_"+str(self.opt.wac_scale)+"/models/weights_19/"
            self.models["encoder_wac"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder_wac"].to(self.device)

            self.models["depth_wac"] = networks.DepthDecoder(
                self.models["encoder_wac"].num_ch_enc, self.opt.scales)
            self.models["depth_wac"].to(self.device)
            custom = {'path':path,
                      'model_names':['encoder', 'depth'],
                      'models_to_load':['encoder_wac', 'depth_wac']}
            self.load_model(custom)
                        
            
            # fixed attention decoder 
            path = "~/tmp/attention/deform_decoder_"+ \
                   str(self.opt.defocused_scale)+"_"+str(self.opt.wac_scale) \
                   +"/models/weights_9/"
            self.models["saliency"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, [0],
                num_output_channels=1)
            self.models["saliency"].to(self.device)
            #self.att_params += list(self.models["saliency"].parameters())
            custom = {'path':path,
                      'model_names':['saliency'],
                      'models_to_load':['saliency']}
            self.load_model(custom)
            
        self.saccade_total = camera.get_resolutions(self.opt)["saccade_total"]
            
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"
            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        if not self.opt.deformable:
            self.depth_optim = optim.Adam(self.depth_params, self.opt.depth_lr)
            self.depth_lr_sched = optim.lr_scheduler.StepLR(
                self.depth_optim, self.opt.depth_sched_step, 0.1)
        if self.opt.deformable or self.opt.finetune_oracleC_and_attention:
            self.att_optim = optim.Adam(self.att_params, self.opt.att_lr)
            self.att_lr_sched = optim.lr_scheduler.StepLR(
                self.att_optim, self.opt.att_sched_step, 0.1)
        
        if self.opt.load_weights_folder is not None:
            self.load_model()
                        
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # load data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]
            
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        if self.opt.no_shuffle:
            shuffle = False
        else:
            shuffle = True

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, opt=self.opt, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=shuffle,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, opt=self.opt, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=shuffle,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        suffix = str(np.random.rand(1,1))
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode+'_'+self.opt.model_name))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        
        #print("There are {:d} training items and {:d} validation items\n".format(
        #    len(train_dataset), len(val_dataset)))

        # remove camera class from options so we can save options as json
        self.opt.camera = None
        self.save_opts()

        
    def set_train(self):
        """Convert all models to training mode
        """

        if self.opt.e2e:
            self.models["saliency"].eval()
            self.models["encoder_wac"].eval()
            self.models["depth_wac"].eval()
            self.models["encoder"].train()
            self.models["depth"].train()

            
        elif self.opt.deformable:
            self.models["saliency"].train()
            self.models["encoder"].eval()
            self.models["depth"].eval()
            
        elif self.opt.oracleC:
            #self.models["encoder_wac"].eval()
            #self.models["depth_wac"].eval()
            #self.models["encoder_focused"].eval()
            #self.models["depth_focused"].eval()
            self.models["encoder"].train()
            self.models["depth"].train()
            if self.opt.finetune_oracleC_and_attention:
                self.models["attention_dec"].train()

        else:
            self.models["encoder"].train()
            self.models["depth"].train()

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
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """


        print("Training")
        self.set_train()


        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

        
            outputs, losses = self.process_batch(inputs)

            if not self.opt.deformable:
                self.depth_optim.zero_grad()
            if self.opt.deformable or self.opt.finetune_oracleC_and_attention:
                self.att_optim.zero_grad()
            losses["loss"].backward()
            if not self.opt.deformable:
                self.depth_optim.step()
            if self.opt.deformable or self.opt.finetune_oracleC_and_attention:
                self.att_optim.step()
            
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
            
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                        
                self.log("train", inputs, outputs, losses)
                if not self.opt.custom_data:
                    self.val()
                
            self.step += 1

            
        if not self.opt.deformable:
            self.depth_lr_sched.step()
        if self.opt.deformable or self.opt.finetune_oracleC_and_attention:
            self.att_lr_sched.step()
            
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
            
        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            image_focused = inputs["color_encoder_aug", 0, 0]
            image_wac = inputs["color_aug", 0, 0]

            encoder_input = image_wac
            self.foveated_visual = encoder_input

            if self.opt.deformable:
                with torch.no_grad():
                    features = self.models["encoder"](encoder_input)
                    outputs = self.models["depth"](features)
                self.attention_mask = self.models["saliency"](features)["disp", 0]
                self.foveated_visual = self.attention_mask*image_focused + \
                                       (1 - self.attention_mask)*image_wac
                self.warp_mask = False
                self.generate_images_pred(inputs, outputs)
                losses = self.compute_losses(inputs, outputs)
                return outputs, losses
                
            if self.opt.oracleC:                
                with torch.no_grad():
                    # get predicted for oracle attention
                    features = self.models["encoder"](encoder_input)
                    outputs = self.models["depth"](features)

                # wac error map
                self.warp_mask, self.create_mask = False, True
                self.generate_images_pred(inputs, outputs)
                self.compute_losses(inputs, outputs)

                '''
                # get learned attention if necessary
                if self.opt.finetune_oracleC_and_attention:
                    self.attention_mask = self.models["attention_dec"](features)[("disp", 0)]
                    self.oracle_mask = torch.zeros_like(self.attention_mask).to(self.device)
                    for i in range(self.attention_mask.size(0)):
                        self.oracle_mask[i] = self.attention_mask[i] >= self.attention_mask[i].flatten().topk(self.num)[0].min()
                    #print('a', oracle_mask.sum() / oracle_mask.size(0))
                    #self.attention_mask = self.attention_mask * self.oracle_mask
                   
                '''
                    
                # foveate based on wac error
                encoder_input = image_focused*self.attention_mask + encoder_input*(1-self.attention_mask)
                
                self.foveated_visual = encoder_input
                features = self.models["encoder"](encoder_input)
                outputs = self.models["depth"](features)

                # compute foveated error
                self.warp_mask, self.create_mask = True, False
                self.generate_images_pred(inputs, outputs)
                losses = self.compute_losses(inputs, outputs)
                return outputs, losses
     
            if self.opt.e2e:
                with torch.no_grad():
                    features = self.models["encoder_wac"](encoder_input)
                    outputs = self.models["depth_wac"](features)
                    self.attention_mask = self.models["saliency"](features)["disp", 0]

                    encoder_input = self.attention_mask*image_focused + \
                                (1 - self.attention_mask)*image_wac
                self.foveated_visual = encoder_input
                features = self.models["encoder"](encoder_input)
                outputs = self.models["depth"](features)
                self.warp_mask = True
                self.generate_images_pred(inputs, outputs)
                losses = self.compute_losses(inputs, outputs)
                return outputs, losses
               
            else:
                features = self.models["encoder"](encoder_input)
                outputs = self.models["depth"](features)
                self.warp_mask, self.create_mask = False, False
                self.generate_images_pred(inputs, outputs)
                losses = self.compute_losses(inputs, outputs)
                return outputs, losses
                
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
            
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                wac_sample = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                if self.warp_mask:
                    focused_sample = F.grid_sample(
                    inputs[("color_encoder", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                    outputs[("color", frame_id, scale)] = self.attention_mask*focused_sample + \
                                                          (1 - self.attention_mask)*wac_sample

                    if self.opt.FT_square_fovea:
                        mask_idx = self.attention_mask > 0
                        outputs[("color", frame_id, scale)][mask_idx] = outputs[("color", frame_id, scale)][mask_idx] + self.brightness
                        outputs[("color", frame_id, scale)][outputs[("color", frame_id, scale)] < 0.0] = 0.0
                        outputs[("color", frame_id, scale)][outputs[("color", frame_id, scale)] > 1.0] = 1.0

                    
                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] \
                            = inputs[("color_encoder", frame_id, source_scale)]*self.attention_mask \
                            + (1-self.attention_mask)*inputs[("color", frame_id, source_scale)]
                else:
                    outputs[("color", frame_id, scale)] = wac_sample
                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] \
                            = inputs[("color", frame_id, source_scale)]
                        
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        # weight regions of image
        if self.opt.weight_regions and self.warp_mask:
            reprojection_loss[self.oracle_mask > 0] *= 1 + self.opt.fovea_weight
            reprojection_loss[self.oracle_mask == 0] *= (1-self.opt.fovea_weight)

        return reprojection_loss

    
    def compute_oracle_mask(self, im):
        """Find top N errors for every image. Compute oracle mask."""
        self.reproj_im = im.detach().clone()
        num = int((self.saccade_total[0]*self.saccade_total[1]))
        self.oracle_mask = torch.zeros_like(self.reproj_im)
        for i in range(self.reproj_im.size(0)):
            self.oracle_mask[i] = self.reproj_im[i] >= self.reproj_im[i].flatten().topk(num)[0].min()
        if not self.opt.deformable:
            self.attention_mask = self.oracle_mask
        else:
            oracle_attention_sum = ( self.attention_mask * self.oracle_mask).sum() / \
                                   (num*self.reproj_im.size(0))
            attention_sum = self.attention_mask.sum() / self.reproj_im.size(0)
            #print( ( (self.attention_mask * self.oracle_mask) ).sum() / self.reproj_im.size(0) , \
            #       attention_sum)
            
            return oracle_attention_sum, attention_sum
        
    def compute_oracle_loss(self, im):
        """Find top N errors for every image. Compute oracle loss."""
        '''
        1. remove weight on oracle mask
        2. train deformable attention mask 
           - unet train against oracle map
        '''
        self.reproj_im = im.detach().clone()
        num = int((self.saccade_total[0]*self.saccade_total[1]))
        self.oracle_mask = torch.zeros_like(self.reproj_im)
        for i in range(self.reproj_im.size(0)):
            self.oracle_mask[i] = self.reproj_im[i] >= self.reproj_im[i].flatten().topk(num)[0].min()
        oracle_attention_sum = ((self.attention_mask * self.oracle_mask)>0).sum() / \
                               (num*self.reproj_im.size(0))
        loss = (self.attention_mask * self.oracle_mask).sum() / (num*self.reproj_im.size(0))
        const = 0.0001
        loss = const * (1 / loss)
        return loss, oracle_attention_sum

        
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch"""
        losses = {}
        total_loss = 0
        
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)] 

            if self.warp_mask: # compare against foveated or wac 
                attention_mask_resize = F.interpolate(
                    self.attention_mask, [disp.shape[2], disp.shape[3]],
                    mode="bilinear", align_corners=False)
                
                color = inputs[("color_encoder", 0, scale)]*attention_mask_resize \
                        + (1-attention_mask_resize)*inputs[("color", 0, scale)]
                target = inputs[("color_encoder", 0, source_scale)]*self.attention_mask \
                         + (1-self.attention_mask)*inputs[("color", 0, source_scale)]
            else:
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]

                if self.opt.deformable:
                    oracle_im = self.compute_reprojection_loss(pred, target)
                    losses["overlap"], losses["attention_sum"] = self.compute_oracle_mask(oracle_im)
                    losses["loss"] = nn.BCELoss()(self.attention_mask, self.oracle_mask)
                    #losses["loss"] = self.deformable_criterion(self.attention_mask, self.oracle_mask)
                    return losses
                '''
                if self.opt.oracleB:
                    oracle_im = self.compute_reprojection_loss(pred, target)
                    oracle_loss, losses["overlap"] = self.compute_oracle_loss(oracle_im)
                '''
 
                if self.opt.oracleC and self.create_mask:
                    oracle_im = self.compute_reprojection_loss(pred, target)
                    self.compute_oracle_mask(oracle_im)
                    return None
                    
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)
            
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)] 
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        # oracle loss term
        if self.opt.finetune_oracleC_and_attention:
            total_loss += 0.1 * nn.BCELoss()(self.attention_mask, self.oracle_mask)
        
        losses["loss"] = total_loss
        return losses

    
    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
            
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    if self.opt.oracleC or self.opt.deformable:
              
                        writer.add_image(
                            "mask_{}_{}/{}".format(frame_id, s, j),
                            self.attention_mask[j].data,self.step)
                        writer.add_image(
                            "foveated_{}_{}/{}".format(frame_id, s, j),
                            self.foveated_visual[j].data, self.step)
                        if not self.opt.e2e:
                            writer.add_image(
                                "reproj_{}_{}/{}".format(frame_id, s, j),
                                ( .5*self.attention_mask[j]+(1-0.5)*self.oracle_mask[j] ).data, self.step)
              
                        
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)


            
        for model_name, model in self.models.items():
            if self.opt.deformable:
                if model_name == "saliency":
                    save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                    to_save = model.state_dict()
                    torch.save(to_save, save_path)
                    save_path = os.path.join(save_folder, "{}.pth".format("att_adam"))
                    torch.save(self.att_optim.state_dict(), save_path)
            elif self.opt.e2e or self.opt.oracleC:
                if (model_name == 'encoder') or (model_name == 'depth') or (model_name == 'attention_dec'):
                    save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                    to_save = model.state_dict()
                    if model_name == 'encoder':
                        # save the sizes - these are needed at prediction time
                        to_save['height'] = self.opt.height
                        to_save['width'] = self.opt.width
                        to_save['use_stereo'] = self.opt.use_stereo
                    torch.save(to_save, save_path)
                                
            else:
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save = model.state_dict()
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save['height'] = self.opt.height
                    to_save['width'] = self.opt.width
                    to_save['use_stereo'] = self.opt.use_stereo
                torch.save(to_save, save_path)
        
    def load_model(self, custom=None):
        """Load model(s) from disk
        """
        if custom is not None:
            self.custom_path = os.path.expanduser(custom['path'])
            models_to_load = custom['models_to_load']
            model_name = custom['model_names']
            print('A\n',models_to_load)
            print(model_name)
            
            assert os.path.isdir(self.custom_path), \
                "Cannot find folder {}".format(self.custom_path)
            print("loading model from folder {}".format(self.custom_path))

            for i, n in enumerate(models_to_load):
                print("Loading {} weights...".format(n))
                path = os.path.join(self.custom_path, "{}.pth".format(model_name[i]))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

            
        else:            
            self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

            assert os.path.isdir(self.opt.load_weights_folder), \
                "Cannot find folder {}".format(self.opt.load_weights_folder)
            print("loading model from folder {}".format(self.opt.load_weights_folder))


            for n in self.opt.models_to_load:
                print("Loading {} weights...".format(n))
                path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state
        '''
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.depth_optim.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
        '''
