from __future__ import absolute_import, division, print_function

import os,sys,time
import cv2
import numpy as np
import random
import matplotlib
import matplotlib.cm as cm

import torch
import torchvision
import torchgeometry as tgm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vision_utils

from layers import disp_to_depth, BackprojectDepth, Project3D
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from CannyEdgePytorch import net_canny

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


def get_focused_depths(opt):
    """Get focused depth predictions"""

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        focused_path = "~/tmp/focused/models/weights_19/"
        opt.load_weights_folder = os.path.expanduser(focused_path)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, opt=opt, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        writer = SummaryWriter(os.path.join(opt.log_dir, 'TEST_'+opt.load_weights_folder))
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                
                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
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

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    #writer_pred = torch.zeros(pred_disps.shape[0], 1, 375, 1242)
    focused_depth = []
    
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp


        
        # tensorboard
        #writer_pred[i,0] = torch.from_numpy(pred_depth)        
        
        if opt.eval_split == "eigen":
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


        
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        focused_depth.append(pred_depth)     
        
        errors.append(compute_errors(gt_depth, pred_depth))
    
    #grid = torchvision.utils.make_grid(writer_pred[:10], normalize=True)
    #writer.add_image('TEST/test', grid)
    #writer.close()

    
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    np.save('focused_depth.npy', np.array(focused_depth))


def oracle_paste(gt_depth, pred_depth, focused_depth, opt):
    """Deformably paste focused depth into wac depth"""

    # compute per pixel error
    error = torch.from_numpy(np.abs(gt_depth - pred_depth))
    #print('error',error.size(0))
    # find top SaccadeCam number of errors
    ratio = error.size(0) / (opt.height*opt.width)
    #print('ratio', ratio)
    num = int(opt.camera.saccade_px[0]*opt.camera.saccade_px[1]*ratio)
    #print('num', num)
    oracle_mask = (error >= error.flatten().topk(num)[0].min()).numpy()
    pred_depth[oracle_mask] = focused_depth[oracle_mask]
    return pred_depth

'''
mask = error.argsort()[-num:][::-1]
pred_depth[mask] = focused_depth[mask]
return pred_depth
'''

def colorize(value, vmin=0, vmax=1000, cmap='magma'):

    #print(value.min(), value.max())
    value = value.cpu().numpy()[0]

    # normalize
    vmin = 0.1#value.min() if vmin is None else vmin
    vmax = 0.95#value.max() if vmax is None else vmax
    value = ((vmax-vmin)*value) + vmin

    

    cmapper =  cm.get_cmap(cmap)
    value = cmapper(value)
    img = value[:,:,:3]

    return img.transpose((2, 0, 1))
                                                
def evaluate(opt):
    """Get WAC errors after deformably pasting focused"""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    #focused_depth = get_focused_depths(opt)
    #np.save('focused_depths.npy', focused_depth)
    focused_depth = np.load('focused_depth.npy', allow_pickle=True)

    os.environ['CUDA_VISIBLE_DEVICES']=opt.gpus
    
    if opt.oracle0 or opt.oracleC or opt.oracleB:
        opt.frame_ids.append("s")
    
    
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    
    if opt.ext_disp_to_eval is None:
        if opt.deformable:

            '''
            if opt.wac_scale < 0.75:
                weights = "4"
            else:
                weights = "19"
            '''

            if opt.weight_regions:
                '''
                if opt.defocused_scale == 0.25:
                    high = 1.15; low = 0.85
                if opt.defocused_scale == 0.2:
                    high = 1.2; low = 0.8
                if opt.defocused_scale == 0.15:
                    high = 1.25; low = 0.75
                '''
                # all weighted models are named the same but
                # were trained with weighting described in paper
                #path = "~/tmp/attention/oracleC_weighted_1.75_0.25_"
                path = "~/tmp/attention/RR_depth_"
            else:
                #if opt.supervised:
                #    path = "~/tmp/attention/oracleC_weight_proper_scratch_FT_"
                #else:
                path = "~/tmp/attention/oracleC_weight_old_scratch_"
                #path = "~/tmp/attention/RR_depth_"
                
            '''
            if opt.deformable:
                #if opt.defocused_scale == 0.2:
                #    weights = "13"
                if opt.defocused_scale == 0.0125:
                    opt.epoch_to_load = "1"
            '''
            
            opt.load_weights_folder = path + \
                                      str(opt.defocused_scale)+ \
                                      "_"+str(opt.wac_scale)+"/models/weights_"+opt.epoch_to_load+"/"

        #if opt.fovea == 1:
        #    opt.load_weights_folder = "~/tmp/defocused_"+str(opt.defocused_scale)+ \
        #                              "_0.0"+"/models/weights_19/"
            

            
            
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           opt.frame_ids, 4, opt=opt, is_train=False)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))




        ###############################


        
        
        '''
        path = "~/tmp/attention/deformable_"+str(opt.defocused_scale)+"_"+str(opt.wac_scale) \
        +"/models/weights_4/saliency.pth"
        att_dec_path = os.path.expanduser(path)
                
        #att_dec_path = os.path.join(opt.load_weights_folder, "attention_dec.pth")
        
        # load attention decoder
        attention_dec = networks.DepthDecoder(
            encoder.num_ch_enc, [0],
            num_output_channels=1)
        att_dict = torch.load(att_dec_path)
        model_dict = attention_dec.state_dict()
        attention_dec.load_state_dict(torch.load(att_dec_path))

        attention_dec.cuda()
        attention_dec.eval()
        '''
        ###############################
        


        
        #####
        # WAC
        ####
        path = "~/tmp/wac_"+str(opt.defocused_scale)+ \
               "_"+str(opt.wac_scale)+"/models/weights_19/"
        enc_path = os.path.expanduser(path)
        assert os.path.isdir(enc_path), \
            "Cannot find a folder at {}".format(enc_path)
        print("-> Loading weights from {}".format(enc_path))
        
        encoder_path = os.path.join(enc_path, "encoder.pth")
        decoder_path = os.path.join(enc_path, "depth.pth")
        encoder_dict = torch.load(encoder_path)

        wac_encoder = networks.ResnetEncoder(opt.num_layers, False)
        wac_depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = wac_encoder.state_dict()
        wac_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        wac_depth_decoder.load_state_dict(torch.load(decoder_path))

        #####
        # FOCUSED
        ####
        path = "~/tmp/focused/models/weights_19/"
        enc_path = os.path.expanduser(path)
        assert os.path.isdir(enc_path), \
            "Cannot find a folder at {}".format(enc_path)
        print("-> Loading weights from {}".format(enc_path))
        
        encoder_path = os.path.join(enc_path, "encoder.pth")
        decoder_path = os.path.join(enc_path, "depth.pth")
        encoder_dict = torch.load(encoder_path)

        focused_encoder = networks.ResnetEncoder(opt.num_layers, False)
        focused_depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = focused_encoder.state_dict()
        focused_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        focused_depth_decoder.load_state_dict(torch.load(decoder_path))


        #####
        # DEFOCUSED
        ####
        path = "~/tmp/defocused_"+str(opt.defocused_scale)+"_0.0/models/weights_19/"
        enc_path = os.path.expanduser(path)
        assert os.path.isdir(enc_path), \
            "Cannot find a folder at {}".format(enc_path)
        print("-> Loading weights from {}".format(enc_path))
        
        encoder_path = os.path.join(enc_path, "encoder.pth")
        decoder_path = os.path.join(enc_path, "depth.pth")
        encoder_dict = torch.load(encoder_path)

        defoc_encoder = networks.ResnetEncoder(opt.num_layers, False)
        defoc_depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = defoc_encoder.state_dict()
        defoc_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        defoc_depth_decoder.load_state_dict(torch.load(decoder_path))


        
        #####
        if opt.deformable:

            if opt.comparison == "edges":
                edge_net = net_canny.Net(threshold=0.125, use_cuda=True)
                edge_net.cuda()
                edge_net.eval()


            #att_dec_path_path = os.path.join(opt.load_weights_folder, "attention_dec.pth")

            
            path = "~/tmp/attention/deformable_"+str(opt.defocused_scale)+"_"+str(opt.wac_scale) \
                   +"/models/weights_3/saliency.pth"

            #path = "~/tmp/attention/deformable_"+str(0.2)+"_"+str(0.75) \
            #       +"/models/weights_3/saliency.pth"
            
            att_dec_path = os.path.expanduser(path)
            

            # load attention decoder
            attention_dec = networks.DepthDecoder(
                encoder.num_ch_enc, [0],
                num_output_channels=1)
            att_dict = torch.load(att_dec_path)
            model_dict = attention_dec.state_dict()
            attention_dec.load_state_dict(torch.load(att_dec_path))

            attention_dec.cuda()
            attention_dec.eval()
            
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        wac_encoder.cuda()
        wac_encoder.eval()
        wac_depth_decoder.cuda()
        wac_depth_decoder.eval()
        focused_encoder.cuda()
        focused_encoder.eval()
        focused_depth_decoder.cuda()
        focused_depth_decoder.eval()

        defoc_encoder.cuda()
        defoc_encoder.eval()
        defoc_depth_decoder.cuda()
        defoc_depth_decoder.eval()

        
        if opt.comparison == "edges":
                edge_net = net_canny.Net(threshold=0.125, use_cuda=True)
                edge_net.cuda()
                edge_net.eval()
            
        
        pred_disps = []
        inputs = []
        att_masks = []
        backproject_depth = {}
        project_3d = {}
        
        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        opt.model_name = str(opt.defocused_scale)+"_"+str(opt.wac_scale)+"_"
        if opt.fovea == 1:
            opt.model_name = str(opt.defocused_scale)+"_"+"DEFOCUSED"
        else:
            opt.model_name = str(opt.defocused_scale)+"_"+"FOVEATED"
            
        writer = SummaryWriter(os.path.join(opt.log_dir, "RESULTS"+'_'+opt.model_name))
        count = 1
        att_list = []
        overlap_sum = []
        binary_sum = []
        num = int(opt.camera.saccade_px[0]*opt.camera.saccade_px[1])
        with torch.no_grad():
            for data in dataloader:
                for key, ipt in data.items():
                    data[key] = ipt.cuda()

                count += 1
                input_color = data[("color", 0, 0)]
                input_focused = data[("color_encoder", 0, 0)]


                h = opt.height
                w = opt.width 
                backproject_depth[0] = BackprojectDepth(input_color.size(0), h, w)
                backproject_depth[0].cuda()
                project_3d[0] = Project3D(input_color.size(0), h, w)
                project_3d[0].cuda()

                
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                
                features = encoder(input_color)
                output = depth_decoder(features)
                
                if opt.deformable:
                    features = encoder(input_color)
                    
                    att_mask = attention_dec(features)[("disp", 0)]
                    oracle_mask = torch.zeros_like(att_mask)
                    for i in range(att_mask.size(0)):
                        oracle_mask[i] = att_mask[i] >= att_mask[i].flatten().topk(num)[0].min()
                    #print('a', oracle_mask.sum() / oracle_mask.size(0))
                    att_mask *= oracle_mask
                    #print('b', att_mask.sum() / att_mask.size(0))
                    #att_mask[att_mask > 0.0] = 1.0
                    #print('c', att_mask.sum() / att_mask.size(0))


                    if opt.comparison == "edges":
                        oracle_mask = torch.zeros_like(output[("disp",0)])
                        for i in range(input_color.size(0)):
                            oracle_mask[i] = (edge_net(input_color[i].unsqueeze(0))[5])
                            oracle_mask[i] = oracle_mask[i] >= oracle_mask[i].flatten().topk(num)[0].min()
                        att_mask = oracle_mask
                    
                    foveated = att_mask*input_focused + (1-att_mask)*input_color
                    output = depth_decoder(encoder(foveated))

                    '''
                    #att_mask[att_mask < 0.001] = 0
                    att_sum = (att_mask).sum() / att_mask.size(0)
                    att_list.append(att_sum.data)

                    '''

                    '''

                    disp = output[("disp", 0)]
                    _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)
                    output[("depth", 0, 0)] = depth
                    T = data["stereo_T"]


                    # generate sampling grid
                    cam_points = backproject_depth[0](
                        depth, data[("inv_K", 0)])
                    pix_coords = project_3d[0](
                        cam_points, data[("K", 0)], T)
                    output[("sample", "s", 0)] = pix_coords

                    # sample second camera with sampling grid
                    wac_sample = F.grid_sample(
                        data[("color", "s", 0)],
                        output[("sample", "s", 0)],
                        padding_mode="border", align_corners=True)

                    # get photometric error
                    pred = wac_sample
                    target = data[("color", 0, 0)]
                    error = torch.abs(target - pred).mean(1, True)
                    oracle_mask = torch.zeros_like(error)
                    for i in range(error.size(0)):
                        oracle_mask[i] = error[i] >= error[i].flatten().topk(num)[0].min()
                    comb = oracle_mask * att_mask
                    oracle_attention_sum = 100* (comb.sum() / \
                                                 (num*error.size(0)) )
                    overlap_sum.append(oracle_attention_sum)
                    mask = comb > 0
                    binary_overlap = 100* (mask.sum() / \
                                           (num*error.size(0)) )
                    binary_sum.append(binary_overlap)
                    '''

                    

                # paste focused depth into wac depth based on
                # photometric error between wac depth and focused depth                
                elif opt.oracleC:
                    # get wac error
                    output = wac_depth_decoder(wac_encoder(input_color))
                    disp = output[("disp", 0)]
                    _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)
                    output[("depth", 0, 0)] = depth
                    T = data["stereo_T"]

                    # generate sampling grid
                    cam_points = backproject_depth[0](
                        depth, data[("inv_K", 0)])
                    pix_coords = project_3d[0](
                        cam_points, data[("K", 0)], T)
                    output[("sample", "s", 0)] = pix_coords

                    # sample second camera with sampling grid
                    wac_sample = F.grid_sample(
                        data[("color", "s", 0)],
                        output[("sample", "s", 0)],
                        padding_mode="border", align_corners=True)

                    # photometric error
                    pred = wac_sample
                    target = data[("color", 0, 0)]
                    wac_error = torch.abs(target - pred).mean(1, True)


                    # get focused error 
                    output = focused_depth_decoder(focused_encoder(input_focused))
                    disp = output[("disp", 0)]
                    _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)
                    output[("depth", 0, 0)] = depth
                    T = data["stereo_T"]

                    # generate sampling grid
                    cam_points = backproject_depth[0](
                        depth, data[("inv_K", 0)])
                    pix_coords = project_3d[0](
                        cam_points, data[("K", 0)], T)
                    output[("sample", "s", 0)] = pix_coords

                    # sample second camera with sampling grid
                    focused_sample = F.grid_sample(
                        data[("color_encoder", "s", 0)],
                        output[("sample", "s", 0)],
                        padding_mode="border", align_corners=True)

                    # photometric error
                    pred = focused_sample
                    target = data[("color_encoder", 0, 0)]
                    focused_error = torch.abs(target - pred).mean(1, True)

                    # focused error < wac error in fovea regions 
                    # focused error ~= wac_error in periphery
                    oracle_error =  wac_error - focused_error
                        
                    num = int(opt.camera.saccade_px[0]*opt.camera.saccade_px[1])
                    oracle_mask = torch.zeros_like(oracle_error)
                    for i in range(oracle_error.size(0)):
                        oracle_mask[i] = oracle_error[i] >= oracle_error[i].flatten().topk(num)[0].min()


                    if opt.comparison == "edges":
                        oracle_mask = torch.zeros_like(output[("disp",0)])
                        for i in range(input_color.size(0)):
                            oracle_mask[i] = (edge_net(input_color[i].unsqueeze(0))[5])
                            oracle_mask[i] = oracle_mask[i] >= oracle_mask[i].flatten().topk(num)[0].min()
                        #print(oracle_mask.sum() / oracle_mask.size(0), num)

                    if opt.finetune_oracleC_and_attention:
                        features = encoder(input_color)                    
                        att_mask = attention_dec(features)[("disp", 0)]
                        oracle_mask = torch.zeros_like(att_mask)
                        for i in range(att_mask.size(0)):
                            oracle_mask[i] = att_mask[i] >= att_mask[i].flatten().topk(num)[0].min()
                            #print('a', oracle_mask.sum() / oracle_mask.size(0))
                            att_mask *= oracle_mask
                            #print('b', att_mask.sum() / att_mask.size(0))
                            #att_mask[att_mask > 0.0] = 1.0
                            #print('c', att_mask.sum() / att_mask.size(0))

                        
                    #input_color = att_mask*input_focused + (1-att_mask)*input_color
                    #output = depth_decoder(encoder(input_color))
                        

                    # replace wac region with wac disparity
                    if opt.replace_wac_regions:
                        wac_disp = wac_depth_decoder(wac_encoder(input_color))
                        output[("disp"), 0] = oracle_mask*output[("disp"), 0] + \
                                              (1 - oracle_mask)*wac_disp[("disp"), 0]

                        
                '''
                foc_output = focused_depth_decoder(focused_encoder(input_focused))
                foc_output = foc_output[("disp", 0)]

                wac_output = wac_depth_decoder(wac_encoder(input_color))
                wac_output = wac_output[("disp", 0)]

                our_output = output[("disp", 0)]

                # blur to make defocused
                gauss = tgm.image.GaussianBlur((5, 5), (10.5, 10.5))
                defocused = gauss(input_focused)

                #blur to make wac
                gauss = tgm.image.GaussianBlur((13, 13), (10.5, 10.5))
                wac = gauss(input_color)


                writer.add_image(
                    "mask",
                    vision_utils.make_grid(oracle_mask.data, nrow=4), count)


                writer.add_image(
                    "WAC",
                    vision_utils.make_grid(input_color.data, nrow=4), count)

                writer.add_image(
                    "DEFOCUSED",
                    vision_utils.make_grid(defocused.data, nrow=4), count)

                writer.add_image(
                    "WAC DEPTH",
                    colorize(vision_utils.make_grid(wac_output.data, nrow=4,
                                                    normalize=True)), count)
                '''

                '''
                writer.add_image(
                    "attention_mask",
                    vision_utils.make_grid(att_mask.data, nrow=4), count)
                
                writer.add_image(
                    "foveated_color",
                    vision_utils.make_grid(foveated.data, nrow=4), count)

                writer.add_image(
                    "foveated_depth",
                    colorize(vision_utils.make_grid(our_output.data, nrow=4,
                                                    normalize=True)), count)                
                writer.add_image(
                    "defocused_color",
                    vision_utils.make_grid(defocused.data, nrow=4), count)
                '''

                if opt.fovea == 1:
                    output = depth_decoder(encoder(input_color))
                    foveated = torch.tensor([0]).cuda()
                    att_mask = torch.tensor([0]).cuda()

                '''
                writer.add_image(
                    "defocused_depth",
                    colorize(vision_utils.make_grid(defocused_depth.data, nrow=4,
                                                    normalize=True)), count)                

                writer.add_image(
                    "defocused_color",
                    vision_utils.make_grid(input_color.data, nrow=4), count)
                
                count += 1
                '''

                # oracle_paste() handles replacing wac depth with focused depth
                if opt.oracle0:
                    foveated = torch.tensor([0]).cuda()
                    att_mask = torch.tensor([0]).cuda()
                # need to replace wac-focused error regions with focused depth
                if opt.oracleC:
                    foveated = torch.tensor([0]).cuda()
                    att_mask = torch.tensor([0]).cuda()
                    output = depth_decoder(encoder(input_color))
                    foc_output = focused_depth_decoder(focused_encoder(input_focused))
                    output[("disp"), 0] = oracle_mask*foc_output[("disp", 0)] + \
                                          (1-oracle_mask)*output[("disp", 0)]                    
                    
                    

                #########
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                inputs.append(foveated.cpu().numpy())
                att_masks.append(att_mask.cpu().numpy())
                

        # create the directory you want to save to
        fn = opt.model_name+"_disp.npy"
        direc = os.path.expanduser('~/projects/foveaCam/monodepth2/results')
        if not(os.path.exists(direc)):
            os.mkdir(direc)
        np.save(os.path.join(direc, fn), pred_disps)

        fn = opt.model_name+"_color.npy"
        np.save(os.path.join(direc, fn), inputs)

        fn = opt.model_name+"_attention.npy"
        att_mask[att_mask > 0.0] = 1.0
        np.save(os.path.join(direc, fn), att_masks)

        
        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
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

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp


        # tensorboard
        #writer_pred[i,0] = torch.from_numpy(pred_depth)        
        
        if opt.eval_split == "eigen":
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


        
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        if opt.oracle0:
            pred_depth = oracle_paste(gt_depth, pred_depth, focused_depth[i], opt)

        errors.append(compute_errors(gt_depth, pred_depth))

        #if i ==12:
        #break

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    '''
    if opt.oracleB:
        overlap_error = (torch.tensor(att_list).mean() - num).abs().numpy()
        overlap_error = 100 * ((num - overlap_error) / num)
        
        att_e = np.array([ [int((torch.tensor(att_list).mean()).numpy())],
                           [overlap_error],
                           [(torch.tensor(binary_sum).mean()).numpy()] ]).squeeze(1)
        mean_errors = np.concatenate((mean_errors, att_e), axis=0)
        print("\n"+ opt.load_weights_folder +"\n  " + ("{:>8} | " * 10).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "att_mask", "BW % Correct", "binary overlap %"))
        print(("&{: 8.3f}  " * 10).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")
    '''

    
    #else:
    print("\n"+ opt.load_weights_folder +"\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    
    
    # print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    # print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    # print("\n-> Done!")


    

if __name__ == "__main__":

    options = MonodepthOptions()
    options = options.parse()
    
    #options.epoch_to_load = str(i)
    evaluate(options)
