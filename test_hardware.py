from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import torch
from torchvision import transforms#, datasetsc

import networks
from layers import disp_to_depth

import camera
#import greedy_torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images')
    parser.add_argument('--output_path', type=str,
                        help='path to a test image or folder of images')
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--data_rep",
                        help="which data to load",
                        default="foveated",
                        type=str)

    parser.add_argument("--height",
                        type=int,
                        help="input image height",
                        default=192)
    parser.add_argument("--width",
                        type=int,
                        help="input image width",
                        default=640)
    parser.add_argument("--defocused_scale",
                        type=float,
                        help="% of focused resolution",
                        default=0.25)
    parser.add_argument("--wac_scale",
                        type=float,
                        help="% of defocused resolution",
                        default=0.75)
    parser.add_argument("--fovea",
                        type=int,
                        default=0)
    parser.add_argument("--num_fovea",
                        type=int,
                        default=10)

    parser.add_argument("--gpus", type=str, default= '7')
    
    return parser.parse_args()

def foveate(wac, fovea):
    """alpha composite fovea onto wac"""
    fovea[fovea<0.1]=0.
    mask = (fovea>0).astype(np.float32)
    #mask = fovea
    
    # gamma correction
    gamma = .35#.35
    fovea = fovea**(1/gamma)
    
    mask = (cv2.blur(mask, (41,41)))**5
    #mask = (cv2.blur(mask, (3,3)))**3
    fovea = fovea*mask
    saccade = fovea + (1-mask)*wac
    return saccade


def color(disp, output_directory, output_name):
    '''save disp to disk'''
    #output_name = os.path.splitext(os.path.basename(image_path))[0]
    '''
    name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
    scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
    np.save(name_dest_npy, scaled_disp.cpu().numpy())
    '''

    # Saving colormapped depth image
    disp_np = disp.squeeze().cpu().numpy()
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    #name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
    #im.save(name_dest_im)
    return im
    
def error_metrics(gt, pred):
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

    
def compute_errors(focused_disp, pred_disp, min_depth=0.1, max_depth=100):
    '''compute errrors between focused and predicted disp'''
    stereo_scale = 5.4
    min_depth = 1e-3; max_depth = 80
    focused_depth = (1 / focused_disp) * stereo_scale
    pred_depth = (1 / pred_disp) * stereo_scale
    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth
    focused_depth[focused_depth < min_depth] = min_depth
    focused_depth[focused_depth > max_depth] = max_depth

    
    return error_metrics(focused_disp, pred_disp)
    

    
def disp_to_depth(disp, min_depth=0.1, max_depth=100):
    '''convert network's sigmoid output into depth prediction'''
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp

    
def test_hardware(args):
    '''Test SaccadeCam prototype images'''

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpus

    #download_model_if_doesnt_exist(args.model_name)

    #  path = "~/tmp/attention/oracleC_weighted_1.75_0.25_0.15_0.75/models/weights_15/"
    #path = "~/tmp/defocused_0.25_0.0/models/weights_19/"#
    #path = "~/tmp/attention/oracleC_weight_old_scratch_0.15_0.75/models/weights_9/" #11




    ################
    # FOCUSED MODELS
    ################
    path = "~/tmp/focused/models/weights_19/"
    model_path = os.path.expanduser(path)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # load encoder
    focused_encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in focused_encoder.state_dict()}
    focused_encoder.load_state_dict(filtered_dict_enc)
    focused_encoder.to(device)
    focused_encoder.eval()

    # load decoder
    focused_decoder = networks.DepthDecoder(
        num_ch_enc=focused_encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    focused_decoder.load_state_dict(loaded_dict)
    focused_decoder.to(device)
    focused_decoder.eval()

    ################
    # TARGET MODELS
    ################
    path = "~/tmp/defocused_"+str(args.defocused_scale)+"_0.0/models/weights_19/"
    model_path = os.path.expanduser(path)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # load encoder
    target_encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in target_encoder.state_dict()}
    target_encoder.load_state_dict(filtered_dict_enc)
    target_encoder.to(device)
    target_encoder.eval()

    # load decoder
    target_decoder = networks.DepthDecoder(
        num_ch_enc=target_encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    target_decoder.load_state_dict(loaded_dict)
    target_decoder.to(device)
    target_decoder.eval()


    ###################
    # SaccadeCam MODELS
    ###################
    '''
    path = "~/tmp/attention/oracleC_weight_old_scratch_"            
    path = path + \
           str(0.25)+ \
           "_"+str(0.75)+"/models/weights_"+str(17)+"/"
    '''
    path = "~/tmp/attention/FT_squares_0.25_0.75/models/weights_9/"
    model_path = os.path.expanduser(path)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # load encoder
    saccade_encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in saccade_encoder.state_dict()}
    saccade_encoder.load_state_dict(filtered_dict_enc)
    saccade_encoder.to(device)
    saccade_encoder.eval()

    # load decoder
    saccade_decoder = networks.DepthDecoder(
        num_ch_enc=saccade_encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    saccade_decoder.load_state_dict(loaded_dict)
    saccade_decoder.to(device)
    saccade_decoder.eval()

    

    # load attention decoder
    '''
    att_path = "~/tmp/attention/deformable_0.25_0.75/models/weights_3/saliency.pth"
    att_dec_path = os.path.expanduser(att_path)

    attention_dec = networks.DepthDecoder(
        encoder.num_ch_enc, [0],
        num_output_channels=1)
    att_dict = torch.load(att_dec_path)
    model_dict = attention_dec.state_dict()
    attention_dec.load_state_dict(torch.load(att_dec_path))
    attention_dec.to(device)
    attention_dec.eval()
    '''
    
    # FINDING INPUT IMAGES
    #args.image_path = 'kitti_data/2011_10_03/2011_10_03_drive_0027_sync/image_02/data'
    #args.output_path = 'custom_video_results/vid2'
    data = '1'
    args.image_path = 'hardware_results/data/imgs/aug9/'+data+'/'#'rebuttal/june16/'+video+'/imgs/'
    args.output_path = 'hardware_results/data/results/aug9/'+data+'/'
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.output_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        defoc_paths = sorted(glob.glob(args.image_path+'defoc_'+'*.png'))
        fovea_paths = sorted(glob.glob(args.image_path+'fovea_'+'*.png'))
        att_paths = sorted(glob.glob(args.image_path+'att_'+'*.png'))
        output_directory = args.output_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    minim = min(len(defoc_paths), len(fovea_paths), len(att_paths))
    defoc_paths = defoc_paths[:minim]
    fovea_paths = fovea_paths[:minim]
    att_paths = att_paths[:minim]

    # video 1
    # 414, 192

    # video 0
    # 33, 519, 670

    print(len(defoc_paths))
    
    a=[]; b=[]; c=[]
    if data == '1':
        frames = [414, 192]
    if data == '0':
        frames = [33, 519, 670]
    for i in frames:
        a.append(defoc_paths[i])
        b.append(fovea_paths[i])
        c.append(att_paths[i])
    defoc_paths = a
    fovea_paths = b
    att_paths = c
    

    # selective frames
    pmin = 0; #pmax = 1000;
    '''
    defoc_paths = defoc_paths[pmin:pmax]
    fovea_paths = fovea_paths[pmin:pmax]
    att_paths = att_paths[pmin:pmax]
    '''
    
    print("-> Predicting on {:d} test images".format(len(defoc_paths)))

    # load camera
    #camera = camera.Camera(camera.load_camera(args))
    saccade_total = camera.get_resolutions(args)["saccade_total"]
    fovea_shape = np.sqrt((saccade_total[0]*saccade_total[1])/args.num_fovea)

    '''
    focused scale = 1.7
    0.25 defocused scale = 8
    wac scale for 0.25 defocused and 0.75 wac = 11
    '''

    if data == '0' or data =='2':
        mincrop = 100; maxcrop = 500;
    if data == '1':
        mincrop = 350; maxcrop = 650;
    if data == '3' or data=='4':
        mincrop = 250; maxcrop = 600;
        
        
    #mincrop = 200; maxcrop = 500
    feed_width = (804//32)*32; feed_height = (300//32)*32
    
    # plotting
    #mosaic = np.zeros((1, 3, input_image.shape[0],
    #                   input_image.shape[1], input_image.shape[2]))

    
    with torch.no_grad():
        for idx, image_path in enumerate(defoc_paths):

            '''
            # prepare focused image
            if video == '1':
                input_image = pil.fromarray(np.array(pil.open(defoc_paths[idx]).convert('RGB'))[200:500])
                fovea = pil.fromarray(np.array(pil.open(fovea_paths[idx]).convert('RGB'))[200:500])
                feed_width = (804//32)*32; feed_height = (300//32)*32 
            else:
                input_image = pil.fromarray(np.array(pil.open(defoc_paths[idx]).convert('RGB'))[300:700])
                fovea = pil.fromarray(np.array(pil.open(fovea_paths[idx]).convert('RGB'))[300:700])
                feed_width = (804//32)*32; feed_height = (300//32)*32
            '''
           
            input_image = pil.fromarray(np.array(pil.open(defoc_paths[idx]).convert('RGB'))[mincrop:maxcrop])
            fovea = pil.fromarray(np.array(pil.open(fovea_paths[idx]).convert('RGB'))[mincrop:maxcrop])
            att = pil.fromarray(np.array(pil.open(att_paths[idx]).convert('RGB'))[mincrop:maxcrop])
            
            

            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            
            
            att = np.array(att.resize((feed_width, feed_height), pil.LANCZOS))
            fovea = fovea.resize((feed_width, feed_height), pil.LANCZOS)

            # lower focused and fovea resolution to match kitti focused resolution
            foc_scale = 1
            target_scale = 8
            wac_scale = 20
            input_image = input_image.resize((int(feed_width/np.sqrt(foc_scale)),
                                              int(feed_height/np.sqrt(foc_scale))), pil.LANCZOS)
            input_image = input_image.resize((feed_width,
                                              feed_height), pil.LANCZOS)
            '''
            fovea = fovea.resize((int(feed_width/np.sqrt(foc_scale)),
                                  int(feed_height/np.sqrt(foc_scale))), pil.LANCZOS)
            fovea = fovea.resize((feed_width,
                                  feed_height), pil.LANCZOS)
            '''
            

            # keep top N fovea
            att = torch.tensor(att)
            num = int(8*saccade_total[0] * saccade_total[1])
            att_mask = att >= att.flatten().topk(num)[0].min()
            att *= att_mask
            att = att.numpy()/255.

            fovea = np.array(fovea).astype(np.float32) / 255.
            focused_rgb = np.array(input_image).copy().astype(np.float32) / 255.
            target_rgb = np.array(input_image).copy().astype(np.float32) / 255.
            wac_rgb = np.array(input_image).copy().astype(np.float32) / 255.



            '''
            target_rgb = cv2.resize(target_rgb,
                                    (int(target_rgb.shape[0]/(np.sqrt(target_scale))),
                                     int(target_rgb.shape[1]/(np.sqrt(target_scale)))),
                                    cv2.INTER_AREA)
            target_rgb = cv2.resize(target_rgb,
                                    (feed_width, feed_height), cv2.INTER_CUBIC)

            '''

            target_rgb = cv2.resize(target_rgb,
                                    (207, 77),
                                    cv2.INTER_AREA)
            target_rgb = cv2.resize(target_rgb,
                                    (feed_width, feed_height), cv2.INTER_CUBIC)
            
            # render wac
            #wac = camera.process(np.array(input_image), args)

            '''
            wac_rgb = cv2.resize(wac_rgb,
                             (int(wac_rgb.shape[0]/(np.sqrt(wac_scale))),
                              int(wac_rgb.shape[1]/(np.sqrt(wac_scale)))),
                             cv2.INTER_AREA)
            wac_rgb = cv2.resize(wac_rgb,
                             (feed_width, feed_height), cv2.INTER_CUBIC)
            '''

            wac_rgb = cv2.resize(wac_rgb,
                             (155,58),
                             cv2.INTER_AREA)
            wac_rgb = cv2.resize(wac_rgb,
                             (feed_width, feed_height), cv2.INTER_CUBIC)


            # render saccadecam
            saccade_rgb = foveate(wac_rgb, fovea)

            '''
            plt.subplot(131)
            plt.imshow(saccade_rgb)
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(att)
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(fovea)
            plt.axis('off')
            plt.show()
            '''
            
            # send to gpu
            focused_rgb = transforms.ToTensor()(focused_rgb).unsqueeze(0).to(device)
            target_rgb = transforms.ToTensor()(target_rgb).unsqueeze(0).to(device)
            saccade_rgb = transforms.ToTensor()(saccade_rgb).unsqueeze(0).to(device)

            '''
            #THIS IS FOR KITTI SIMULATION
            # get deformable attention
            deform_att = attention_dec(encoder(wac))[("disp", 0)]
            oracle_mask = torch.zeros_like(deform_att)
            num = int(saccade_total[0]*saccade_total[1])
            for i in range(deform_att.size(0)):
                oracle_mask[i] = deform_att[i] >= deform_att[i].flatten().topk(num)[0].min()
            deform_att *= oracle_mask

            
            # find patch fovea based on greedy algo
            patch_att = greedy_torch.greedy_mask(deform_att, args.num_fovea, fovea_shape)

            # foveate
            foveated = input_image*patch_att + wac*(1-patch_att)
            '''

            ###########
            # get depth
            ###########
            focused_disp = disp_to_depth(focused_decoder(focused_encoder(focused_rgb))[("disp", 0)])
            target_disp = disp_to_depth(target_decoder(target_encoder(target_rgb))[("disp", 0)])
            saccade_disp = disp_to_depth(saccade_decoder(saccade_encoder(saccade_rgb))[("disp", 0)])

            print(focused_disp.shape)
            
            # save disp
            focused_disp_colored = color(focused_disp, output_directory, "focused")
            target_disp_colored = color(target_disp, output_directory, "target")
            saccade_disp_colored = color(saccade_disp, output_directory, "saccade")

            print(focused_disp.shape)
            
            # send to cpu
            focused_rgb = focused_rgb.cpu().numpy()[0].transpose((1,2,0))
            target_rgb = target_rgb.cpu().numpy()[0].transpose((1,2,0))
            saccade_rgb = saccade_rgb.cpu().numpy()[0].transpose((1,2,0))
            focused_disp = focused_disp.cpu().numpy()[0].transpose((1,2,0))[:,:,0]
            target_disp = target_disp.cpu().numpy()[0].transpose((1,2,0))[:,:,0]
            saccade_disp = saccade_disp.cpu().numpy()[0].transpose((1,2,0))[:,:,0]
            
            
            # get errors
            target_errors = compute_errors(focused_disp, target_disp)
            saccade_errors = compute_errors(focused_disp, saccade_disp)

            e = np.array(target_errors) - np.array(saccade_errors)

            if (e[:4]>0).all() and (e[4:]<0).all():
                #if e is not None:
                
                print("\n \n \n FRAME: "+str(pmin+idx)+" \n target \n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                print(("{: 8.3f} | " * 7).format(*target_errors))

                print('saccade' +"\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                print(("{: 8.3f} | " * 7).format(*saccade_errors))


                fig, axs = plt.subplots(1, 6, figsize=(17,5), dpi=200)
                fig.subplots_adjust(hspace = 0.1, wspace=0.01)
                axs = axs.ravel()

                axs[0].imshow(target_rgb)
                axs[1].imshow(saccade_rgb)
                axs[2].imshow(att)
                axs[3].imshow(fovea)
                axs[4].imshow(target_disp_colored)
                axs[5].imshow(saccade_disp_colored)
                
                
                for i in range(6):
                    axs[i].axes.get_xaxis().set_visible(False)
                    axs[i].axes.get_yaxis().set_visible(False)
                    axs[i].axis('off')

                fig.tight_layout()
                plt.show()

            
            
            '''
            plt.subplot(1,3,1)
            plt.imshow(focused_disp)
            plt.subplot(1,3,2)
            plt.imshow(target_disp)
            plt.subplot(1,3,3)
            plt.imshow(saccade_disp)
            plt.show()
            '''
            '''
            mosaic[idx,1] = np.array(im).astype(np.float32) / 255.
            if args.data_rep == "focused":
                mosaic[idx,0] = focused.astype(np.float32)
            if args.data_rep == "defocused":
                mosaic[idx,0] = input_image.astype(np.float32)
            if args.data_rep == "foveated":
                mosaic[idx,0] = saccade_render.astype(np.float32)
            '''

        '''
        fig, axs = plt.subplots(len(defoc_paths), 2, figsize=(9,12), dpi=200, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 0.1, wspace=0.01)
        axs = axs.ravel()
        for i in range(len(defoc_paths)*2):
            axs[i].imshow(mosaic[i//2, i%2])
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
            
        plt.show()
        '''
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_hardware(args)
