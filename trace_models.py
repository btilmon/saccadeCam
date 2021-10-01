'''
- Trace models to c++
- Have to use decoder with different i/o in order to use torch.jit.trace()
'''
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms#, datasetsc

import networks
import networks.depth_decoder_trace as DDT
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
import camera
import greedy_torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images')
    parser.add_argument('--output_path', type=str,
                        help='path to a test image or folder of images')
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
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
                        default=0.15)
    parser.add_argument("--wac_scale",
                        type=float,
                        help="% of defocused resolution",
                        default=0.75)
    parser.add_argument("--fovea",
                        type=int,
                        default=0)
    parser.add_argument("--num_fovea",
                        type=int,
                        default=1)

    parser.add_argument("--gpus", type=str, default= '7')
    
    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """


    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device=torch.device("cpu")
        
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpus

    #download_model_if_doesnt_exist(args.model_name)

    #  path = "~/tmp/attention/oracleC_weighted_1.75_0.25_0.15_0.75/models/weights_15/"
    
    if args.data_rep == "foveated":
        path = "~/tmp/attention/FT_squares_0.25_0.75/models/weights_9/"
        #path = "~/tmp/defocused_0.15_0.0/models/weights_19/"#
        #path = "~/tmp/attention/oracleC_weight_old_scratch_0.15_0.75/models/weights_9/" #11
    if args.data_rep == "focused":
        path = "~/tmp/focused/models/weights_19/"
    if args.data_rep == "defocused":
        path = "~/tmp/defocused_0.15_0.0/models/weights_19/"
        
    #model_path = os.path.join("models", args.model_name)
    model_path = os.path.expanduser(path)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = DDT.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    att_path = "~/tmp/attention/deformable_0.25_0.75/models/weights_3/saliency.pth"
    att_dec_path = os.path.expanduser(att_path)

    # load attention decoder
    attention_dec = DDT.DepthDecoder(
        encoder.num_ch_enc, [0],
        num_output_channels=1)
    att_dict = torch.load(att_dec_path)
    model_dict = attention_dec.state_dict()
    attention_dec.load_state_dict(torch.load(att_dec_path))
    attention_dec.to(device)
    attention_dec.eval()


    # Begin tracing

    image = torch.rand(1,3,192,640).to(device)

    traced_encoder = torch.jit.trace(encoder, image, strict=False) # strict=False allows list output

    #attention_dec(example)

    '''
    ex_dict = {"0":example[0], "1":example[1],
               "2":example[2], "3":example[3], "4":example[4]}
    '''
    feat = encoder(image)
    #a,b,c,d,e = feat[0], feat[1], feat[2], feat[3], feat[4]
    #depth_in = (( feat[0], feat[1], feat[2], feat[3], feat[4] ))
    traced_att_decoder = torch.jit.trace(attention_dec, [feat])
    traced_decoder = torch.jit.trace(depth_decoder, [feat])


    # prepare focused image
    input_image = pil.open('traced_models/kitti.jpg').convert('RGB')#)[300:900, 300:1200])
    input_image = input_image.resize((640, 192), pil.LANCZOS)

    # gpu
    im = transforms.ToTensor()(input_image).unsqueeze(0)            

    out = traced_encoder(im)
    depth = traced_decoder(out)
    disp_np = depth.squeeze().detach().numpy()

    # Saving colormapped depth image
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save('depth2.jpeg')


    
    traced_encoder.save('traced_models/foveated_encoder.pt')
    #traced_att_decoder.save('traced_models/traced_att_decoder_att.pt')
    traced_decoder.save('traced_models/foveated_decoder.pt')

    
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
