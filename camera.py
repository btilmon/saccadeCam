"""
/* HARDWARE *\
SaccadeCam and WAC = FLIR Blackfly S: BFS-U3-16S2C-CS
"""

import numpy as np
import math as m
import sys
import cv2


class Camera():
    """
    /*Camera class simulates angular resolution*\

    main idea:
    ---------
    - Simulate SaccadeCam, wide angle camera, and defocused camera whose 
      resolutions are intertwined based on camera parameters.

    important parameters:
    -------------------
    - down = Downsample ground truth to save training time.
    - num_fovea = How many fovea to image.
    - fovea_percent = How much of native FOV a single fovea consumes.
    - periphery_scale = How much to blur wac and defocused. See conceptual eqn in WAC().
    """
    def __init__(self, params):
        self.fov_mm = params['resolution_mm']
        self.defocused_scale = params['defocused_scale']
        self.wac_scale = params['wac_scale']
        self.aspect = params['resolution_px'][0] / params['resolution_px'][1]
        self.native_res = params['resolution_px']
        self.native_ang = self.native_res / self.fov_mm
        #self.sensor_mm = params['resolution_mm'] 

        # initialize cameras
        self.Defocused()
        self.WAC()
        self.SaccadeCam()
        
    def SaccadeCam(self):
        """
        Select number of Saccadecam pixels such that the average 
        angular resolution after upsampling is equal to defocused
        """
        self.saccade_px = self.reshape(((self.defocused_ang.mean() - self.wac_ang.mean()) *
                      self.native_res[0]*self.native_res[1]) / (self.native_ang.mean() - self.wac_ang.mean()))
        self.saccade_ang = self.native_ang
        

    def Defocused(self):
        """Set Defocused camera where Saccade+WAC = Defocused (after overlay!!!)"""
        #self.defocused_px = self.reshape(self.wac_px[0]*self.wac_px[1] + self.saccade_px[0]*self.saccade_px[1])
        self.defocused_px = np.sqrt(self.defocused_scale) * self.native_res
        self.defocused_ang = self.defocused_px / self.fov_mm

        
    def WAC(self):
        """Set wide angle camera parameters"""

        # conceptually:  (1-fovea_percent*num_fovea)*wac + saccade = periphery_scale * native_res
        '''
        sensor_px = self.reshape(((self.periphery_scale*self.native_res[0]*self.native_res[1]) - (self.saccade_px[0]*self.saccade_px[1])) / (1-(self.fovea_percent*self.num_fovea)))
        ang = sensor_px / self.fov_mm
        self.wac_fov_mm = self.reshape(self.fov_mm[0]*self.fov_mm[1] - self.saccade_fov[0]*self.saccade_fov[1])
        self.wac_px = ang * self.wac_fov_mm
        self.wac_ang = self.wac_px / self.fov_mm
        '''
        self.wac_px = np.sqrt(self.wac_scale) * self.defocused_px
        self.wac_ang = self.wac_px / self.fov_mm

    def reshape(self, vec):
        """reshapes 1D to 2D"""
        vec = np.array([ np.sqrt(vec), np.sqrt(vec)])
        c = (vec[0] - self.aspect * vec[1]) / (1 + self.aspect)
        vec[0] -= c
        vec[1] += c
        return vec        


    
def load_camera(args=None):

    '''
    # kinect
    kinect_params  = {'resolution_px': np.array([480, 640]),
                      'resolution_mm': np.array([5.08, 6.52]),
                      'fov_mm': np.array([4894, 6281]),
                      'lens': 5.19,
                      'num_fovea': None,
                      'defocused_scale': None,
                      'wac_scale': None}

    if args: kinect_params['num_fovea'] = args.num_fovea
    if args: kinect_params['down'] = args.spatial_scale
    if args: kinect_params['defocused_scale'] = args.defocused_scale
    if args: kinect_params['wac_scale'] = args.wac_scale 
    return kinect_params
    '''

    res = np.array([4.80, 6.4])
        
    # kitti camera
    kitti_params  = {'resolution_px': None,
                     'resolution_mm': res,
                      'defocused_scale': None,
                      'wac_scale': None}

    if args: kitti_params['resolution_px'] = np.array([args.height, args.width])
    if args: kitti_params['defocused_scale'] = args.defocused_scale
    if args: kitti_params['wac_scale'] = args.wac_scale 
    return kitti_params



def make_even(h, w):
    """make image dims even"""
    if h % 2:
        h += 1
    if w % 2:
        w += 1
    return h, w
    
def get_resolutions(args):
    """Get resolution from class"""
    # load camera parameters and initialize
    cam_params = load_camera(args)
    cam = Camera(cam_params)

    # load either wac or defocused based on args
    if args.fovea == 0:
        px = cam.wac_px
    if args.fovea == 1:
        px = cam.defocused_px
    if args.fovea == 2:
        px = cam.native_res
    native = cam.native_res
    h, w = native[0].astype(int), native[1].astype(int) # should be [480, 640] // down
    px = px.astype(int)

    ratio = (native[0]*native[1]) / (px[0]*px[1])
    #saccade_total = cam.reshape(cam.saccade_px[0] * cam.saccade_px[1]).astype(int)
    '''
    if args.num_fovea is not None:
        fovea_px = cam.reshape(saccade_total * args.num_fovea)
    else:
        fovea_px = 0
    '''
    #print('camera',fovea_px[0]*fovea_px[1]*args.num_fovea)
    
    # ensure even native res
    h, w = make_even(h, w)
    
    return {'h':h, 'w':w, 'px':px,
            'saccade_ang':cam.saccade_ang.mean(),
            'defocused_ang':cam.defocused_ang.mean(),
            'wac_ang':cam.wac_ang.mean(),
            'saccade_total':cam.saccade_px}

def process(image, args):
    """   
    /*fovea options*\
    0 = load WAC 
    1 = load defocused
    2 = load focused
    """
    # sampling methods
    interp_shrink = cv2.INTER_AREA
    interp_grow = cv2.INTER_CUBIC
    
    res = get_resolutions(args)
    h, w = res['h'], res['w']
    px = res['px']
    
    # lower spatial res
    image = np.array(image)
    image = cv2.resize(image, (w, h), interpolation=interp_shrink)
    focused = np.copy(image)

    # lower angular res
    #print('before',image.shape)
    image = cv2.resize(image, (px[1], px[0]), interpolation=interp_shrink)
    #print('after', image.shape, '\n')
    image = cv2.resize(image, (w, h), interpolation=interp_grow)
    # resize depth
    #depth = cv2.GaussianBlur(depth, (3,3), 0)
    #depth = cv2.resize(depth, (w//2, h//2), interpolation=interp_shrink)
    #depth = depth[:,:,np.newaxis]
    return image

def unit_test():
    params = load_camera()
    #params['num_fovea'] = 10
    

    params['defocused_scale'] = 0.25 # .65, .7, .8, .85
    params['wac_scale'] = .75#, .5, .25 # .65, .7, .8, .85
    params['resolution_px'] = np.array([1080, 1440]) 
    params['resolution_mm'] = np.array([3.73, 4.97])

    aspect = np.array([300, 804]) / params['resolution_px'] 
    params['resolution_mm'] *= aspect
    params['resolution_px'] = params['resolution_px']*aspect

    # decimation
    focused_scale = 17
    target_scale = 70
    wac_scale = 90
    params['resolution_px'] = params['resolution_px'] / np.sqrt(focused_scale)
    import matplotlib.pyplot as plt
    import cv2
    foc = plt.imread('hardware_results/data/imgs/3/defoc_12.png')
    x = cv2.resize(foc, (195, 195))
    x = cv2.resize(foc, (804, 804))

    y = cv2.resize(foc, (98, 98))
    y = cv2.resize(y, (804, 804))

    z = cv2.resize(foc, (84, 84))
    z = cv2.resize(z, (804, 804))

    '''
    plt.subplot(131)
    plt.imshow(x)
    plt.subplot(132)
    plt.imshow(y)
    plt.subplot(133)
    plt.imshow(z)
    plt.show()
    '''

    cam = Camera(params=params)
    
    print('focused ang \n', cam.native_ang.mean(), '\n')
    print('focused px \n', cam.native_res, '\n')
    
    print('target ang \n', cam.defocused_ang.mean(), '\n')
    print('target px \n', cam.defocused_px, '\n')

    print('wac ang \n', cam.wac_ang.mean(), '\n')
    print('wac px \n', cam.wac_px, '\n')

    print(params['resolution_px'] / np.sqrt(4))
    print(params['resolution_px'] / np.sqrt(5.5))
    
    #print('saccade ang \n', cam.saccade_ang.mean(), '\n')
    
#unit_test()
