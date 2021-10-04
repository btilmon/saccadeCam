# SaccadeCam

This is the PyTorch implementation for:

> SaccadeCam: Adaptive Visual Attention for Monocular Depth Sensing
>
> [Brevin Tilmon](https://btilmon.github.io/) and [Sanjeev Koppal](http://focus.ece.ufl.edu/people/)
>
> [ICCV 2021 (arXiv pdf)](https://arxiv.org/abs/2103.12981)


If you find our work useful in your research please consider citing our paper:

```
@misc{tilmon2021saccadecam,
title={SaccadeCam: Adaptive Visual Attention for Monocular Depth Sensing},
author={Brevin Tilmon and Sanjeev J. Koppal},
year={2021},
eprint={2103.12981},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```


## Data

Download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:

```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```

Following monodepth2, our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:

```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

You can skip this step and include the `--png` flag, but the results will be slightly different and the loading times will be slower.


## Dependencies

This code was developed on Ubuntu 16.04 with a NVIDIA GTX 1080 Ti.

* opencv-python==4.5.1.48
* protobuf==3.14.0
* tensorboardx==1.4
* torch==1.7.1
* torchvision==0.8.2


## Training

The models and tensorboard events are saved to `~/tmp/<model_name>` by default. This can be changed with the `--log_dir` flag.

#### Bandwidth flags

`--defocused_scale` was set to 0.25, 0.2, 0.15, 0.0125.

`--wac_scale` was always 0.75 in our results. 

See `camera.py` for how the following bandwidth flags affect image resolution. Conceptually, `--defocused_scale` is the percentage of full resolution pixels used to form the target resolution. `--wac_scale` is the percentage of target resolution used to form the periphery in a SaccadeCam rendered image, before full resolution fovea are added.

See `experiments/train.sh` to train various models that should closely repeat results from our paper. An example from `train.sh` for training our network is as follows:

```shell
CUDA_VISIBLE_DEVICES=0 python ../train_main.py \
		    --model_name attention/RR_depth_ \
		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
		    --depth_lr 1e-4 --oracleC \
		    --defocused_scale 0.25 --wac_scale 0.75  \
		    --disable_automasking --weight_regions --fovea_weight 0.15 
```

`--oracleC` means we train the depth network and not the attention network. See `experiments/deformable_train.sh` to train attention networks.

`--fovea` = 0 for loading WAC resolutions, 1 for loading target resolutions, and 2 for full resolution.

`--weight_regions` weights the fovea regions to encourage the network to focus on them more. `--fovea_weight` determines how much more to weight the fovea regions. See `trainer_main.py` for more.

### Testing

`test.sh` repeats the tables in our paper. You will need to change where the models are loaded from to test your own results. An example from `test.sh` for testing our network is as follows:

```shell
python ../test.py \
       --eval_stereo --fovea 0 --defocused_scale 0.25 --wac_scale 0.75 --frame_ids 0 --gpus 0 \
       --deformable --weight_regions --epoch_to_load 6
```

As a guide, the below table explains which epochs of models were used for our results. We can consistently reproduce results with these epochs.

| defocused/wac scales | no weighting epochs | weighting epochs |
|----------------------|---------------------|------------------|
|0.25/.75              |17                   |7                 |
|0.2/0.75              |13                   |16                |
|0.15/0.75             |11                   |14                |
|0.0125/0.75           |2                    |1                 |

## Copyright and License Statement

The network encoder and decoder architectures and reprojection layers were adapted from [monodepth2](https://github.com/nianticlabs/monodepth2). We reimplement `layers.py`, `resnet_encoder.py`, and `depth_decoder.py` based on their paper to avoid infringing on their license. Our contributions aim for improving existing self supervised depth techniques, such as monodepth2, with adaptive resolution placement. 


