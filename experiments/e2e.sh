CUDA_VISIBLE_DEVICES=2 python ../train.py --model_name attention/e2e_ \
		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 5 \
		    --depth_lr 1e-7 --e2e \
		    --defocused_scale 0.2 --wac_scale 0.5 --num_fovea 0 \
		    --exp saliency --disable_automasking 
