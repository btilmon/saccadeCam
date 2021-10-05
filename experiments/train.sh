

####################
# NO REGION WEIGHTING
#####################
# CUDA_VISIBLE_DEVICES=0 python ../train_main.py \
# 		    --model_name attention/oracleC_weight_old_scratch_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.15 --wac_scale 0.75  \
# 		    --disable_automasking &

# CUDA_VISIBLE_DEVICES=3 python ../train_main.py \
# 		    --model_name attention/oracleC_weight_old_scratch_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.15 --wac_scale 0.5  \
# 		    --disable_automasking &

# CUDA_VISIBLE_DEVICES=2 python ../train_main.py \
# 		    --model_name attention/oracleC_weight_old_scratch_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.25 --wac_scale 0.75  \
# 		    --disable_automasking &

# CUDA_VISIBLE_DEVICES=4 python ../train_main.py \
# 		    --model_name attention/oracleC_weight_old_scratch_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.25 --wac_scale 0.5  \
# 		    --disable_automasking 



####################
# REGION WEIGHTING
#####################
# CUDA_VISIBLE_DEVICES=0 python ../train_main.py \
# 		    --model_name attention/oracleC_weighted_1.75_0.25_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.25 --wac_scale 0.75  \
# 		    --disable_automasking --weight_regions --fovea_weight 0.15 &

# CUDA_VISIBLE_DEVICES=2 python ../train_main.py \
# 		    --model_name attention/oracleC_weighted_1.75_0.25_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.2 --wac_scale 0.75  \
# 		    --disable_automasking --weight_regions --fovea_weight 0.2 &

# CUDA_VISIBLE_DEVICES=3 python ../train_main.py \
# 		    --model_name attention/oracleC_weighted_1.75_0.25_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.15 --wac_scale 0.75  \
# 		    --disable_automasking --weight_regions --fovea_weight 0.25 &

# CUDA_VISIBLE_DEVICES=4 python ../train_main.py \
# 		    --model_name attention/oracleC_weighted_1.75_0.25_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.0125 --wac_scale 0.75  \
# 		    --disable_automasking --weight_regions --fovea_weight 0.5 &

# CUDA_VISIBLE_DEVICES=5 python ../train_main.py \
# 		    --model_name attention/oracleC_weighted_1.75_0.25_ \
# 		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
# 		    --depth_lr 1e-4 --oracleC \
# 		    --defocused_scale 0.0125 --wac_scale 0.5  \
# 		    --disable_automasking --weight_regions --fovea_weight 0.5 

############################
# repeat results from paper
###########################

# train depth --oracleC
CUDA_VISIBLE_DEVICES=0 python ../train_main.py \
		    --model_name attention/RR_depth_ \
		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 20 \
		    --depth_lr 1e-4 --oracleC \
		    --defocused_scale 0.25 --wac_scale 0.75  \
		    --disable_automasking --weight_regions --fovea_weight 0.15 

# train attention --deformable
CUDA_VISIBLE_DEVICES=0 python ../train_main.py \
		    --model_name attention/RR_deformable_ \
		    --use_stereo --frame_ids 0 --split eigen_full --fovea 0 --num_epochs 10 \
		    --depth_lr 1e-4 --deformable \
		    --defocused_scale 0.25 --wac_scale 0.75  \
		    --disable_automasking &
