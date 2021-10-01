#*********************
# repeats oracle table
#*********************
###################
# BANDWIDTH (31.3 target, 27 -> 15 WAC)
###################

# target 20 epochs
#python ../test_conventional.py --load_weights_folder ~/tmp/defocused_0.2_0.0/models/weights_19/ --eval_stereo --fovea 1 --defocused_scale 0.2 --wac_scale 0.0  

# target finetuned
#python ../test_saccadecam.py --load_weights_folder ~/tmp/defocused_FT_0.2_0.0/models/weights_9/ --eval_stereo --fovea 1 --defocused_scale 0.2 --wac_scale 0.25  --frame_ids 0 --gpus 0 


# get oracle0 (replace wac-gt regions with focused depth)
#wac 0.75
#python ../test_saccadecam.py --load_weights_folder ~/tmp/wac_0.2_0.75/models/weights_19/ \
#       --eval_stereo --fovea 0 --defocused_scale 0.2 --wac_scale 0.75 --gpus 0 --oracle0 --frame_ids 0

#wac 0.5
#python ../test_saccadecam.py --load_weights_folder ~/tmp/wac_0.2_0.5/models/weights_19/ \
#       --eval_stereo --fovea 0 --defocused_scale 0.2 --wac_scale 0.5 --gpus 0 --oracle0 --frame_ids 0

#wac 0.25
#python ../test_saccadecam.py --load_weights_folder ~/tmp/wac_0.2_0.25/models/weights_19/ \
#       --eval_stereo --fovea 0 --defocused_scale 0.2 --wac_scale 0.25 --gpus 0 --oracle0 --frame_ids 0

# get oracleC (replace wac-focused regions with focused depth)
# wac 0.75
#python ../test_saccadecam.py --load_weights_folder ~/tmp/wac_0.2_0.75/models/weights_19/ \
#       --eval_stereo --fovea 0 --defocused_scale 0.2 --wac_scale 0.75 --gpus 0 --oracleC --frame_ids 0

#wac 0.5
#python ../test_saccadecam.py --load_weights_folder ~/tmp/wac_0.2_0.5/models/weights_19/ \
#       --eval_stereo --fovea 0 --defocused_scale 0.2 --wac_scale 0.5 \
#       --gpus 0 --oracleC --frame_ids 0 

# wac 0.25
#python ../test_saccadecam.py --load_weights_folder ~/tmp/wac_0.2_0.25/models/weights_19/ \
#       --eval_stereo --fovea 0 --defocused_scale 0.2 --wac_scale 0.25 \
#       --gpus 0 --oracleC --frame_ids 0 

#*******************
# repeats main table
#*******************
###################
# BANDWIDTH (35 target, 30.31 WAC)
###################
# full res
python ../test_conventional.py --load_weights_folder ~/tmp/focused/models/weights_19/ --eval_stereo --fovea 2

# target
#python ../test_conventional.py --load_weights_folder ~/tmp/defocused_0.25_0.0/models/weights_19/ --eval_stereo --fovea 1 --defocused_scale 0.25 --wac_scale 0.0  

# wac
#python ../test_conventional.py --load_weights_folder ~/tmp/wac_0.25_0.75/models/weights_19/ --eval_stereo --fovea 0 --defocused_scale 0.25 --wac_scale 0.75  

# ours without weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.25 --wac_scale 0.75 --deformable \
#       --frame_ids 0 --gpus 0 --epoch_to_load 16

# ours with weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.25 --wac_scale 0.75 --frame_ids 0 --gpus 0 \
#       --deformable --weight_regions --epoch_to_load 6

# edges without weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.25 --wac_scale 0.75 --frame_ids 0 \
#       --gpus 0 --comparison edges --epoch_to_load 16 --deformable

# edges with weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.25 --wac_scale 0.75 --frame_ids 0 \
#       --gpus 0 --comparison edges --weight_regions --epoch_to_load 6 --deformable

#####################################################

###################
# BANDWIDTH (27.11 target, 23.48 WAC)
###################

# target
#python ../test_conventional.py --load_weights_folder ~/tmp/defocused_0.15_0.0/models/weights_19/ --eval_stereo --fovea 1 --defocused_scale 0.15 --wac_scale 0.0  

# wac
#python ../test_conventional.py --load_weights_folder ~/tmp/wac_0.15_0.75/models/weights_19/ --eval_stereo --fovea 0 --defocused_scale 0.15 --wac_scale 0.75  

# ours without weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.15 --wac_scale 0.75 --deformable \
#       --frame_ids 0 --gpus 0 --epoch_to_load 11 


# ours with weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.15 --wac_scale 0.75 --deformable \
#       --frame_ids 0 --gpus 0 --epoch_to_load 13 --weight_regions


# edges without weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.15 --wac_scale 0.75 --frame_ids 0 \
#       --gpus 0 --comparison edges --epoch_to_load 11 --deformable

# edges with weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.15 --wac_scale 0.75 --frame_ids 0 \
#       --gpus 0 --comparison edges --weight_regions --epoch_to_load 13 --deformable


###################
# BANDWIDTH (7.82 target, 6.78 WAC)
###################

# target
#python ../test_conventional.py --load_weights_folder ~/tmp/defocused_0.0125_0.0/models/weights_19/ --eval_stereo --fovea 1 --defocused_scale 0.0125 --wac_scale 0.0  

# wac
#python ../test_conventional.py --load_weights_folder ~/tmp/wac_0.0125_0.75/models/weights_19/ --eval_stereo --fovea 0 --defocused_scale 0.0125 --wac_scale 0.75  

# ours without weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.0125 --wac_scale 0.75 --deformable \
#       --frame_ids 0 --gpus 0 --epoch_to_load 1 


# ours with weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.0125 --wac_scale 0.75 --deformable \
#       --frame_ids 0 --gpus 0 --epoch_to_load 0 --weight_regions


# edges without weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.0125 --wac_scale 0.75 --frame_ids 0 \
#       --gpus 0 --comparison edges --epoch_to_load 1 --deformable

# edges with weighting
#python ../test_saccadecam.py \
#       --eval_stereo --fovea 0 --defocused_scale 0.0125 --wac_scale 0.75 --frame_ids 0 \
#       --gpus 0 --comparison edges --weight_regions --epoch_to_load 0 --deformable



