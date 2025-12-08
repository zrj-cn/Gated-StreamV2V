#!/bin/bash

input_json_file_name="eval.json"

output_dir_name="similarity_gate_default"

cached_attn_style="similarity" # default: similarity
# cached_attn_style="origin" 
# cached_attn_style="confidence" 

reverse_tag="True" # default: True
ttt_lr=1.0 # scale for beta，default: 1.0
use_attn_concat="True" # Whether to concatenate self-attention mechanism, default: True

use_feature_injection="True" # default: True
feature_similarity_threshold=0.98 # Feature similarity threshold, default: 0.98
feature_injection_strength=0.5 # strength of feature injection in original StreamV2V, default: 0.5

# 其他信息
cache_interval="1"
noise_strength="0.4"


# 可见cuda设备
device="cuda"
cuda_visible_devices="7"


cd vid2vid
python ./batch_eval.py \
    --json_file "/home/zrj/project/ori_v2v/streamv2v/vid2vid/source_video/$input_json_file_name" \
    --output_dir "/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/$output_dir_name" \
    --noise_strength $noise_strength \
    --cache_interval $cache_interval \
    --cuda_visible_devices $cuda_visible_devices \
    --cached_attn_style $cached_attn_style \
    --reverse_tag $reverse_tag \
    --use_attn_concat $use_attn_concat \
    --use_feature_injection $use_feature_injection \
    --feature_similarity_threshold $feature_similarity_threshold \
    --feature_injection_strength $feature_injection_strength \
    --ttt_lr $ttt_lr
cd ..

cd tools
python ./clip_score.py \
    --device $device \
    --method_version $output_dir_name \
    --set_file_path "./user_study_upload/$input_json_file_name" \
    --cuda_visible_devices $cuda_visible_devices


python ./warp_error.py \
    --device $device \
    --method_version $output_dir_name \
    --set_file_path "./user_study_upload/$input_json_file_name" \
    --cuda_visible_devices $cuda_visible_devices
cd ..

