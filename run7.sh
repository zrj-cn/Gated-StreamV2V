#!/bin/bash

# 输入的序列
input_json_file_name="eval.json"

output_dir_name="ttt_lr_0.6"

# TTT缓存相关设置
use_ttt_cache="True" # default: false
reverse_tag="Flase" # default: false
ttt_lr=0.6 # 更新cache时的学习率，default: 0.5
use_attn_concat="True" # 自注意力机制是否连接 default: false

# 特征注入相关设置
use_feature_injection="True" # default: false
feature_similarity_threshold=0.98 # 特征相似度阈值，default: 0.98
feature_injection_strength=0.5 # 特征注入时的强度，default: 0.5 TTT模式下不起效

# 其他信息
cache_interval="1"
noise_strength="0.4"

random_cache_interval="False" # 仅针对原本的模式

# 可见cuda设备
device="cuda"
cuda_visible_devices="7"

cd vid2vid
python ./batch_eval.py \
    --json_file "/home/zrj/project/ori_v2v/streamv2v/vid2vid/source_video/$input_json_file_name" \
    --output_dir "/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/$output_dir_name" \
    --random_cache_interval $random_cache_interval \
    --noise_strength $noise_strength \
    --cache_interval $cache_interval \
    --cuda_visible_devices $cuda_visible_devices \
    --use_ttt_cache $use_ttt_cache \
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

