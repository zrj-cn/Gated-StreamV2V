#!/bin/bash

# This is an example of how to run the batch evaluation script.
# You should replace the placeholder values with your actual file paths and desired parameters.

input_json_file_name="eval.json"
cache_interval="6"
cuda_visible_devices="6"
output_dir_name="cache_interval_${cache_interval}"
random_cache_interval="False"
noise_strength="0.4"
device="cuda"

cd vid2vid
python ./batch_eval.py \
    --json_file "/home/zrj/project/ori_v2v/streamv2v/vid2vid/source_video/$input_json_file_name" \
    --output_dir "/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/$output_dir_name" \
    --random_cache_interval $random_cache_interval \
    --noise_strength $noise_strength \
    --cache_interval $cache_interval \
    --cuda_visible_devices $cuda_visible_devices
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

