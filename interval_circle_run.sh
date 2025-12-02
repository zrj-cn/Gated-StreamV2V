#!/bin/bash

# This is an example of how to run the batch evaluation script.
# You should replace the placeholder values with your actual file paths and desired parameters.

input_json_file_name="eval.json"
output_dir_name="cache_interval_1"
random_cache_interval="False"
noise_strength="0.4"
device="cuda"
cache_interval="1"
# 遍历noise_strength，从0.1到0.8，每次增加0.1
for cache_interval in $(seq 1 1 8)
do
    # 如果cache_interval为4，跳过
    if [ "$cache_interval" -eq 4 ]; then
        continue
    fi
    # 1对应cache_interval_1，2对应cache_interval_2，依此类推
    index=$(echo "$cache_interval" | bc | cut -d'.' -f1)
    output_dir_name="cache_interval_${index}"
    echo "cache_interval: $cache_interval, output_dir_name: $output_dir_name"

    cd vid2vid
    python ./batch_eval.py \
        --json_file "/home/zrj/project/ori_v2v/streamv2v/vid2vid/source_video/$input_json_file_name" \
        --output_dir "/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/$output_dir_name" \
        --random_cache_interval $random_cache_interval \
        --noise_strength $noise_strength \
        --cache_interval $cache_interval
    cd ..

    cd tools

    python ./clip_score.py \
        --device $device \
        --method_version $output_dir_name \
        --set_file_path "./user_study_upload/$input_json_file_name"

    python ./warp_error.py \
        --device $device \
        --method_version $output_dir_name \
        --set_file_path "./user_study_upload/$input_json_file_name"
        
    cd ..
done

