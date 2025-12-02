#!/bin/bash

# This is an example of how to run the batch evaluation script.
# You should replace the placeholder values with your actual file paths and desired parameters.

input_json_file_name="eval.json"
output_dir_name="noise_strength_1"
random_cache_interval="False"
noise_strength="0.1"
device="cuda"
# 遍历noise_strength，从0.1到0.8，每次增加0.1
for noise_strength in $(seq 0.1 0.1 0.9)
do
    # 0.2对应noise_strength_2，0.3对应noise_strength_3，依此类推
    # 将小数乘以10取整，得到对应的序号
    index=$(echo "$noise_strength * 10" | bc | cut -d'.' -f1)
    output_dir_name="noise_strength_${index}"
    echo "noise_strength: $noise_strength, output_dir_name: $output_dir_name"

    cd vid2vid
    python ./batch_eval.py \
        --json_file "/home/zrj/project/ori_v2v/streamv2v/vid2vid/source_video/$input_json_file_name" \
        --output_dir "/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/$output_dir_name" \
        --random_cache_interval $random_cache_interval \
        --noise_strength $noise_strength 
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

