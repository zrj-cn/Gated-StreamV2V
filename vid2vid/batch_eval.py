import os
import json
import argparse
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a JSON file contains multiple edits.")
    parser.add_argument('--cuda_visible_devices', type=str, default="7", help='CUDA visible devices.')
    parser.add_argument('--json_file', type=str, help='The path to the JSON file to process.')
    parser.add_argument('--output_dir', type=str, default="/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/motion_strength_1", help='The directory to save the output videos.')
    parser.add_argument('--random_cache_interval', type=str, default="False", help='Whether to use random cache interval.')
    parser.add_argument('--cache_interval', type=int, default=4, help='Cache interval for processing.')
    parser.add_argument('--noise_strength', type=str, default=None, help='Noise strength to use for all videos. Overrides the value in the json file.')
    parser.add_argument('--use_ttt_cache', type=str, default="False", help='Whether to use TTT cache.')
    parser.add_argument('--reverse_tag', type=str, default="False", help='Whether to use reverse tag.')
    parser.add_argument('--use_attn_concat', type=str, default="False", help='Whether to use attn concat.')
    parser.add_argument('--use_feature_injection', type=str, default="True", help='Whether to use feature injection.')
    parser.add_argument('--feature_similarity_threshold', type=float, default=0.98, help='Feature similarity threshold.')
    parser.add_argument('--feature_injection_strength', type=float, default=0.5, help='Feature injection strength.')
    parser.add_argument('--ttt_lr', type=float, default=0.5, help='TTT learning rate.')

    return parser.parse_args()

data = []
args = parse_arguments()
json_file = args.json_file

# Load the JSON data
with open(json_file, "r") as file:
    for line in file:
        data.append(json.loads(line))

for item in data:
    file_path = item["file_path"]
    src_vid_name = item["src_vid_name"]
    prompt = item["prompt"]
    diffusion_steps = item["diffusion_steps"]
    json_noise_strength = item["noise_strength"]
    
    if args.noise_strength is not None:
        noise_strength = args.noise_strength
    else:
        noise_strength = json_noise_strength

    try:
        # TODO -ZRJ-json文件中的model_id需要修改
        model_id = item["model_id"]
    except:
        # TODO -ZRJ-如果需要验证，请修改此处
        model_id = "/home/zrj/project/ori_v2v/streamv2v/data/checkpoints/stable-diffusion-1.5"
        # model_id = "Jiali/stable-diffusion-1.5"
    
    try:
        video_name = item["vid_name"]
    except:
        video_name = None

    if video_name is not None:
        out_put_video = f"{args.output_dir}/{video_name}.mp4"
        # 如果存在这个video，则跳过
        if os.path.exists(out_put_video):
            print(f"视频已存在：{out_put_video}")
            continue
        command = [
            'python', "main.py",
            "--input", f"{file_path}/{src_vid_name}.mp4",
            "--prompt", prompt,
            "--cuda_visible_devices", args.cuda_visible_devices,
            "--video_name", video_name,
            "--output_dir", args.output_dir,
            "--model_id", model_id,
            "--diffusion_steps", diffusion_steps,
            "--noise_strength", noise_strength,
            "--acceleration", "xformers",
            "--use_cached_attn",
            "--cache_maxframes", "1",
            "--use_tome_cache",
            "--do_add_noise", 
            "--guidance_scale", "1.0" ,
            "--cache_interval", str(args.cache_interval),
            "--use_random_cache_interval", args.random_cache_interval,
            "--use_attn_concat", args.use_attn_concat,
            "--use_feature_injection", args.use_feature_injection,
            "--feature_similarity_threshold", str(args.feature_similarity_threshold),
            "--feature_injection_strength", str(args.feature_injection_strength),
            "--ttt_lr", str(args.ttt_lr),
            "--use_ttt_cache", args.use_ttt_cache,
            "--reverse_tag", args.reverse_tag,
        ]
    else:
        command = [
            'python', "main.py",
            "--cuda_visible_devices", args.cuda_visible_devices,
            "--input", f"{file_path}/{src_vid_name}.mp4",
            "--prompt", prompt,
            "--output_dir", args.output_dir,
            "--model_id", model_id,
            "--diffusion_steps", diffusion_steps,
            "--noise_strength", noise_strength,
            "--acceleration", "xformers",
            "--use_cached_attn",
            "--use_feature_injection",
            "--cache_maxframes", "1",
            "--use_tome_cache",
            "--do_add_noise", 
            "--cache_interval", str(args.cache_interval),
            "--guidance_scale", "1.0" ,
        ]
    subprocess.run(command)
