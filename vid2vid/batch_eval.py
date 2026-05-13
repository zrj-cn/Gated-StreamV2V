import os
import json
import argparse
import subprocess
 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a JSON file contains multiple edits.")
    parser.add_argument('--cuda_visible_devices', type=str, default="7", help='CUDA visible devices.')
    parser.add_argument('--json_file', type=str, help='The path to the JSON file to process.')
    parser.add_argument('--output_dir', type=str, default="/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/default", help='The directory to save the output videos.')
    parser.add_argument('--cache_interval', type=int, default=1, help='Cache interval for processing.')
    parser.add_argument('--noise_strength', type=str, default=None, help='Noise strength to use for all videos. Overrides the value in the json file.')
    parser.add_argument('--use_cached_attn', type=str, default="True", help='Whether to use cached attention.')
    parser.add_argument('--cached_attn_style', type=str, default="similarity", help='Cached attention style.')
    parser.add_argument('--reverse_tag', type=str, default="True", help='Whether to use reverse tag.')
    parser.add_argument('--use_attn_concat', type=str, default="False", help='Whether to use attn concat.')
    parser.add_argument('--use_feature_injection', type=str, default="True", help='Whether to use feature injection.')
    parser.add_argument('--feature_similarity_threshold', type=float, default=0.98, help='Feature similarity threshold.')
    parser.add_argument('--feature_injection_strength', type=float, default=0.5, help='Feature injection strength.')
    parser.add_argument('--ttt_lr', type=float, default=1.0, help='Used to scale the beta of the gated attention layer.')

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
            print(f"video already exists, jump: {out_put_video}")
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
            "--use_attn_concat", args.use_attn_concat,
            "--use_feature_injection", args.use_feature_injection,
            "--feature_similarity_threshold", str(args.feature_similarity_threshold),
            "--feature_injection_strength", str(args.feature_injection_strength),
            "--ttt_lr", str(args.ttt_lr),
            "--cached_attn_style", args.cached_attn_style,
            "--reverse_tag", args.reverse_tag,
        ]
    else:
            # Error: vid_name not found in JSON; please use ori_batch_eval.py
            print(f"vid_name not found in JSON, please use ori_batch_eval.py: {item}")
            continue
    subprocess.run(command)
