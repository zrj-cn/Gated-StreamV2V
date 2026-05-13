from PIL import Image
import json 
import os 
import argparse
import cv2
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_visible_devices', type=str, default='7', help='CUDA visible devices.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--method_version', type=str, default='default', help='The name of the method view.')
    parser.add_argument('--set_file_path', type=str, default='user_study_upload/eval_motion.json', help='Path to your JSON file.')
    return parser.parse_args()

# Function to load JSON data from a file
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Function to create a dictionary with vid_name as keys and prompt as values
def create_vid_prompt_dict(json_data):
    vid_prompt_dict = {}
    for item in json_data:
        vid_name = item.get('vid_name', '')
        prompt = item.get('prompt', '')
        vid_prompt_dict[vid_name] = prompt
    return vid_prompt_dict

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from transformers import CLIPProcessor, CLIPModel

    device = args.device
    method_version = args.method_version
    set_file_path = args.set_file_path
    file_path = set_file_path


    with open(file_path, 'r') as file:
        json_data = json.load(file)
    video_maps = create_vid_prompt_dict(json_data)

    edit_video_dir = f"/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/{method_version}" 
    video_names = list(video_maps.keys())

    # You may need to download the model in advance or use hf-mirror
    # Replace the model path with your own
    model = CLIPModel.from_pretrained("/home/zrj/project/ori_v2v/streamv2v/data/checkpoints/clip-vit-base-patch32")
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("/home/zrj/project/ori_v2v/streamv2v/data/checkpoints/clip-vit-base-patch32")

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    consistency_score = []
    prompt_score = []


    out_json = {}

    for v in video_names:
        try:
            out_json[v] = {}
            prompt = video_maps[v]
            video_path = os.path.join(edit_video_dir, f"{v}.mp4")
            video_embs = []

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            # Check if video opened successfully
            if not cap.isOpened():
                print("Error opening video file")
            # Process video frames
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert the BGR frame captured by cv2 to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert the numpy array frame to PIL Image
                    image = Image.fromarray(frame_rgb)
                    
                    # Your existing processing code
                    with torch.no_grad():
                        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = model(**inputs)

                        image_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                    video_embs.append(image_embeds)
                else:
                    # Break the loop if no frames are returned (end of video)
                    break

            # Release the video capture object
            cap.release()
            video_embs = torch.cat(video_embs, dim=0)   # (T, 768)
            # 
            text_score = cos(text_embeds, video_embs)    # (1, T)
            text_score = text_score.mean().cpu().item()
            prompt_score.append(text_score)

            # two continue frames cos similarity
            emb1 = video_embs[:-1]  # (N, 768)
            emb2 = video_embs[1:]   # (N, 768)
            score = cos(emb1, emb2) # (N,)
            score = score.mean().cpu().item()

            consistency_score.append(score)
            out_json[v][prompt] = score
            print(v, prompt, score)
        except:
            print(f'{v} does not exist!')

    print("Number of videos ", len(prompt_score))
    print("Avg consistency score ", sum(consistency_score) / len(consistency_score))
    # print("Avg prompt score ", sum(prompt_score) / len(prompt_score))

    json.dump(out_json, open(f"./clip_score_log/{method_version}.clipscore", "w"), sort_keys=True, indent=4)
    # Also write the average score into the JSON file
    out_json['avg_score'] = sum(consistency_score) / len(consistency_score)
    json.dump(out_json, open(f"./clip_score_log/{method_version}.clipscore", "w"), sort_keys=True, indent=4)
