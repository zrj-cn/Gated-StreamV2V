import os
import sys
import time
from typing import Literal, Dict, Optional
import cv2
import fire

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))




def main(
    input: str,
    prompt: str,
    # TODO -ZRJ- You can choose the cuda_visible_devices.
    cuda_visible_devices: str = "7",
    video_name: str = None,
    output_dir: str = os.path.join(CURRENT_DIR, "tests/vid"),
    # TODO -ZRJ- If use batch_eval: The model_id in the JSON file needs to be modified.
    model_id: str = "/home/zrj/project/ori_v2v/streamv2v/data/checkpoints/stable-diffusion-1.5",
    scale: float = 1.0,
    guidance_scale: float = 1.0,
    diffusion_steps: int = 4,
    noise_strength: float = 0.4,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_cached_attn: bool = True,
    cached_attn_style: Literal["origin", "confidence", "similarity"] = "similarity",
    use_denoising_batch: bool = True,
    use_feature_injection: bool = True,
    feature_injection_strength: float = 0.8,
    feature_similarity_threshold: float = 0.98,
    cache_interval: int = 1,
    cache_maxframes: int = 1,
    use_tome_cache: bool = True,
    do_add_noise: bool = True,
    enable_similar_image_filter: bool = False,
    use_random_cache_interval: bool = False,
    save_attn_map: bool = False,
    seed: int = 2,
    reverse_tag: bool = True,
    vis: bool = False,
    use_attn_concat: bool = True,
    ttt_lr: float = 1.0,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if vis:
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    import torch
    from torchvision.io import read_video, write_video
    from tqdm import tqdm
    from utils.wrapper import StreamV2VWrapper

    def write_video_opencv(output_path, frames, fps=30):
        """
        Write video with OpenCV (replacing torchvision's write_video)
        frames: list of video frames, shape (num_frames, height, width, 3), dtype np.uint8 (0-255)
        """
        if len(frames) == 0:
            raise ValueError("No video frames to write")
        
            # Get video frame height and width (ensure HWC format)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()  
                if frame.dtype in (np.float32, np.float64):
                    frame = (frame * 255).astype(np.uint8)
            
            if frame.shape[-1] == 3:  # Ensure it is a 3-channel color image
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            out.write(frame)
        
        out.release()
        print(f"Video saved to: {output_path}")
    """
    Perform video-to-video translation with StreamV2V.

    Parameters
    ----------
    input: str
        The input video name.
    prompt: str
        The editting prompt to perform video translation.
    output_dir: str, optional
        The directory of the output video.
    model_id: str, optional
        The base image diffusion model. 
        By default, it is SD 1.5 ("/home/zrj/project/ori_v2v/streamv2v/data/checkpoints/stable-diffusion-1.5").
    scale: float, optional
        The scale of the resolution, by default 1.0.
    guidance_scale: float, optional
        Classifier-free guidance (CFG).
        By default, it is not enabled, 1.0.
    diffusion_steps: int, optional
        Diffusion steps to perform. Higher steps ususally lead to higher quality but slower speed.
        By default, it is 4.
    noise_strength: float, optional
        Our editing method is SDEdit. Higher the noise_strength means more noise is added to the starting frames.
        Highter strength ususally leads to better edit effects but may sacrifice the consistency.
        [0,1]
        By default, it is 0.4.
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers"
        The type of acceleration to use for video translation. 
        By default, it is xformers.
    cached_attn_style: Literal["origin", "confidence", "similarity"], optional
        The style of cached attention.
        origin: Use the original StreamV2V.
        confidence: Use the alignment confidence to gate the feature bank.
        similarity: Use the similarity score to gate the feature bank.
        By default, it is similarity.
    use_denoising_batch: bool, optional
        Whether to use denoising batch or not.
        By default, it is True.
    use_cached_attn: bool, optional
        Whether to cache the self attention maps of the pervious frames to imporve temporal consistency.
        If it is set to False, it would roll back to per-frame StreamDiffusion.
        By default, it is True
    use_feature_injection: bool, optional
        Whether directly to inject the features of the pervious frames to imporve temporal consistency.
        By default, it is True
    feature_injection_strength: float, optional
        The strength to perform feature injection. Higher value means higher weights from previous frames.
        By default, it is 0.8
    feature_similarity_threshold: float, optional
        The threshold to identify the similar features.
        By default, it is 0.98
    cache_interval: int, optional
        The frame interval to update the feature bank.
        By default, it is 1
    cache_maxframes: int, optional
        The max frames to cache in the feature bank. Use FIFO (First-In-First-Out) strategy to update. 
        Only effective when use_tome_cache = False, otherwise, cache_maxframes is set to 1.
    use_tome_cache : bool, optional
        Use Token Merging (ToMe) to update the bank.
        By default, it is True.
    enable_similar_image_filter: bool, optional
        Whether to enable similar image filter or not,
        By default, it is False.
    ttt_lr: float, optional
        Used to scale the beta of the gated attention layer.
        By default, it is 1.0.
    seed: int, optional
        The seed, by default 2. if -1, use random seed.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if use_random_cache_interval:
        print("Use random cache interval")
    else:
        print("Do not use random cache interval")

    video_info = read_video(input)
    video = video_info[0] / 255
    fps = video_info[2]["video_fps"]
    height = int(video.shape[1] * scale)
    width = int(video.shape[2] * scale)

    # Initial denoising step
    init_step = int(50 * (1 - noise_strength))
    # Denoising step interval for each diffusion_steps
    interval = int(50 * noise_strength) // diffusion_steps
    # List of denoising time indices
    t_index_list = [init_step + i * interval for i in range(diffusion_steps)]

    print(f"cache_interval: {cache_interval}")

    stream = StreamV2VWrapper(
        model_id_or_path=model_id,
        mode="img2img",
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        use_cached_attn=use_cached_attn,
        cached_attn_style=cached_attn_style,
        use_feature_injection=use_feature_injection,
        feature_injection_strength=feature_injection_strength,
        feature_similarity_threshold=feature_similarity_threshold,
        cache_interval=cache_interval,
        cache_maxframes=cache_maxframes,
        use_tome_cache=use_tome_cache,
        use_random_cache_interval=use_random_cache_interval,
        save_attn_map=save_attn_map,
        seed=seed,
        reverse_tag=reverse_tag,
        vis=vis,
        use_attn_concat=use_attn_concat,
        ttt_lr=ttt_lr,
    )
    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
    )

    # Specify LORAs
    if any(word in prompt for word in ['pixelart', 'pixel art', 'Pixel art', 'PixArFK']):
        stream.stream.load_lora("./lora_weights/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors", adapter_name='pixelart')
        stream.stream.pipe.set_adapters(["lcm", "pixelart"], adapter_weights=[1.0, 1.0])
        print("Use LORA: pixelart in ./lora_weights/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors")
    elif any(word in prompt for word in ['lowpoly', 'low poly', 'Low poly']):
        stream.stream.load_lora("./lora_weights/low_poly.safetensors", adapter_name='lowpoly')
        stream.stream.pipe.set_adapters(["lcm", "lowpoly"], adapter_weights=[1.0, 1.0])
        print("Use LORA: lowpoly in ./lora_weights/low_poly.safetensors")
    elif any(word in prompt for word in ['Claymation', 'claymation']):
        stream.stream.load_lora("./lora_weights/Claymation.safetensors", adapter_name='claymation')
        stream.stream.pipe.set_adapters(["lcm", "claymation"], adapter_weights=[1.0, 1.0])
        print("Use LORA: claymation in ./lora_weights/Claymation.safetensors")
    elif any(word in prompt for word in ['crayons', 'Crayons', 'crayons doodle', 'Crayons doodle']):
        stream.stream.load_lora("./lora_weights/doodle.safetensors", adapter_name='crayons')
        stream.stream.pipe.set_adapters(["lcm", "crayons"], adapter_weights=[1.0, 1.0])
        print("Use LORA: crayons in ./lora_weights/doodle.safetensors")
    elif any(word in prompt for word in ['sketch', 'Sketch', 'pencil drawing', 'Pencil drawing']):
        stream.stream.load_lora("./lora_weights/Sketch_offcolor.safetensors", adapter_name='sketch')
        stream.stream.pipe.set_adapters(["lcm", "sketch"], adapter_weights=[1.0, 1.0])
        print("Use LORA: sketch in ./lora_weights/Sketch_offcolor.safetensors")
    elif any(word in prompt for word in ['oil painting', 'Oil painting']):
        stream.stream.load_lora("./lora_weights/bichu-v0612.safetensors", adapter_name='oilpainting')
        stream.stream.pipe.set_adapters(["lcm", "oilpainting"], adapter_weights=[1.0, 1.0])
        print("Use LORA: oilpainting in ./lora_weights/bichu-v0612.safetensors")

    video_result = torch.zeros(video.shape[0], height, width, 3)

    for _ in range(stream.batch_size):
        stream(image=video[0].permute(2, 0, 1))

    inference_time = []
    for i in tqdm(range(video.shape[0])):
        iteration_start_time = time.time()
        output_image = stream(video[i].permute(2, 0, 1))
        video_result[i] = output_image.permute(1, 2, 0)
        iteration_end_time = time.time()
        inference_time.append(iteration_end_time -iteration_start_time )
    print(f'Avg time: {sum(inference_time[20:])/len(inference_time[20:])}')

    video_result = video_result * 255
    prompt_txt = prompt.replace(' ', '-')
    input_vid = input.split('/')[-1]
    if video_name == None:
        output = os.path.join(output_dir, f"{input_vid.rsplit('.', 1)[0]}_{prompt_txt}.{input_vid.rsplit('.', 1)[1]}")

    else:
        output = os.path.join(output_dir, f"{video_name}.mp4")
    write_video_opencv(output, [frame.numpy().astype("uint8") for frame in video_result], fps=float(fps))


if __name__ == "__main__":
    fire.Fire(main)
