import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.models.attention_processor import XFormersAttnProcessor, AttnProcessor2_0
from PIL import Image
import logging
import random

from src.streamv2v import StreamV2V
from src.streamv2v.image_utils import postprocess_image
from src.streamv2v.models.attention_processor import CachedSTXFormersAttnProcessor, CachedSTAttnProcessor2_0
from src.streamv2v.models.confidence_gate_attention import ConfidenceGateCachedSTXFormersAttnProcessor
from src.streamv2v.models.similarity_gate_attention import SimilarityGateCachedSTXFormersAttnProcessor


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamV2VWrapper:
    def __init__(
        self,
        model_id_or_path: str,  
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        mode: Literal["img2img", "txt2img"] = "img2img",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10, 
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "none",
        use_cached_attn: bool = True,
        cached_attn_style: Literal["origin", "confidence", "similarity"] = "similarity",
        use_feature_injection: bool = True,
        feature_injection_strength: float = 0.8,
        feature_similarity_threshold: float = 0.98,
        cache_interval: int = 1,
        cache_maxframes: int = 1, 
        use_tome_cache: bool = True,
        use_random_cache_interval: bool = False,
        tome_metric: str = "keys",
        tome_ratio: float = 0.5,
        use_grid: bool = False,
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        save_attn_map: bool = False,
        reverse_tag: bool = True,
        vis: bool = False,
        use_attn_concat: bool = True,
        ttt_lr: float = 1.0,
    ):

        self.sd_turbo = "turbo" in model_id_or_path
        self.sd_xl = "xl" in model_id_or_path
        
        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )
        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "vid2vid mode must use denoising batch for now."
                )
        self.mode = mode

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_cached_attn = use_cached_attn
        self.use_feature_injection = use_feature_injection
        self.feature_injection_strength = feature_injection_strength
        self.feature_similarity_threshold = feature_similarity_threshold
        self.cache_interval = cache_interval
        self.cache_maxframes = cache_maxframes
        self.use_tome_cache = use_tome_cache
        self.tome_metric = tome_metric
        self.tome_ratio = tome_ratio
        self.use_grid = use_grid
        self.use_safety_checker = use_safety_checker
        self.use_random_cache_interval = use_random_cache_interval
        self.save_attn_map = save_attn_map
        self.reverse_tag = reverse_tag
        self.ttt_lr = ttt_lr
        self.vis = vis
        self.use_attn_concat = use_attn_concat



        self.stream: StreamV2V = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            cached_attn_style=cached_attn_style,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        # 数据并行
        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.0.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
       Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.
        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.
        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):

            image = self.preprocess_image(image)

        image_tensor = self.stream(image)

        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        warmup: int = 10,
        cached_attn_style: Literal["origin", "confidence", "similarity"] = "similarity",
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ) -> StreamV2V:

        # Choose the pipeline based on the flag
        pipeline_cls = StableDiffusionXLPipeline if self.sd_xl else StableDiffusionPipeline

        try:
            # Attempt to load the model from a local directory
            pipe = pipeline_cls.from_pretrained(model_id_or_path).to(device=self.device, dtype=self.dtype)
        except ValueError:
            # If the model is not found locally, load from Hugging Face
            try:
                pipe = pipeline_cls.from_single_file(model_id_or_path).to(device=self.device, dtype=self.dtype)
            except Exception as e:
                logging.error(f"Failed to load model from Hugging Face: {e}")
                sys.exit("Model load has failed from both local and Hugging Face sources.")
        except Exception as e:
            # Handle unexpected errors
            logging.error(f"Unexpected error occurred: {e}")
            traceback.print_exc()
            sys.exit("Model load has failed due to an unexpected error.")

        if self.sd_xl:
            # Avoid error if "text_embeds" not in added_cond_kwargs: TypeError: argument of type 'NoneType' is not iterable
            # https://github.com/huggingface/diffusers/issues/4649
            pipe.unet.config.addition_embed_type = None
            
        stream = StreamV2V(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        if not self.sd_turbo:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id,
                        adapter_name="lcm")
                else:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict="/home/zrj/project/ori_v2v/streamv2v/data/checkpoints/lcm-lora-sdv1-5",
                        adapter_name="lcm"
                        )

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    
        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                    device=pipe.device, dtype=pipe.dtype
                )
        
        if self.use_random_cache_interval:
            cache_interval_list = []

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
                # original StreamV2V
                if self.use_cached_attn and cached_attn_style == "origin":
                    attn_processors = stream.pipe.unet.attn_processors
                    new_attn_processors = {}
                    for key, attn_processor in attn_processors.items():
                        assert isinstance(attn_processor, XFormersAttnProcessor), \
                              "We only replace 'XFormersAttnProcessor' to 'CachedSTXFormersAttnProcessor'"

                        if self.use_random_cache_interval:
                            self.cache_interval = random.randint(1, 8)
                            cache_interval_list.append(self.cache_interval)
                        
                        new_attn_processors[key] = CachedSTXFormersAttnProcessor(name=key,
                                                                                 use_feature_injection=self.use_feature_injection,
                                                                                 feature_injection_strength=self.feature_injection_strength,
                                                                                 feature_similarity_threshold=self.feature_similarity_threshold,
                                                                                 interval=self.cache_interval, 
                                                                                 max_frames=self.cache_maxframes,
                                                                                 use_tome_cache=self.use_tome_cache,
                                                                                 tome_metric=self.tome_metric,
                                                                                 tome_ratio=self.tome_ratio,
                                                                                 use_grid=self.use_grid,
                                                                                 save_attn_map=self.save_attn_map)
                    stream.pipe.unet.set_attn_processor(new_attn_processors)
                # SimilarityGate cached attention
                if self.use_cached_attn and cached_attn_style == "similarity":
                    # Initialize SimilarityGate cached attention
                    print("use_cached_attn:", self.use_cached_attn)
                    print("cached_attn_style:", cached_attn_style)
                    attn_processors = stream.pipe.unet.attn_processors
                    new_attn_processors = {}
                    for key, attn_processor in attn_processors.items():
                        assert isinstance(attn_processor, XFormersAttnProcessor), \
                              "We only replace 'XFormersAttnProcessor' to 'SimilarityGateCachedSTXFormersAttnProcessor'"
                        new_attn_processors[key] = SimilarityGateCachedSTXFormersAttnProcessor(name=key,
                                                                                 use_feature_injection=self.use_feature_injection,
                                                                                 feature_similarity_threshold=self.feature_similarity_threshold,
                                                                                 interval=self.cache_interval, 
                                                                                 save_attn_map=self.save_attn_map,
                                                                                 reverse_tag=self.reverse_tag,
                                                                                 vis=self.vis,
                                                                                 use_concat=self.use_attn_concat,
                                                                                 ttt_lr=self.ttt_lr,
                                                                                 )
                    stream.pipe.unet.set_attn_processor(new_attn_processors)
                # ConfidenceGate cached attention
                if self.use_cached_attn and cached_attn_style == "confidence":
                    # Initialize ConfidenceGate cached attention
                    print("use_cached_attn:", self.use_cached_attn)
                    print("cached_attn_style:", cached_attn_style)
                    attn_processors = stream.pipe.unet.attn_processors
                    new_attn_processors = {}
                    for key, attn_processor in attn_processors.items():
                        assert isinstance(attn_processor, XFormersAttnProcessor), \
                              "We only replace 'XFormersAttnProcessor' to 'ConfidenceGateCachedSTXFormersAttnProcessor'"
                        new_attn_processors[key] = ConfidenceGateCachedSTXFormersAttnProcessor(name=key,
                                                                                 use_feature_injection=self.use_feature_injection,
                                                                                 feature_similarity_threshold=self.feature_similarity_threshold,
                                                                                 interval=self.cache_interval, 
                                                                                 save_attn_map=self.save_attn_map,
                                                                                 vis=self.vis,
                                                                                 use_concat=self.use_attn_concat,
                                                                                 ttt_lr=self.ttt_lr,
                                                                                 )
                    stream.pipe.unet.set_attn_processor(new_attn_processors)
            if acceleration == "tensorrt":
                if self.use_cached_attn:
                    raise NotImplementedError("TensorRT seems not support the costom attention_processor")
                else:
                    stream.pipe.enable_xformers_memory_efficient_attention()
                    if self.use_cached_attn:
                        attn_processors = stream.pipe.unet.attn_processors
                        new_attn_processors = {}
                        for key, attn_processor in attn_processors.items():
                            assert isinstance(attn_processor, XFormersAttnProcessor), \
                                "We only replace 'XFormersAttnProcessor' to 'CachedSTXFormersAttnProcessor'"
                            new_attn_processors[key] = CachedSTXFormersAttnProcessor(name=key,
                                                                                    use_feature_injection=self.use_feature_injection,
                                                                                    feature_injection_strength=self.feature_injection_strength,
                                                                                    feature_similarity_threshold=self.feature_similarity_threshold,
                                                                                    interval=self.cache_interval, 
                                                                                    max_frames=self.cache_maxframes,
                                                                                    use_tome_cache=self.use_tome_cache,
                                                                                    tome_metric=self.tome_metric,
                                                                                    tome_ratio=self.tome_ratio,
                                                                                    use_grid=self.use_grid)
                        stream.pipe.unet.set_attn_processor(new_attn_processors)

                from polygraphy import cuda
                from streamv2v.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamv2v.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamv2v.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--cache--{self.use_cached_attn}--mode-{self.mode}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--cache--{self.use_cached_attn}--mode-{self.mode}"

                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.frame_bff_size,
                        min_batch_size=stream.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.frame_bff_size,
                        min_batch_size=stream.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                    )
                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=stream.frame_bff_size,
                        min_batch_size=stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=stream.frame_bff_size,
                        min_batch_size=stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=stream.frame_bff_size,
                    )

                cuda_steram = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_steram, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_steram,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                if self.use_cached_attn:
                    raise NotImplementedError
                from streamv2v.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if self.use_random_cache_interval:
            # 输出缓存间隔的列表
            print(f"缓存间隔列表/Cache interval list: {cache_interval_list}")

        if seed < 0: # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            # TODO:去噪步数的消融研究
            num_inference_steps=50,
            # TODO:guidance_scale的消融研究，是否增加prompt遵循能力
            guidance_scale=1.2 if stream.cfg_type in ["full", "self", "initialize"] else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
