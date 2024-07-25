import torch
import numpy as np
from diffusers import LuminaText2ImgPipeline, FlowMatchEulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer
import comfy.model_management as mm
import os
import traceback
import math
from .utils import get_2d_rotary_pos_embed_lumina, REGIONAL_PROMPT
import inspect

class LuminaDiffusersNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Lumina-Next-SFT-diffusers"}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": -1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "scaling_watershed": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "proportional_attn": ("BOOLEAN", {"default": True}),
                "clean_caption": ("BOOLEAN", {"default": True}),
                "max_sequence_length": ("INT", {"default": 256, "min": 64, "max": 512}),
            },
            "optional": {
                "regional_prompts": ("REGIONAL_PROMPTS",),
                "mask": ("MASK",),
                "latents": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "LuminaWrapper"

    def __init__(self):
        self.pipe = None

    def load_model(self, model_path):
        try:
            device = mm.get_torch_device()
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

            print(f"Loading Lumina model from: {model_path}")
            print(f"Device: {device}, Dtype: {dtype}")

            full_path = os.path.join(os.path.dirname(__file__), model_path)
            if not os.path.exists(full_path):
                print(f"Model not found. Downloading Lumina model to: {full_path}")
                self.pipe = LuminaText2ImgPipeline.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers", torch_dtype=dtype)
                self.pipe.save_pretrained(full_path)
            else:
                self.pipe = LuminaText2ImgPipeline.from_pretrained(full_path, torch_dtype=dtype)

            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.to(device)
            print("Pipeline successfully loaded and moved to device.")
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
            traceback.print_exc()

    def generate(self, model_path, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, seed,
                batch_size, scaling_watershed, proportional_attn, clean_caption, max_sequence_length):
        try:
            if self.pipe is None:
                self.load_model(model_path)

            device = mm.get_torch_device()
            if seed == -1:
                seed = int.from_bytes(os.urandom(4), "big")
            generator = torch.Generator(device=device).manual_seed(seed)

            default_image_size = self.pipe.transformer.config.sample_size * self.pipe.vae_scale_factor

            if proportional_attn:
                self.pipe.cross_attention_kwargs = {"base_sequence_length": (default_image_size // 16) ** 2}
            else:
                self.pipe.cross_attention_kwargs = None

            print(f"Starting generation with seed: {seed}")

            pipe_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": batch_size,
                "generator": generator,
                "output_type": "pt",
                "clean_caption": clean_caption,
                "max_sequence_length": max_sequence_length,
                "scaling_watershed": scaling_watershed,
            }

            # Generate images
            output = self.pipe(**pipe_args)

            processed_images = self.process_output(output.images)
            latents_dict = self.process_latents(output.images)

            return (processed_images, latents_dict)

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            traceback.print_exc()
            return (torch.zeros((batch_size, height, width, 3), dtype=torch.float32),
                    {"samples": torch.zeros((batch_size, 4, height // 8, width // 8), dtype=torch.float32)})

    def process_output(self, images):
        print(f"Raw output images shape: {images.shape}")
        print(f"Raw output images dtype: {images.dtype}")
        print(f"Raw output images min: {images.min()}, max: {images.max()}")

        images = images.float()
        images = images.clamp(0, 1)
        images = images.permute(0, 2, 3, 1)

        print(f"Processed images shape: {images.shape}")
        print(f"Processed images dtype: {images.dtype}")
        print(f"Processed images min: {images.min()}, max: {images.max()}")

        return images

    def process_latents(self, images):
        latents = images.clone().detach()
        latents = latents.to(dtype=torch.float32)
        
        print(f"Latents shape before processing: {latents.shape}")
        print(f"Latents dtype: {latents.dtype}")
        print(f"Latents min: {latents.min()}, max: {latents.max()}")

        return {"samples": latents}

NODE_CLASS_MAPPINGS = {
    "LuminaDiffusersNode": LuminaDiffusersNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaDiffusersNode": "Lumina-Next-SFT Diffusers"
}
