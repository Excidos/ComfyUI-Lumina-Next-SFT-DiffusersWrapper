import torch
from diffusers import LuminaText2ImgPipeline, FlowMatchEulerDiscreteScheduler
import comfy.model_management as mm
import os
import numpy as np
import traceback
import math

class LuminaDiffusersNode:
    @classmethod
    def INPUT_TYPES(s):
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
                "time_aware_scaling": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0}),
                "context_drop_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5}),
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

            print(f"Attempting to load Lumina model from: {model_path}")
            print(f"Device: {device}, Dtype: {dtype}")

            full_path = os.path.join(os.path.dirname(__file__), model_path)
            if not os.path.exists(full_path):
                raise ValueError(f"Model path does not exist: {full_path}")

            print(f"Loading Lumina model from: {full_path}")
            self.pipe = LuminaText2ImgPipeline.from_pretrained(full_path, torch_dtype=dtype)
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.to(device)
            print("Pipeline successfully loaded and moved to device.")
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
            traceback.print_exc()

    def apply_time_aware_scaling(self, transformer, scale_factor):
        if hasattr(transformer, 'text_encoder'):
            transformer.text_encoder.config.time_aware_scaling = scale_factor
        if hasattr(transformer, 'unet'):
            transformer.unet.config.time_aware_scaling = scale_factor

    def apply_context_drop(self, transformer, drop_ratio):
        if hasattr(transformer, 'text_encoder'):
            transformer.text_encoder.config.context_drop_ratio = drop_ratio
        if hasattr(transformer, 'unet'):
            transformer.unet.config.context_drop_ratio = drop_ratio

    def generate(self, model_path, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, seed, batch_size, scaling_watershed, time_aware_scaling, context_drop_ratio):
        try:
            if self.pipe is None:
                print("Pipeline not loaded. Attempting to load model.")
                self.load_model(model_path)

            if self.pipe is None:
                raise ValueError("Failed to load the pipeline.")

            device = mm.get_torch_device()
            print(f"Generation device: {device}")

            if seed == -1:
                seed = int.from_bytes(os.urandom(4), "big")
            generator = torch.Generator(device=device).manual_seed(seed)

            # Prepare Lumina-specific kwargs
            scale_factor = math.sqrt(width * height / 1024**2)
            
            # Modify the pipe's transformer to include Lumina-specific features
            if hasattr(self.pipe, 'transformer'):
                self.pipe.transformer.scale_factor = scale_factor
                self.pipe.transformer.scale_watershed = scaling_watershed
                self.apply_time_aware_scaling(self.pipe.transformer, time_aware_scaling)
                self.apply_context_drop(self.pipe.transformer, context_drop_ratio)

            print(f"Starting generation with seed: {seed}")
            output = self.pipe(
                prompt=[prompt] * batch_size,
                negative_prompt=[negative_prompt] * batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=1,
                output_type="pt",
            )

            print(f"Raw output shape: {output.images.shape}")
            print(f"Raw output min: {output.images.min()}, max: {output.images.max()}")

            images = output.images
            images = images.permute(0, 2, 3, 1).cpu()
            
            print(f"Permuted images shape: {images.shape}")
            print(f"Images min: {images.min()}, max: {images.max()}")

            # Apply normalization
            images = (images + 1) / 2  # Assuming the output is in the range [-1, 1]
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            
            print(f"Final images shape: {images.shape}")
            print(f"Final images min: {images.min()}, max: {images.max()}")

            # Generate latents
            with torch.no_grad():
                latents = self.pipe.vae.encode(output.images.to(self.pipe.vae.dtype)).latent_dist.sample()
                latents = latents * self.pipe.vae.config.scaling_factor
            
            print(f"Latents shape: {latents.shape}")
            print(f"Latents min: {latents.min()}, max: {latents.max()}")

            latents_for_comfy = {"samples": latents.cpu()}

            return (images, latents_for_comfy)

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            traceback.print_exc()
            return (torch.zeros((batch_size, height, width, 3), dtype=torch.uint8), 
                    {"samples": torch.zeros((batch_size, 4, height // 8, width // 8), dtype=torch.float32)})

NODE_CLASS_MAPPINGS = {
    "LuminaDiffusersNode": LuminaDiffusersNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaDiffusersNode": "Lumina-Next-SFT Diffusers"
}
