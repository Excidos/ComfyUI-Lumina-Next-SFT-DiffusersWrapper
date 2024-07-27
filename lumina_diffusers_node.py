import torch
import numpy as np
from diffusers import LuminaText2ImgPipeline, FlowMatchEulerDiscreteScheduler
import comfy.model_management as mm
import os
import traceback

class LuminaDiffusersNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Alpha-VLLM/Lumina-Next-SFT-diffusers"}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0}),
                "seed": ("INT", {"default": -1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "scaling_watershed": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "proportional_attn": ("BOOLEAN", {"default": True}),
                "clean_caption": ("BOOLEAN", {"default": True}),
                "max_sequence_length": ("INT", {"default": 256, "min": 64, "max": 512}),
                "use_time_shift": ("BOOLEAN", {"default": False}),
                "t_shift": ("INT", {"default": 4, "min": 1, "max": 20}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "latents": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "LuminaWrapper"

    def __init__(self):
        self.pipe = None
        self.vae_scale_factor = 0.13025  # SDXL scaling factor

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

    def generate(self, model_path, prompt, negative_prompt, num_inference_steps, guidance_scale, seed,
                 batch_size, scaling_watershed, proportional_attn, clean_caption, max_sequence_length, 
                 use_time_shift, t_shift, strength, latents=None):
        try:
            if self.pipe is None:
                self.load_model(model_path)

            device = mm.get_torch_device()
            if seed == -1:
                seed = int.from_bytes(os.urandom(4), "big")
            generator = torch.Generator(device=device).manual_seed(seed)

            if latents is not None:
                latent_height, latent_width = latents['samples'].shape[2:]
                height, width = latent_height * 8, latent_width * 8
            else:
                height = width = 1024
                latent_height, latent_width = height // 8, width // 8

            if proportional_attn:
                self.pipe.cross_attention_kwargs = {"base_sequence_length": (height * width) // 256}
            else:
                self.pipe.cross_attention_kwargs = None

            if use_time_shift:
                time_shift_factor = 1 + t_shift
                self.pipe.scheduler.config.shift = time_shift_factor
                print(f"Time shift factor: {time_shift_factor}")
            else:
                print("Time shift disabled")

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
                "output_type": "latent",
                "clean_caption": clean_caption,
                "max_sequence_length": max_sequence_length,
                "scaling_watershed": scaling_watershed,
            }

            if latents is not None:
                print("Input latents provided:")
                print(f"Latents shape: {latents['samples'].shape}")
                print(f"Latents dtype: {latents['samples'].dtype}")
                print(f"Latents min: {latents['samples'].min()}, max: {latents['samples'].max()}")
                
                # Scale input latents
                scaled_latents = latents['samples'] * self.vae_scale_factor
                pipe_args["latents"] = scaled_latents.to(device=device, dtype=self.pipe.transformer.dtype)

                # Apply strength
                if strength < 1.0:
                    noise = torch.randn_like(pipe_args["latents"])
                    pipe_args["latents"] = self.pipe.scheduler.scale_noise(pipe_args["latents"], num_inference_steps, noise)
                    pipe_args["latents"] = strength * pipe_args["latents"] + (1 - strength) * noise

                # Check if latents are all zeros and add noise if necessary
                if pipe_args["latents"].min() == pipe_args["latents"].max() == 0:
                    print("Warning: Input latents are all zeros. Adding noise.")
                    noise = torch.randn_like(pipe_args["latents"])
                    pipe_args["latents"] = noise * 0.99 + pipe_args["latents"] * 0.01

            # Generate latents
            output = self.pipe(**pipe_args)

            # Process output latents
            latents = output.images

            # Ensure no NaN values in the output
            latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)

            # Scale output latents
            latents = latents / self.vae_scale_factor

            print(f"Generated latents shape: {latents.shape}")
            print(f"Generated latents dtype: {latents.dtype}")
            print(f"Generated latents min: {latents.min()}, max: {latents.max()}")

            return ({"samples": latents.to(device)},)

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            traceback.print_exc()
            return ({"samples": torch.zeros((batch_size, 4, latent_height, latent_width), dtype=torch.float32, device=device)},)

NODE_CLASS_MAPPINGS = {
    "LuminaDiffusersNode": LuminaDiffusersNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaDiffusersNode": "Lumina-Next-SFT Diffusers"
}
