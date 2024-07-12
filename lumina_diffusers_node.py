import torch
from diffusers import LuminaText2ImgPipeline
import comfy.model_management as mm
import os
import numpy as np

class LuminaDiffusersNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.1, "max": 20.0}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": -1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "LuminaWrapper"

    def __init__(self):
        self.pipe = None

    def generate(self, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, seed, batch_size):
        device = mm.get_torch_device()
        
        if self.pipe is None:
            model_path = os.path.join(os.path.dirname(__file__), "Lumina-Next-SFT-diffusers")
            if not os.path.exists(model_path):
                print(f"Downloading Lumina model to: {model_path}")
                self.pipe = LuminaText2ImgPipeline.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers", torch_dtype=torch.float16)
                self.pipe.save_pretrained(model_path)
            else:
                print(f"Loading Lumina model from: {model_path}")
                self.pipe = LuminaText2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            
            self.pipe.to(device)

        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator(device=device).manual_seed(seed)

        try:
            # Generate images
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

            # Debug: Print raw output
            print(f"Raw output shape: {output.images.shape}")
            print(f"Raw output min: {output.images.min()}, max: {output.images.max()}")

            # Extract images and convert to the format ComfyUI expects
            images = output.images
            images = images.permute(0, 2, 3, 1).cpu()
            
            # Debug: Print intermediate values
            print(f"Permuted images shape: {images.shape}")
            print(f"Images min: {images.min()}, max: {images.max()}")

            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            
            # Debug: Print final image values
            print(f"Final images shape: {images.shape}")
            print(f"Final images min: {images.min()}, max: {images.max()}")

            # Generate latents
            latents = self.pipe.vae.encode(output.images).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
            
            # Debug: Print latent values
            print(f"Latents shape: {latents.shape}")
            print(f"Latents min: {latents.min()}, max: {latents.max()}")

            # Prepare latents for ComfyUI
            latents_for_comfy = {"samples": latents.cpu()}

            return (images, latents_for_comfy)

        except Exception as e:
            print(f"Error in Lumina generation: {str(e)}")
            # Return empty tensors in case of error
            return (torch.zeros((batch_size, height, width, 3), dtype=torch.uint8), 
                    {"samples": torch.zeros((batch_size, 4, height // 8, width // 8), dtype=torch.float32)})

NODE_CLASS_MAPPINGS = {
    "LuminaDiffusersNode": LuminaDiffusersNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaDiffusersNode": "Lumina-Next-SFT Diffusers"
}