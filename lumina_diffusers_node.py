import torch
from diffusers import LuminaText2ImgPipeline, FlowMatchEulerDiscreteScheduler
import comfy.model_management as mm
import os
import traceback
import math
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def apply_model_configurations(self, scale_factor, scaling_watershed, time_aware_scaling, context_drop_ratio):
        if hasattr(self.pipe, 'transformer'):
            self.pipe.transformer.scale_factor = scale_factor
            self.pipe.transformer.scale_watershed = scaling_watershed
            self._apply_config(self.pipe.transformer, 'time_aware_scaling', time_aware_scaling)
            self._apply_config(self.pipe.transformer, 'context_drop_ratio', context_drop_ratio)

    def _apply_config(self, transformer, config_name, value):
        for component in ['text_encoder', 'unet']:
            if hasattr(transformer, component):
                setattr(getattr(transformer, component).config, config_name, value)

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

            scale_factor = math.sqrt(width * height / 1024**2)
            self.apply_model_configurations(scale_factor, scaling_watershed, time_aware_scaling, context_drop_ratio)

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

            images = self.process_output(output.images)
            latents = self.generate_latents(output.images)

            # Save debug image
            if images.shape[0] > 0:
                debug_image = Image.fromarray((images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                debug_image.save("debug_output.png")
                print("Debug image saved successfully.")

            return (images, {"samples": latents.cpu()})

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            traceback.print_exc()
            return (torch.zeros((batch_size, 3, height, width), dtype=torch.float32), 
                    {"samples": torch.zeros((batch_size, 4, height // 8, width // 8), dtype=torch.float32)})

    def process_output(self, images):
        print(f"Initial images shape and dtype: {images.shape}, {images.dtype}")
        
        # Ensure images are in the correct range [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        # Convert to uint8
        images = (images * 255).round().to(torch.uint8)
        print(f"After conversion to uint8: {images.shape}, {images.dtype}")
        
        # Move to CPU and convert to numpy
        images = images.cpu().numpy()
        print(f"After conversion to numpy: {images.shape}, {images.dtype}")
        
        processed_images = []
        for img in images:
            print(f"Processing image: {img.shape}, {img.dtype}")
            
            # Ensure the image is in the format [height, width, channels]
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            elif img.shape[0] == 1:
                img = img.squeeze(0)
                
            print(f"After potential transpose: {img.shape}, {img.dtype}")
            processed_images.append(img)
        
        print(f"Processed images: {len(processed_images)}")
        print(f"First image shape and dtype: {processed_images[0].shape}, {processed_images[0].dtype}")
        
        # Convert back to torch tensor in the format ComfyUI expects
        comfy_images = torch.from_numpy(np.stack(processed_images)).permute(0, 3, 1, 2).float() / 255.0
        print(f"Final comfy_images: {comfy_images.shape}, {comfy_images.dtype}")
        
        return comfy_images

    def generate_latents(self, images):
        with torch.no_grad():
            latents = self.pipe.vae.encode(images.to(self.pipe.vae.dtype)).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
        
        print(f"Latents shape: {latents.shape}")
        print(f"Latents min: {latents.min()}, max: {latents.max()}")
        
        return latents

NODE_CLASS_MAPPINGS = {
    "LuminaDiffusersNode": LuminaDiffusersNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaDiffusersNode": "Lumina-Next-SFT Diffusers"
}
