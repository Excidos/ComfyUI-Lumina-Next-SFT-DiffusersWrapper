import torch
from diffusers import LuminaText2ImgPipeline, FlowMatchEulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer
import comfy.model_management as mm
import os
import traceback
import math
import numpy as np

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
                "scaling_watershed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "proportional_attn": ("BOOLEAN", {"default": True}),
                "clean_caption": ("BOOLEAN", {"default": True}),
                "max_sequence_length": ("INT", {"default": 256, "min": 64, "max": 512}),
            },
            "optional": {
                "output_type": (["pil", "latent", "pt"], {"default": "pil"}),
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
                raise ValueError(f"Model path does not exist: {full_path}")

            self.pipe = LuminaText2ImgPipeline.from_pretrained(full_path, torch_dtype=dtype)
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.to(device)
            print("Pipeline successfully loaded and moved to device.")
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
            traceback.print_exc()

    def generate(self, model_path, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, seed,
                batch_size, scaling_watershed, proportional_attn, clean_caption, max_sequence_length,
                output_type="pil"):
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

            # Calculate the default image size based on the transformer's configuration
            default_image_size = self.pipe.transformer.config.sample_size * self.pipe.vae_scale_factor

            # Set cross_attention_kwargs as an attribute if proportional_attn is True
            if proportional_attn:
                self.pipe.cross_attention_kwargs = {"base_sequence_length": (default_image_size // 16) ** 2}
            else:
                self.pipe.cross_attention_kwargs = None

            print(f"Starting generation with seed: {seed}")

            # Prepare the arguments for the pipeline call
            pipe_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": batch_size,
                "generator": generator,
                "output_type": output_type,
                "clean_caption": clean_caption,
                "max_sequence_length": max_sequence_length,
                "scaling_watershed": scaling_watershed,
            }

            output = self.pipe(**pipe_args)

            processed_images = self.process_output(output.images, output_type)

            # Return processed images and latents if output_type is "latent"
            if output_type == "latent":
                return (processed_images, output.images)
            else:
                return (processed_images, None)

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            traceback.print_exc()
            # Return empty tensors for both IMAGE and LATENT in case of error
            return (torch.zeros((batch_size, 3, height, width), dtype=torch.float32),
                    torch.zeros((batch_size, 4, height // 8, width // 8), dtype=torch.float32))

    def process_output(self, images, output_type):
        if output_type == "pil":
            # Convert PIL images to tensors
            return torch.stack([torch.from_numpy(np.array(img)).float() / 255.0 for img in images])
        elif output_type == "latent":
            # Latents are already in the correct format
            return images
        elif output_type == "pt":
            # PyTorch tensors might need normalization
            return images.clamp(0, 1)
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

NODE_CLASS_MAPPINGS = {
    "LuminaDiffusersNode": LuminaDiffusersNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaDiffusersNode": "Lumina-Next-SFT Diffusers"
}
