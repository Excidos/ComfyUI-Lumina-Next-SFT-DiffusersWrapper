import torch

class REGIONAL_PROMPT:
    def __init__(self, prompt, mask):
        self.prompt = prompt
        self.mask = mask.to('cpu')  # Store mask on CPU initially

class AreaPromptNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "x1": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "y1": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "x2": ("INT", {"default": 1024, "min": 0, "max": 1024}),
                "y2": ("INT", {"default": 1024, "min": 0, "max": 1024}),
            },
            "optional": {
                "previous_prompts": ("AREA_PROMPTS",),
            }
        }
    
    RETURN_TYPES = ("AREA_PROMPTS",)
    FUNCTION = "add_prompt"
    CATEGORY = "LuminaWrapper"

    def add_prompt(self, prompt, x1, y1, x2, y2, previous_prompts=None):
        new_prompt = {
            "prompt": prompt,
            "area": (x1, y1, x2, y2)
        }
        if previous_prompts is None:
            return ([new_prompt],)
        else:
            return (previous_prompts + [new_prompt],)