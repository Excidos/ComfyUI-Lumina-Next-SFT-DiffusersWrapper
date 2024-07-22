# ComfyUI-Lumina-Next-SFT-DiffusersWrapper

# Lumina Diffusers Node for ComfyUI

This custom node integrates the Lumina-Next-SFT model into ComfyUI, allowing for high-quality image generation using the Lumina text-to-image pipeline. Still a massive work in progress but functional.

## Features

- Utilizes the Lumina-Next-SFT model for image generation
- Supports various generation parameters including prompt, negative prompt, inference steps, and guidance scale
- Implements Lumina-specific features such as scaling watershed and proportional attention
- Automatic model downloading if not found locally
- Outputs both generated images and latent representations

## Installation

1. Ensure you have ComfyUI installed and set up.
2. Clone this repository into your ComfyUI custom nodes directory:
   ```
   git clone https://github.com/Excidos/ComfyUI-Lumina-Diffusers.git
   ```
3. Dependencies will be automatically installed

   NOTE: Will install a development branch of diffusers (may conflict with some nodes)

## Usage

1. Start ComfyUI.
2. Look for the "Lumina-Next-SFT Diffusers" node in the node selection menu.
3. Add the node to your workflow.
4. Connect the necessary inputs and outputs.
5. Configure the node parameters as desired.
6. Run your workflow to generate images.

## Parameters

- `model_path`: Path to the Lumina model (default: "Lumina-Next-SFT-diffusers")
- `prompt`: Text prompt for image generation
- `negative_prompt`: Negative text prompt
- `num_inference_steps`: Number of denoising steps (default: 30)
- `guidance_scale`: Classifier-free guidance scale (default: 4.0)
- `width`: Output image width (default: 1024)
- `height`: Output image height (default: 1024)
- `seed`: Random seed for generation (-1 for random)
- `batch_size`: Number of images to generate in one batch (default: 1)
- `scaling_watershed`: Scaling watershed parameter (default: 1.0)
- `proportional_attn`: Enable proportional attention (default: True)
- `clean_caption`: Clean input captions (default: True)
- `max_sequence_length`: Maximum sequence length for text input (default: 256)
- `t_shift`: Time shift parameter for scheduler (default: 4)

## Outputs

1. `IMAGE`: Generated image(s) in tensor format
2. `LATENT`: Latent representation of the generated image(s) (Currently Not Working)

## Notes

- The node will automatically download the Lumina model if it's not found in the specified path.
- Ensure you have sufficient GPU memory for running the Lumina model.
- This is my first Diffusers Wrapper and is a big work in progress.

## Example Outputs

![Screenshot 2024-07-22 103940](https://github.com/user-attachments/assets/5678611c-c468-40df-b6d9-b44c64ac2fd9)

![Screenshot 2024-07-22 131142](https://github.com/user-attachments/assets/ffa516d6-cb72-4c51-bf19-e6c85b490cc3)

![Screenshot 2024-07-22 104629](https://github.com/user-attachments/assets/12cc7089-d7f7-42ae-8228-43b77f1e24fa)

![image](https://github.com/user-attachments/assets/a0851ad1-10e8-4eca-9f1f-82ab94f60427)


## Troubleshooting

If you encounter any issues, please check the console output for error messages. Common issues include:

- Insufficient GPU memory
- Missing dependencies
- Incorrect model path

For further assistance, please open an issue on the GitHub repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgements

- [Lumina-Next-SFT-Diffusers]([https://www.luminai.com/](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)) for the Lumina-Next-SFT model
- The ComfyUI community for their continuous support and inspiration
