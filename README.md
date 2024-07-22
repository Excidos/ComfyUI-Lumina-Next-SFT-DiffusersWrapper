# ComfyUI-Lumina-Next-SFT-DiffusersWrapper

## Lumina Diffusers Node for ComfyUI

This custom node seamlessly integrates the Lumina-Next-SFT model into ComfyUI, enabling high-quality image generation using the advanced Lumina text-to-image pipeline. While still under active development, it offers a robust and functional implementation.

## Features

- Harnesses the power of the Lumina-Next-SFT model for state-of-the-art image generation
- Offers a wide range of generation parameters for fine-tuned control
- Implements Lumina-specific features including scaling watershed and proportional attention
- Incorporates time-aware scaling for improved generation quality
- Utilizes an ODE solver for enhanced stability and quality in the generation process
- Automatic model downloading for seamless setup
- Outputs generated images and working on latent representations (Latent isn't working at the moment)

## Installation

## Now in ComfyUI Manager!

or for manual installation.

1. Ensure you have ComfyUI installed and properly set up.
2. Clone this repository into your ComfyUI custom nodes directory:
   ```
   git clone https://github.com/Excidos/ComfyUI-Lumina-Diffusers.git
   ```
3. The required dependencies will be automatically installed.

   **NOTE:** This installation includes a development branch of diffusers, which may conflict with some existing nodes.

## Usage

1. Launch ComfyUI.
2. Locate the "Lumina-Next-SFT Diffusers" node in the node selection menu.
3. Add the node to your workflow.
4. Connect the necessary inputs and outputs.
5. Configure the node parameters as desired.
6. Execute your workflow to generate images.

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
- `solver`: ODE solver type (options: "euler", "midpoint", "rk4", default: "midpoint")

## Outputs

1. `IMAGE`: Generated image(s) in tensor format
2. `LATENT`: Latent representation of the generated image(s)

## Advanced Features

### Time-Aware Scaling

The node now implements time-aware scaling, which adjusts the generation process based on the current timestep. This results in improved image quality, especially for high-resolution outputs.

### ODE Solver

An Ordinary Differential Equation (ODE) solver has been integrated into the generation process. This advanced technique enhances the stability and quality of the generated images, particularly for complex prompts or high-resolution outputs.
## Example Outputs

![Screenshot 2024-07-22 103940](https://github.com/user-attachments/assets/5678611c-c468-40df-b6d9-b44c64ac2fd9)

![image](https://github.com/user-attachments/assets/e839fb67-851f-456d-aec7-e727b95dc968)

![Screenshot 2024-07-22 131142](https://github.com/user-attachments/assets/ffa516d6-cb72-4c51-bf19-e6c85b490cc3)

![Screenshot 2024-07-22 104629](https://github.com/user-attachments/assets/12cc7089-d7f7-42ae-8228-43b77f1e24fa)

![image](https://github.com/user-attachments/assets/94e046e3-b39b-4b3c-ae7f-723b1c8af70f)

![image](https://github.com/user-attachments/assets/36295516-2ced-4a17-89ac-85ae8ae313bf)

![image](https://github.com/user-attachments/assets/1890f870-761e-4510-aba2-b6bcf55ea1e7)

![ComfyUI_temp_ntirq_00004_](https://github.com/user-attachments/assets/1bcacf31-208a-4983-8bc3-e2480b226ccc)

![ComfyUI_temp_ntirq_00003_](https://github.com/user-attachments/assets/c787f20b-3470-4c52-9907-f926d2729e02)

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
