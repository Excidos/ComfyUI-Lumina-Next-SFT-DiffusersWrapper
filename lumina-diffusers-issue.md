# Lumina Diffusers Wrapper Issue in ComfyUI

## Problem Description

I'm experiencing an issue with my Lumina Diffusers wrapper in ComfyUI. The model seems to generate a coherent image, but ComfyUI is unable to preview or decode it properly.

## Error Message

```
!!! Exception during processing!!! Cannot handle this data type: (1, 1, 1024), |u1
Traceback (most recent call last):
  File "K:\AI-Art\ComfyUI_windows_portable\python_embeded\Lib\site-packages\PIL\Image.py", line 3277, in fromarray
    mode, rawmode = *fromarray*typemap[typekey]
                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^
KeyError: ((1, 1, 1024), '|u1')
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "K:\AI-Art\ComfyUI_windows_portable\ComfyUI\execution.py", line 151, in recursive_execute
    output_data, output_ui = get_output_data(obj, input_data_all)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "K:\AI-Art\ComfyUI_windows_portable\ComfyUI\execution.py", line 81, in get_output_data
    return_values = map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "K:\AI-Art\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-0246\utils.py", line 381, in new_func
    res_value = old_func(*final_args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "K:\AI-Art\ComfyUI_windows_portable\ComfyUI\execution.py", line 74, in map_node_over_list
    results.append(getattr(obj, func)(**slice_dict(input_data_all, i)))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "K:\AI-Art\ComfyUI_windows_portable\ComfyUI\nodes.py", line 1437, in save_images
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "K:\AI-Art\ComfyUI_windows_portable\python_embeded\Lib\site-packages\PIL\Image.py", line 3281, in fromarray
    raise TypeError(msg) from e
TypeError: Cannot handle this data type: (1, 1, 1024), |u1
```

## Code Snippet

The relevant part of the `LuminaDiffusersNode` class:

```python
def process_output(self, images):
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    images = (images * 255).round().to(torch.uint8)
    
    # Ensure correct shape: [batch_size, channels, height, width]
    images = images.cpu().numpy()
    
    processed_images = []
    for img in images:
        # Reshape if necessary
        if img.shape[0] == 1:
            img = img.squeeze(0)  # Remove batch dimension if it's 1
        
        # Ensure the image is in the format [height, width, channels]
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        processed_images.append(img)
    
    print(f"Processed images: {len(processed_images)}")
    print(f"First image shape: {processed_images[0].shape}")
    
    # Convert to the format ComfyUI expects: [batch_size, channels, height, width]
    comfy_images = torch.from_numpy(np.stack(processed_images)).permute(0, 3, 1, 2).float() / 255.0
    
    return comfy_images
```

## ComfyUI Workflow

![image](https://github.com/user-attachments/assets/1ccad406-a902-419a-9fde-5971c82fcef6)

## Debug Output

![debug_output](https://github.com/user-attachments/assets/084e3dc3-6d35-4b79-8d31-7b662d9b8b3f)


The `debug_output.png` shows that the Lumina wrapper model is generating a mostly coherent image:



## Additional Information

- The model successfully loads and generates an image.
- The issue occurs when trying to preview or decode the latent in ComfyUI.
- The error suggests a problem with the data type or shape of the processed image.

## Questions

1. Is there a mismatch between the output format of the Lumina model and what ComfyUI expects?
2. Could the issue be in the `process_output` method, specifically in how the image is reshaped or converted?
3. Is there a problem with how the latents are being handled or decoded?

Any assistance in resolving this issue would be greatly appreciated.
