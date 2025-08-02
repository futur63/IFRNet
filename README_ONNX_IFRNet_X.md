# IFRNet_X ONNX Export

This directory contains scripts to export IFRNet_X model's `flow_inference` method to ONNX format for deployment and inference.

## Files

- `export_onnx_ifrnet_x.py`: Main ONNX export script for IFRNet_X
- `test_onnx_export_ifrnet_x.py`: Test script to verify IFRNet_X functionality
- `onnx_inference_ifrnet_x.py`: Example inference script using ONNX model
- `requirements_onnx.txt`: Required dependencies for ONNX export

## Key Differences from IFRNet_S

IFRNet_X has several key differences from IFRNet_S:

1. **Pixel Unshuffle**: Uses pixel unshuffle preprocessing (3 channels → 12 channels)
2. **Flow Inference Focus**: Exports the `flow_inference` method specifically
3. **Multiple Outputs**: Returns 4 outputs: flows, mask, and residual
4. **Different Input Shape**: Expects (B, 12, H//2, W//2) instead of (B, 3, H, W)

## Installation

Install the required dependencies:

```bash
pip install -r requirements_onnx.txt
```

## Usage

### 1. Test the Export Functionality

First, test if the IFRNet_X model works correctly:

```bash
python test_onnx_export_ifrnet_x.py
```

This will test the model functionality without requiring a trained checkpoint.

### 2. Export Your Trained Model

To export your trained IFRNet_X model to ONNX:

1. Update the checkpoint path in `export_onnx_ifrnet_x.py`:
   ```python
   checkpoint_path = "./path/to/your/IFRNet_X_checkpoint.pth"
   ```

2. Run the export script:
   ```bash
   python export_onnx_ifrnet_x.py
   ```

### 3. Use the ONNX Model for Inference

The export script will create an example inference script. Use it like this:

```python
from onnx_inference_ifrnet_x import inference_with_onnx

# Run flow inference
flow0, flow1, mask, res = inference_with_onnx(
    onnx_path="IFRNet_X_flow.onnx",
    img0_path="frame1.png",
    img1_path="frame2.png", 
    output_path="flow_output.npz",
    time_ratio=0.5  # Time ratio between 0 and 1
)
```

## Model Input/Output Specifications

### Inputs
- `img0_`: First input frame (B, 12, H//2, W//2) - Pixel unshuffled RGB image
- `img1_`: Second input frame (B, 12, H//2, W//2) - Pixel unshuffled RGB image  
- `embt`: Time embedding (B, 1, 1, 1) - Time ratio between 0 and 1

### Outputs
- `up_flow0_1`: Flow from img0 to target (B, 2, H, W)
- `up_flow1_1`: Flow from img1 to target (B, 2, H, W)
- `up_mask_1`: Mask for blending (B, 1, H, W)
- `up_res_1`: Residual for refinement (B, 3, H, W)

## Preprocessing Pipeline

IFRNet_X requires specific preprocessing:

1. **Normalize images**: Subtract mean from both images
2. **Pixel unshuffle**: Convert (B, 3, H, W) → (B, 12, H//2, W//2)
3. **Time embedding**: Create time ratio tensor

```python
import torch.nn.functional as F

def preprocess_for_ifrnet_x(img0, img1, embt):
    # Normalize
    mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
    img0_norm = img0 - mean_
    img1_norm = img1 - mean_
    
    # Pixel unshuffle
    img0_ = F.pixel_unshuffle(img0_norm, 2)
    img1_ = F.pixel_unshuffle(img1_norm, 2)
    
    return img0_, img1_, embt
```

## Features

- **Flow Inference Focus**: Exports the core flow inference method
- **Pixel Unshuffle Support**: Handles the unique preprocessing of IFRNet_X
- **Multiple Outputs**: Returns flows, mask, and residual for complete pipeline
- **Dynamic Axes**: Supports flexible input sizes
- **Validation**: Automatic ONNX model validation and testing

## Example Usage

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torch.nn.functional as F

# Load ONNX model
session = ort.InferenceSession("IFRNet_X_flow.onnx")

# Load and preprocess images
img0 = load_image("frame1.png")  # Shape: (1, 3, H, W), range [0, 1]
img1 = load_image("frame2.png")  # Shape: (1, 3, H, W), range [0, 1]

# Preprocess for IFRNet_X
img0_, img1_ = preprocess_for_ifrnet_x(img0, img1)  # Shape: (1, 12, H//2, W//2)

# Create time embedding (0.5 = middle frame)
embt = np.array([[0.5]], dtype=np.float32).reshape(1, 1, 1, 1)

# Run inference
inputs = {'img0_': img0_, 'img1_': img1_, 'embt': embt}
outputs = session.run(None, inputs)
flow0, flow1, mask, res = outputs

# Use outputs for complete pipeline
# flow0, flow1: Optical flows
# mask: Blending mask
# res: Residual for refinement
```

## Complete Pipeline Integration

For complete frame interpolation, you'll need to implement the full pipeline:

```python
def complete_ifrnet_x_inference(flow0, flow1, mask, res, img0, img1, mean_):
    """Complete IFRNet_X inference pipeline"""
    # Warp images using flows
    img0_warp = warp(img0, flow0)
    img1_warp = warp(img1, flow1)
    
    # Blend warped images
    imgt_merge = mask * img0_warp + (1 - mask) * img1_warp + mean_
    
    # Add residual
    imgt_pred = imgt_merge + res
    imgt_pred = np.clip(imgt_pred, 0, 1)
    
    return imgt_pred
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install torch torchvision onnx onnxruntime numpy pillow imageio
   ```

2. **Pixel Unshuffle Issues**: Ensure input images are properly preprocessed
   - Input should be (B, 12, H//2, W//2) for the ONNX model
   - Use `F.pixel_unshuffle(img, 2)` for preprocessing

3. **Shape Mismatches**: 
   - IFRNet_X expects pixel unshuffled inputs
   - Output shapes depend on input resolution

4. **ONNX Version**: If you encounter ONNX compatibility issues, try different opset versions (11, 12, 13, 16).

### Performance Tips

- Use ONNX Runtime with GPU provider for faster inference
- The flow inference is the most computationally intensive part
- Consider batching multiple frames for better throughput

## Advanced Configuration

You can customize the export parameters in `export_onnx_ifrnet_x.py`:

```python
export_ifrnet_x_to_onnx(
    checkpoint_path="your_checkpoint.pth",
    output_path="IFRNet_X_flow.onnx",
    input_height=720,      # Input height (will be halved for pixel unshuffle)
    input_width=1280,      # Input width (will be halved for pixel unshuffle)
    batch_size=1,          # Batch size
    opset_version=16,      # ONNX opset version
    dynamic_axes=True      # Enable dynamic axes
)
```

## Comparison with IFRNet_S

| Feature | IFRNet_S | IFRNet_X |
|---------|----------|----------|
| Input Shape | (B, 3, H, W) | (B, 12, H//2, W//2) |
| Preprocessing | Simple normalization | Pixel unshuffle + normalization |
| Output | Single frame | 4 outputs (flows, mask, residual) |
| Focus | Complete inference | Flow inference only |
| Complexity | Lower | Higher |

## License

This ONNX export functionality follows the same license as the original IFRNet project. 