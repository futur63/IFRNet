# IFRNet_S ONNX Export

This directory contains scripts to export IFRNet_S model to ONNX format for deployment and inference.

## Files

- `export_onnx_ifrnet_s.py`: Main ONNX export script
- `test_onnx_export.py`: Test script to verify export functionality
- `onnx_inference_example.py`: Example inference script using ONNX model
- `requirements_onnx.txt`: Required dependencies for ONNX export

## Installation

Install the required dependencies:

```bash
pip install -r requirements_onnx.txt
```

## Usage

### 1. Test the Export Functionality

First, test if the ONNX export works correctly:

```bash
python test_onnx_export.py
```

This will test the export functionality without requiring a trained checkpoint.

### 2. Export Your Trained Model

To export your trained IFRNet_S model to ONNX:

1. Update the checkpoint path in `export_onnx_ifrnet_s.py`:
   ```python
   checkpoint_path = "./path/to/your/IFRNet_S_checkpoint.pth"
   ```

2. Run the export script:
   ```bash
   python export_onnx_ifrnet_s.py
   ```

### 3. Use the ONNX Model for Inference

The export script will create an example inference script. Use it like this:

```python
from onnx_inference_example import inference_with_onnx

# Run inference
inference_with_onnx(
    onnx_path="IFRNet_S.onnx",
    img0_path="frame1.png",
    img1_path="frame2.png", 
    output_path="interpolated_frame.png",
    time_ratio=0.5  # Time ratio between 0 and 1
)
```

## Model Input/Output Specifications

### Inputs
- `img0`: First input frame (B, 3, H, W) - RGB image normalized to [0, 1]
- `img1`: Second input frame (B, 3, H, W) - RGB image normalized to [0, 1]  
- `embt`: Time embedding (B, 1, 1, 1) - Time ratio between 0 and 1

### Output
- `imgt_pred`: Predicted intermediate frame (B, 3, H, W) - RGB image in range [0, 1]

## Features

- **Dynamic Axes**: The exported model supports dynamic batch sizes and image dimensions
- **Validation**: Automatic ONNX model validation and ONNX Runtime testing
- **Flexible Input**: Supports different input image sizes
- **Time Control**: Control the interpolation time ratio (0.0 = first frame, 1.0 = second frame)

## Example Usage

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("IFRNet_S.onnx")

# Load and preprocess images
img0 = load_image("frame1.png")  # Shape: (1, 3, H, W), range [0, 1]
img1 = load_image("frame2.png")  # Shape: (1, 3, H, W), range [0, 1]

# Create time embedding (0.5 = middle frame)
embt = np.array([[0.5]], dtype=np.float32).reshape(1, 1, 1, 1)

# Run inference
inputs = {'img0': img0, 'img1': img1, 'embt': embt}
outputs = session.run(None, inputs)
result = outputs[0]  # Shape: (1, 3, H, W), range [0, 1]

# Save result
save_image(result, "interpolated_frame.png")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install torch torchvision onnx onnxruntime numpy pillow imageio
   ```

2. **CUDA Issues**: The export script uses CPU by default. For GPU inference, modify the model loading in the export script.

3. **Memory Issues**: For large images, try reducing the batch size or using smaller input dimensions.

4. **ONNX Version**: If you encounter ONNX compatibility issues, try different opset versions (11, 12, 13).

### Performance Tips

- Use ONNX Runtime with GPU provider for faster inference
- Batch multiple frames together for better throughput
- Consider using TensorRT for further optimization on NVIDIA GPUs

## Advanced Configuration

You can customize the export parameters in `export_onnx_ifrnet_s.py`:

```python
export_ifrnet_s_to_onnx(
    checkpoint_path="your_checkpoint.pth",
    output_path="IFRNet_S.onnx",
    input_height=720,      # Default input height
    input_width=1280,      # Default input width  
    batch_size=1,          # Batch size
    opset_version=11,      # ONNX opset version
    dynamic_axes=True      # Enable dynamic axes
)
```

## License

This ONNX export functionality follows the same license as the original IFRNet project. 