import os
import torch
import torch.nn as nn
import numpy as np
from models.IFRNet_S import Model
import onnx
import onnxruntime as ort


class IFRNetSForONNX(nn.Module):
    """
    Wrapper class for IFRNet_S model optimized for ONNX export
    """
    def __init__(self, model):
        super(IFRNetSForONNX, self).__init__()
        self.model = model
        
    def forward(self, img0, img1, embt):
        """
        Forward pass optimized for ONNX export
        Args:
            img0: First input image (B, 3, H, W)
            img1: Second input image (B, 3, H, W) 
            embt: Time embedding (B, 1, 1, 1)
        Returns:
            imgt_pred: Predicted intermediate frame (B, 3, H, W)
        """
        return self.model.inference(img0, img1, embt)


def export_ifrnet_s_to_onnx(checkpoint_path, output_path, input_height=720, input_width=1280, 
                           batch_size=1, opset_version=11, dynamic_axes=True):
    """
    Export IFRNet_S model to ONNX format
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_path: Path to save the ONNX model
        input_height: Input image height (default: 720)
        input_width: Input image width (default: 1280)
        batch_size: Batch size (default: 1)
        opset_version: ONNX opset version (default: 11)
        dynamic_axes: Whether to use dynamic axes for flexible input sizes
    """
    
    # Initialize model
    model = Model()
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model for ONNX export")
    
    # Create wrapper for ONNX export
    model_onnx = IFRNetSForONNX(model)
    model_onnx.eval()
    
    # Create dummy inputs
    if dynamic_axes:
        # Dynamic batch size and spatial dimensions
        dummy_img0 = torch.randn(batch_size, 3, input_height, input_width)
        dummy_img1 = torch.randn(batch_size, 3, input_height, input_width)
        dummy_embt = torch.randn(batch_size, 1, 1, 1)
        
        # Define dynamic axes
        dynamic_axes_dict = {
            'img0': {0: 'batch_size', 2: 'height', 3: 'width'},
            'img1': {0: 'batch_size', 2: 'height', 3: 'width'},
            'embt': {0: 'batch_size'},
            'imgt_pred': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    else:
        # Fixed dimensions
        dummy_img0 = torch.randn(batch_size, 3, input_height, input_width)
        dummy_img1 = torch.randn(batch_size, 3, input_height, input_width)
        dummy_embt = torch.randn(batch_size, 1, 1, 1)
        dynamic_axes_dict = None
    
    # Export to ONNX
    print("Exporting model to ONNX...")
    torch.onnx.export(
        model_onnx,
        (dummy_img0, dummy_img1, dummy_embt),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['img0', 'img1', 'embt'],
        output_names=['imgt_pred'],
        dynamic_axes=dynamic_axes_dict,
        verbose=False
    )
    
    print(f"ONNX model exported to {output_path}")
    
    # Validate ONNX model
    print("Validating ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # Test ONNX model with ONNX Runtime
    print("Testing ONNX model with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)
    
    # Test with dummy data
    ort_inputs = {
        'img0': dummy_img0.numpy(),
        'img1': dummy_img1.numpy(),
        'embt': dummy_embt.numpy()
    }
    
    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"ONNX Runtime test successful! Output shape: {ort_outputs[0].shape}")
    
    return output_path


def create_inference_script():
    """
    Create a simple inference script for the exported ONNX model
    """
    inference_script = '''import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

def load_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def save_image(image_array, output_path):
    """Save image array to file"""
    img = image_array[0].transpose(1, 2, 0)  # CHW to HWC
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)

def inference_with_onnx(onnx_path, img0_path, img1_path, output_path, time_ratio=0.5):
    """
    Run inference using ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        img0_path: Path to first input image
        img1_path: Path to second input image
        output_path: Path to save output image
        time_ratio: Time ratio between 0 and 1 (default: 0.5 for middle frame)
    """
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    
    # Load and preprocess images
    img0 = load_image(img0_path)
    img1 = load_image(img1_path)
    
    # Create time embedding
    embt = np.array([[time_ratio]], dtype=np.float32)
    embt = np.expand_dims(embt, axis=(0, 2, 3))  # Shape: (1, 1, 1, 1)
    
    # Run inference
    inputs = {
        'img0': img0,
        'img1': img1,
        'embt': embt
    }
    
    outputs = session.run(None, inputs)
    result = outputs[0]
    
    # Save result
    save_image(result, output_path)
    print(f"Intermediate frame saved to {output_path}")
    
    return result

if __name__ == "__main__":
    # Example usage
    onnx_model_path = "IFRNet_S.onnx"
    img0_path = "input_frame1.png"
    img1_path = "input_frame2.png"
    output_path = "interpolated_frame.png"
    
    inference_with_onnx(onnx_model_path, img0_path, img1_path, output_path, time_ratio=0.5)
'''
    
    with open('IFRNet/onnx_inference_example.py', 'w') as f:
        f.write(inference_script)
    
    print("Created ONNX inference example script: onnx_inference_example.py")


def main():
    """
    Main function to export IFRNet_S to ONNX
    """
    # Configuration
    checkpoint_path = "./checkpoint/IFRNet_S/IFRNet_S_Vimeo90K.pth"  # Update with actual path
    output_path = "./IFRNet_S.onnx"
    
    # Export model
    try:
        export_ifrnet_s_to_onnx(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            input_height=720,
            input_width=1280,
            batch_size=1,
            opset_version=16,
            dynamic_axes=True
        )
        
        # Create inference script
        create_inference_script()
        
        print("\\n" + "="*50)
        print("ONNX Export Summary:")
        print("="*50)
        print(f"Model: IFRNet_S")
        print(f"Output: {output_path}")
        print(f"Input shapes: img0/img1 (B, 3, H, W), embt (B, 1, 1, 1)")
        print(f"Output shape: imgt_pred (B, 3, H, W)")
        print(f"Dynamic axes: Enabled for flexible input sizes")
        print(f"Inference script: onnx_inference_example.py")
        print("="*50)
        
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install onnx onnxruntime")


if __name__ == "__main__":
    main() 