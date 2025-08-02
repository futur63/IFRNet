import os
import torch
import torch.nn as nn
import numpy as np
from models.IFRNet_X import Model
import onnx
import onnxruntime as ort

#  Key Differences from IFRNet_S:
# Feature	IFRNet_S	IFRNet_X
# Input Shape	(B, 3, H, W)	(B, 12, H//2, W//2)
# Preprocessing	Simple normalization	Pixel unshuffle + normalization
# Output	Single frame	4 outputs (flows, mask, residual)
# Focus	Complete inference	Flow inference only


class IFRNetXForONNX(nn.Module):
    """
    Wrapper class for IFRNet_X model optimized for ONNX export
    Focuses on the flow_inference method
    """
    def __init__(self, model):
        super(IFRNetXForONNX, self).__init__()
        self.model = model
        
    def forward(self, img0_, img1_, embt):
        """
        Forward pass optimized for ONNX export
        Args:
            img0_: Preprocessed first image (B, 12, H, W) - pixel unshuffled
            img1_: Preprocessed second image (B, 12, H, W) - pixel unshuffled
            embt: Time embedding (B, 1, 1, 1)
        Returns:
            up_flow0_1: Flow from img0 to target (B, 2, H, W)
            up_flow1_1: Flow from img1 to target (B, 2, H, W)
            up_mask_1: Mask for blending (B, 1, H, W)
            up_res_1: Residual for refinement (B, 3, H, W)
        """
        return self.model.flow_inference(img0_, img1_, embt)


def export_ifrnet_x_to_onnx(checkpoint_path, output_path, input_height=720, input_width=1280, 
                           batch_size=1, opset_version=16, dynamic_axes=True):
    """
    Export IFRNet_X model to ONNX format
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_path: Path to save the ONNX model
        input_height: Input image height (default: 720)
        input_width: Input image width (default: 1280)
        batch_size: Batch size (default: 1)
        opset_version: ONNX opset version (default: 16)
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
    model_onnx = IFRNetXForONNX(model)
    model_onnx.eval()
    
    # Create dummy inputs (note: IFRNet_X expects pixel unshuffled inputs)
    if dynamic_axes:
        # Dynamic batch size and spatial dimensions
        # IFRNet_X uses pixel unshuffle, so input is 12 channels instead of 3
        dummy_img0_ = torch.randn(batch_size, 12, input_height//2, input_width//2)  # Pixel unshuffled
        dummy_img1_ = torch.randn(batch_size, 12, input_height//2, input_width//2)  # Pixel unshuffled
        dummy_embt = torch.randn(batch_size, 1, 1, 1)
        
        # Define dynamic axes
        dynamic_axes_dict = {
            'img0_': {0: 'batch_size', 2: 'height', 3: 'width'},
            'img1_': {0: 'batch_size', 2: 'height', 3: 'width'},
            'embt': {0: 'batch_size'},
            'up_flow0_1': {0: 'batch_size', 2: 'height', 3: 'width'},
            'up_flow1_1': {0: 'batch_size', 2: 'height', 3: 'width'},
            'up_mask_1': {0: 'batch_size', 2: 'height', 3: 'width'},
            'up_res_1': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    else:
        # Fixed dimensions
        dummy_img0_ = torch.randn(batch_size, 12, input_height//2, input_width//2)
        dummy_img1_ = torch.randn(batch_size, 12, input_height//2, input_width//2)
        dummy_embt = torch.randn(batch_size, 1, 1, 1)
        dynamic_axes_dict = None
    
    # Export to ONNX
    print("Exporting IFRNet_X flow_inference to ONNX...")
    torch.onnx.export(
        model_onnx,
        (dummy_img0_, dummy_img1_, dummy_embt),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['img0_', 'img1_', 'embt'],
        output_names=['up_flow0_1', 'up_flow1_1', 'up_mask_1', 'up_res_1'],
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
        'img0_': dummy_img0_.numpy(),
        'img1_': dummy_img1_.numpy(),
        'embt': dummy_embt.numpy()
    }
    
    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"ONNX Runtime test successful!")
    print(f"Output shapes: flow0{ort_outputs[0].shape}, flow1{ort_outputs[1].shape}, mask{ort_outputs[2].shape}, res{ort_outputs[3].shape}")
    
    return output_path


def create_inference_script():
    """
    Create a simple inference script for the exported ONNX model
    """
    inference_script = '''import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torch.nn.functional as F

def load_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def pixel_unshuffle_preprocess(img_tensor):
    """Apply pixel unshuffle preprocessing for IFRNet_X"""
    # Convert numpy to tensor
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor)
    
    # Apply pixel unshuffle
    img_unshuffled = F.pixel_unshuffle(img_tensor, 2)
    return img_unshuffled.numpy()

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
    
    # Apply pixel unshuffle preprocessing
    img0_ = pixel_unshuffle_preprocess(img0)
    img1_ = pixel_unshuffle_preprocess(img1)
    
    # Create time embedding
    embt = np.array([[time_ratio]], dtype=np.float32)
    embt = np.expand_dims(embt, axis=(0, 2, 3))  # Shape: (1, 1, 1, 1)
    
    # Run inference
    inputs = {
        'img0_': img0_,
        'img1_': img1_,
        'embt': embt
    }
    
    outputs = session.run(None, inputs)
    up_flow0_1, up_flow1_1, up_mask_1, up_res_1 = outputs
    
    print(f"Flow inference completed!")
    print(f"Output shapes: flow0{up_flow0_1.shape}, flow1{up_flow1_1.shape}, mask{up_mask_1.shape}, res{up_res_1.shape}")
    
    # Note: This is just the flow inference part. For complete inference,
    # you would need to implement the full pipeline including warping and blending
    return up_flow0_1, up_flow1_1, up_mask_1, up_res_1

if __name__ == "__main__":
    # Example usage
    onnx_model_path = "IFRNet_X_flow.onnx"
    img0_path = "input_frame1.png"
    img1_path = "input_frame2.png"
    output_path = "flow_output.npz"
    
    # Run flow inference
    flow0, flow1, mask, res = inference_with_onnx(onnx_model_path, img0_path, img1_path, output_path, time_ratio=0.5)
    
    # Save outputs
    np.savez(output_path, flow0=flow0, flow1=flow1, mask=mask, res=res)
    print(f"Flow outputs saved to {output_path}")
'''
    
    with open('IFRNet/onnx_inference_ifrnet_x.py', 'w') as f:
        f.write(inference_script)
    
    print("Created ONNX inference example script: onnx_inference_ifrnet_x.py")


def main():
    """
    Main function to export IFRNet_X to ONNX
    """
    # Configuration
    checkpoint_path = "./checkpoint/IFRNet_X/IFRNet_X_Vimeo90K.pth"  # Update with actual path
    output_path = "./IFRNet_X_flow.onnx"
    
    # Export model
    try:
        export_ifrnet_x_to_onnx(
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
        
        print("\n" + "="*50)
        print("ONNX Export Summary:")
        print("="*50)
        print(f"Model: IFRNet_X (flow_inference)")
        print(f"Output: {output_path}")
        print(f"Input shapes: img0_/img1_ (B, 12, H//2, W//2), embt (B, 1, 1, 1)")
        print(f"Output shapes: up_flow0_1/up_flow1_1 (B, 2, H, W), up_mask_1 (B, 1, H, W), up_res_1 (B, 3, H, W)")
        print(f"Dynamic axes: Enabled for flexible input sizes")
        print(f"Inference script: onnx_inference_ifrnet_x.py")
        print("="*50)
        
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install onnx onnxruntime")


if __name__ == "__main__":
    main() 