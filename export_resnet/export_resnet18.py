import torch
import torchvision.models as models

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# Dummy input for export (e.g., 1x3x224x224 for ResNet)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX with opset 12
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",  # Corrected filename
    export_params=True,
    opset_version=12,  
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("ResNet model exported to resnet18.onnx")  
