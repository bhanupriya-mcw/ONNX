import onnx

model = onnx.load("resnet18.onnx")  # Load the model
print(onnx.helper.printable_graph(model.graph))  # Print model structure
