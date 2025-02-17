import onnx
import onnxruntime as ort
import numpy as np
import time
import pandas as pd
from collections import Counter
import onnx_graphsurgeon as gs

# Load ONNX Model
onnx_model_path = r"D:\Multicoreware\onnx\export_resnet\resnet18.onnx"  # Change this to your ONNX model path
model = onnx.load(onnx_model_path)
graph = gs.import_onnx(model)  

# Count Node Occurrences
node_types = [node.op for node in graph.nodes]  
node_counts = Counter(node_types)

# Convert to DataFrame
df = pd.DataFrame(node_counts.items(), columns=["Node Type", "Occurrences"])
df = df.sort_values(by="Occurrences", ascending=False)

# Load ONNX Runtime
ort_session = ort.InferenceSession(onnx_model_path)

# Get input and output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Handle Dynamic Input Shapes
input_shape = ort_session.get_inputs()[0].shape
fixed_input_shape = [1 if isinstance(dim, (str, type(None))) else dim for dim in input_shape]

# Generate Dummy Input
dummy_input = np.random.randn(*fixed_input_shape).astype(np.float32)

# FPS Calculation (Optimized)
def measure_fps(iterations=100):
    start_time = time.perf_counter()  
    for _ in range(iterations):
        _ = ort_session.run([output_name], {input_name: dummy_input})
    end_time = time.perf_counter()
    avg_time_per_inference = (end_time - start_time) / iterations
    return 1 / avg_time_per_inference if avg_time_per_inference > 0 else 0

fps = round(measure_fps(), 2)

# Model Details
model_details = {
    "Model": "ResNet18",
    "ONNX File": onnx_model_path,
    "Input Shape": fixed_input_shape,
    "FPS": fps
}

# Print Results
print("\nModel Details:")
for key, value in model_details.items():
    print(f"{key}: {value}")

print("\nNode Type Occurrences:")
print(df)

# Save node count details to CSV
df.to_csv("onnx_node_analysis.csv", index=False)
print("\nNode occurrences saved to onnx_node_analysis.csv")
