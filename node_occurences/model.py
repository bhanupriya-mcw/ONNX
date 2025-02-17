import onnx
import onnx_graphsurgeon as gs
import argparse

def count_operations(onnx_model_path, op_type):
    model = onnx.load(onnx_model_path)  # Use function argument instead of hardcoded path
    graph = gs.import_onnx(model)
    
    op_counts = {}
    for node in graph.nodes:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
    
    return op_counts.get(op_type, 0)

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Count specific ONNX operations.")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model file")
    parser.add_argument("op_type", type=str, help="Operation type to count (e.g., Add, Sub, Mul, Div)")
    args = parser.parse_args()
    
    count = count_operations(args.model_path, args.op_type)
    print(f"Occurrences of {args.op_type}: {count}")
