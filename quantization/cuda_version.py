import torch
import transformers
import datasets
import onnx
import onnxruntime as ort
import numpy as np
import platform
import sys

def check_system_and_versions():
    print("=== System Information ===")
    print(f"Python Version: {sys.version}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    
    print("\n=== Library Versions ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Datasets: {datasets.__version__}")
    print(f"ONNX: {onnx.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"NumPy: {np.__version__}")
    
    print("\n=== CUDA Information ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("No CUDA-capable GPU detected.")

    print("\n=== ONNX Runtime GPU Support ===")
    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    if "CUDAExecutionProvider" in providers:
        print("ONNX Runtime is configured with CUDA support.")
    else:
        print("ONNX Runtime is not using CUDA.")

if __name__ == "__main__":
    check_system_and_versions()