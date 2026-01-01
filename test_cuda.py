import torch
import numpy as np

print("="*50)
print("PyTorch CUDA Test")
print("="*50)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("CUDA not available, using CPU")
    device = torch.device("cpu")

print(f"\nDevice: {device}")

# Run a simple test
print("\n" + "="*50)
print("Running Simple Matrix Multiplication Test")
print("="*50)

# Create two random matrices
A = torch.randn(1000, 1000).to(device)
B = torch.randn(1000, 1000).to(device)

print(f"\nMatrix A shape: {A.shape}, device: {A.device}")
print(f"Matrix B shape: {B.shape}, device: {B.device}")

# Multiply
import time
start = time.time()
C = torch.matmul(A, B)
if cuda_available:
    torch.cuda.synchronize()
end = time.time()

print(f"\nMatrix multiplication completed!")
print(f"Result shape: {C.shape}")
print(f"Time taken: {(end-start)*1000:.2f} ms")
print(f"\nSample values from result:\n{C[:3, :3]}")

print("\n" + "="*50)
print("Test completed successfully!")
print("="*50)
