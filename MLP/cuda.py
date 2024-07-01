import torch
import sys

print(f"Python executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"Is cuDnn available: {torch.backends.cudnn.enabled}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

