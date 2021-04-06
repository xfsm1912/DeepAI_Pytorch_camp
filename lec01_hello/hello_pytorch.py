import torch

print(f"Hello World, Hello PyTorch {torch.__version__}.")

cuda_available = torch.cuda.is_available()
print(f"\nCUDA is available:{cuda_available}, version is {torch.version.cuda}")

if cuda_available:
    print(f"\ndevice_name: {torch.cuda.get_device_name(0)}")
else:
    print("We don't have cuda in this device.")
