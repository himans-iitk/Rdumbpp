#!/bin/bash
# Install PyTorch with CUDA support for RunPod
# Run this script in your RunPod terminal

echo "Installing PyTorch with CUDA support..."
echo "=========================================="

# Uninstall existing PyTorch
pip uninstall torch torchvision -y

# Install NumPy < 2.0 first
pip install "numpy<2.0"

# Try CUDA 11.8 first (most compatible)
echo "Attempting CUDA 11.8 installation..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=========================================="
echo "Installation complete!"


