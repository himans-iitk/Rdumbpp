# Fix GPU Detection Issue

## Problem
PyTorch is installed as CPU-only version, so GPU is not detected even though you're connected to RunPod with GPU.

## Solution

### Option 1: Run this command in your RunPod terminal (RECOMMENDED)

```bash
# Uninstall CPU version
pip uninstall torch torchvision -y

# Install CUDA-enabled PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is detected
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Option 2: Use the provided script

1. Copy the `install_pytorch_cuda.sh` file to your RunPod instance
2. Make it executable: `chmod +x install_pytorch_cuda.sh`
3. Run it: `./install_pytorch_cuda.sh`

### Option 3: Update the notebook cell

Replace Cell 5 in your notebook with this code:

```python
# Install dependencies with CUDA support for GPU
import sys
import subprocess

print("Installing dependencies with CUDA support for GPU...")
print("="*60)

# Uninstall existing packages
print("Step 1: Cleaning up existing installations...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "numpy", "-y", "-q"], 
              check=False, capture_output=True)
print("  ✓ Cleaned up existing installations")

# Install NumPy
print("\nStep 2: Installing NumPy < 2.0...")
subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2.0", "-q"], check=True, capture_output=True)
print("  ✓ NumPy < 2.0 installed")

# Install PyTorch with CUDA 11.8
print("\nStep 3: Installing PyTorch with CUDA 11.8...")
print("  CRITICAL: Installing CUDA-enabled version (NOT CPU)")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2",
     "--index-url", "https://download.pytorch.org/whl/cu118"],
    check=True,
    capture_output=True,
    text=True
)
print("  ✓ PyTorch 2.0.1 with CUDA 11.8 installed!")

# Install other dependencies
print("\nStep 4: Installing other dependencies...")
for package in ["webdataset>=0.2.0", "Pillow>=8.0.0"]:
    subprocess.run([sys.executable, "-m", "pip", "install", package, "-q"], check=True, capture_output=True)
    print(f"  ✓ {package} installed")

print("\n" + "="*60)
print("Installation completed! Please restart kernel and verify GPU.")
print("="*60)
```

## After Installation

1. **Restart the Jupyter kernel** (important!)
2. Run the verification cell to confirm GPU is detected
3. You should see: `CUDA available: True` and your GPU name

## Troubleshooting

If CUDA 11.8 doesn't work, try CUDA 12.1:
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
```

If you still have issues, check:
- Run `nvidia-smi` to confirm GPU is visible
- Check Python version: `python3 --version` (should be 3.8-3.10)
- Verify pip is up to date: `pip install --upgrade pip`


