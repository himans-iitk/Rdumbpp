# Run Jupyter Notebooks in Cursor IDE with RunPod GPU

This guide shows how to run notebooks in Cursor IDE on your Mac, but execute them on RunPod's GPU.

## Method 1: Remote Jupyter Kernel (Recommended)

This lets you edit notebooks in Cursor, but execution happens on RunPod.

### Step 1: Setup Jupyter Kernel on RunPod

SSH into RunPod:
```bash
ssh -i ~/.ssh/runpod_key eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io
```

On RunPod, install Jupyter and create a kernel:
```bash
# Install Jupyter
pip install jupyter ipykernel

# Create a named kernel (replace 'gpu-env' with your preferred name)
python -m ipykernel install --user --name gpu-env --display-name "Python (RunPod GPU)"

# Start Jupyter kernel server
python -m jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0 --KernelGatewayApp.port=8888
```

**Keep this running!**

### Step 2: Configure Cursor to Use Remote Kernel

1. **Install Jupyter extension in Cursor** (if not already installed)
   - Open Extensions (Cmd+Shift+X)
   - Search for "Jupyter" and install

2. **Create SSH tunnel from your Mac:**
   ```bash
   # In a new terminal on your Mac
   ssh -L 8888:localhost:8888 -i ~/.ssh/runpod_key eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io
   ```

3. **In Cursor, open your notebook and select kernel:**
   - Open your `.ipynb` file
   - Click on kernel selector (top right)
   - Choose "Select Another Kernel"
   - Enter: `http://localhost:8888/api/kernels`

### Step 3: Alternative - Use SSH Remote Execution

If kernel gateway doesn't work, use direct SSH execution.

## Method 2: SSH Remote Execution (Simpler)

This runs code on RunPod via SSH when you execute cells.

### Setup Script for Cursor

Create a Python script that executes code on RunPod:

```python
# remote_executor.py
import subprocess
import sys
import json

def execute_on_runpod(code):
    """Execute Python code on RunPod via SSH"""
    ssh_cmd = [
        "ssh", "-i", "~/.ssh/runpod_key",
        "eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io",
        f"python3 -c '{code}'"
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    return result.stdout, result.stderr
```

But this is complex. Better to use Method 1 or Method 3.

## Method 3: VS Code Remote SSH (Best for Cursor)

Since Cursor is based on VS Code, you can use Remote SSH extension.

### Step 1: Install Remote SSH Extension

1. In Cursor, open Extensions (Cmd+Shift+X)
2. Search for "Remote - SSH"
3. Install it

### Step 2: Configure SSH

1. Press `Cmd+Shift+P` (Command Palette)
2. Type "Remote-SSH: Open SSH Configuration File"
3. Add RunPod config:
   ```
   Host runpod
       HostName ssh.runpod.io
       User eqdlc2mhm8ogbt-64411dd7
       IdentityFile ~/.ssh/runpod_key
   ```

### Step 3: Connect to RunPod

1. Press `Cmd+Shift+P`
2. Type "Remote-SSH: Connect to Host"
3. Select "runpod"
4. Cursor will open a new window connected to RunPod
5. Open your project folder on RunPod
6. Install Python and Jupyter extensions in the remote window
7. Open your notebook - it will run on RunPod's GPU!

## Method 4: Use Jupyter Server (Easiest Setup)

### On RunPod:
```bash
# SSH into RunPod
ssh -i ~/.ssh/runpod_key eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io

# Install Jupyter
pip install jupyter

# Start Jupyter server
jupyter server --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### On Your Mac:
```bash
# Create SSH tunnel
ssh -L 8888:localhost:8888 -i ~/.ssh/runpod_key eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io
```

### In Cursor:
1. Open Command Palette (Cmd+Shift+P)
2. Type "Jupyter: Specify Jupyter Server for Connections"
3. Enter: `http://localhost:8888`
4. Open your notebook - it will use RunPod's Jupyter server

## Recommended: Method 3 (Remote SSH)

This is the cleanest solution:
- ✅ Edit files in Cursor on Mac
- ✅ All execution happens on RunPod
- ✅ Full GPU access
- ✅ No port forwarding needed
- ✅ Works seamlessly

Try Method 3 first - it's the most integrated solution!


