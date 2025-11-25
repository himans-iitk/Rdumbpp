# How to Connect to RunPod GPU from Your Mac

## Important Understanding

**You DON'T need CUDA on your Mac!** 

- ❌ Macs don't have NVIDIA GPUs (no CUDA support)
- ✅ RunPod has the GPU and CUDA already installed
- ✅ You connect to RunPod from your Mac to use their GPU

## Method 1: SSH Port Forwarding (Recommended)

This lets you run Jupyter on RunPod but access it from your Mac browser.

### Step 1: SSH into RunPod
```bash
ssh -i ~/.ssh/runpod_key eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io
```

### Step 2: On RunPod, install Jupyter (if not already installed)
```bash
pip install jupyter notebook
```

### Step 3: Start Jupyter on RunPod
```bash
# Navigate to your project directory
cd /path/to/your/project

# Start Jupyter (replace with your actual path)
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

You'll see output like:
```
[I 2024-01-01 12:00:00.000 NotebookApp] http://0.0.0.0:8888/?token=abc123xyz...
```

**Keep this terminal open!**

### Step 4: On Your Mac, Create SSH Tunnel

Open a **NEW terminal window on your Mac** (keep the RunPod SSH session running):

```bash
ssh -L 8888:localhost:8888 -i ~/.ssh/runpod_key eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io
```

This creates a tunnel that forwards port 8888 from RunPod to your Mac.

**Keep this terminal open too!**

### Step 5: Access Jupyter from Your Mac Browser

1. Open your browser on your Mac
2. Go to: `http://localhost:8888`
3. You'll be asked for a token - copy it from Step 3 output
4. You're now running Jupyter on RunPod, but viewing it on your Mac!

## Method 2: Use RunPod's Built-in Jupyter (Easiest)

Many RunPod instances have Jupyter pre-installed:

1. Go to RunPod's web dashboard
2. Look for "Jupyter" or "Notebook" link/button
3. Click it - it will open Jupyter in a new tab
4. Upload your notebook there

## Method 3: VS Code Remote SSH (Advanced)

If you use VS Code:

1. Install "Remote - SSH" extension
2. Add RunPod to your SSH config:
   ```
   Host runpod
       HostName ssh.runpod.io
       User eqdlc2mhm8ogbt-64411dd7
       IdentityFile ~/.ssh/runpod_key
   ```
3. Connect via VS Code's remote SSH feature
4. Open your project folder on RunPod
5. Install Jupyter extension in VS Code
6. Run notebooks directly in VS Code, but they execute on RunPod

## Method 4: Run Commands Directly via SSH

You can also just run Python scripts directly on RunPod:

```bash
# SSH into RunPod
ssh -i ~/.ssh/runpod_key eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io

# On RunPod, run your script
python your_script.py
```

## Troubleshooting

### Port 8888 already in use?
Use a different port:
```bash
# On RunPod
jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser --allow-root

# On Mac (use port 8889)
ssh -L 8889:localhost:8889 -i ~/.ssh/runpod_key eqdlc2mhm8ogbt-64411dd7@ssh.runpod.io
```

### Can't access localhost:8888?
- Make sure the SSH tunnel terminal is still running
- Check that Jupyter is running on RunPod
- Try `http://127.0.0.1:8888` instead

### Jupyter not starting?
- Make sure you're in the right directory
- Check if port is already in use: `lsof -i :8888`
- Try `--allow-root` flag if you're root user

## Summary

**You don't need CUDA on Mac!** Just:
1. Run Jupyter on RunPod (where GPU is)
2. Access it from your Mac via SSH tunnel
3. Your notebooks run on RunPod's GPU, but you edit them on your Mac


