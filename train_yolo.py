import os
import sys
import subprocess

# # --- MPS / Env Setup ---
# user = os.getenv('USER')
# os.environ['CUDA_MPS_PIPE_DIRECTORY'] = f'/tmp/nvidia-mps-{user}'
# os.environ['CUDA_MPS_LOG_DIRECTORY'] = f'/tmp/nvidia-log-{user}'
# subprocess.run(['mkdir', '-p', os.environ['CUDA_MPS_PIPE_DIRECTORY']])
# subprocess.run(['mkdir', '-p', os.environ['CUDA_MPS_LOG_DIRECTORY']])

import torch
from ultralytics import YOLO

def train(epochs=100, batch_size=32, device='mps'):
    print(f"--> PyTorch Version: {torch.__version__}")

    # 1. Load Standard YOLO Model (Not OBB)
    # Using 'yolo26n.pt' automatically sets the task to 'detect'
    print("--> Loading Model (YOLO26 Standard Detection)...")
    model = YOLO('yolo26n.pt')

    print("--> Starting Training...")
    model.train(
        data='dataset.yaml', # Update this filename if you change your yaml
        epochs=epochs, # change to other number of epochs 
        imgsz=960,
        batch=batch_size, 
        device=device, # Use GPU 0; change to 'cpu' if you want to train on CPU 'mps' for Apple Silicon
        project='/Users/xuzichun/Desktop/project/results',
        name=f'run_{epochs}_{batch_size}', # Changed run name
        # Performance Settings
        workers=3,
        amp=True,
        exist_ok=True,
        # dropout = 0.2
    )

if __name__ == '__main__':
    train()