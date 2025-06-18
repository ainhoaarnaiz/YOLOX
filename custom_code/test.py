#!/usr/bin/env python3

import os
import sys
import subprocess

YOLOX_PATH = r"C:\Users\aarnaizl\Documents\YOLOX"

def clear_cache_and_train():
    """
    Clear Python cache and start training.
    """
    print("=== Cache-Safe YOLOX Training ===")
    print("[INFO] This script will start a fresh Python process")
    print("[INFO] This ensures all code changes are loaded")
    print()
    
    # Create the training command
    train_script = os.path.join(YOLOX_PATH, "tools", "train.py")
    exp_file = os.path.join(YOLOX_PATH, "exps", "default", "yolox_m.py")
    
    cmd = [
        "python", train_script,
        "-f", exp_file,
        "-d", "1",
        "-b", "8",
        "--fp16"
    ]
    
    print("[INFO] Starting fresh training process:")
    print(" ".join(cmd))
    print("-" * 50)
    
    try:
        # Start completely fresh Python process
        result = subprocess.run(cmd, cwd=YOLOX_PATH)
        
        print("-" * 50)
        print(f"[INFO] Training completed with exit code: {result.returncode}")
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        return False
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return False

if __name__ == "__main__":
    clear_cache_and_train()
