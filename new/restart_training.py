"""
Script to clear GPU memory and restart training
"""

import torch
import gc
import os
import subprocess
import sys

def clear_gpu_memory():
    """Clear GPU memory completely"""
    print("🧹 Clearing GPU memory...")
    
    if torch.cuda.is_available():
        # Clear all cached memory
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Reset CUDA state
        torch.cuda.reset_peak_memory_stats()
        
        # Check memory status
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        
        print(f"GPU Memory Status:")
        print(f"  Total: {total_memory:.1f} GB")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Cached: {cached:.1f} GB")
        print(f"  Free: {total_memory - allocated:.1f} GB")
        
    else:
        print("CUDA not available")

def restart_training():
    """Restart training with memory optimizations"""
    print("🚀 Restarting training with memory optimizations...")
    
    # Set environment variables for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Clear memory first
    clear_gpu_memory()
    
    print("\n" + "="*60)
    print("MEMORY OPTIMIZED TRAINING CONFIGURATION:")
    print("- Batch size reduced to 4 (from 16)")
    print("- Gradient checkpointing enabled")
    print("- Memory fraction limited to 90%")
    print("- Pin memory disabled")
    print("- Periodic cache clearing")
    print("="*60)
    
    # Start training
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return False
    
    return True

if __name__ == "__main__":
    print("TrOCR Training - Memory Optimized Restart")
    print("="*50)
    
    success = restart_training()
    
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Try reducing batch_size further (to 2 or 1)")
        print("2. Close other GPU applications")
        print("3. Restart your system to clear all GPU memory")
