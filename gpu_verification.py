#!/usr/bin/env python3
"""
GPU Usage Verification Script for SpikeNet2

This script helps verify that all GPUs, including GPU 0, are being used correctly
in your parallel processing setup.
"""

import torch
import time
import os
from threading import Thread

def check_gpu_usage():
    """Check current GPU usage and availability."""
    print("🔍 GPU System Check")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} CUDA-capable GPU(s)")
    
    for i in range(num_gpus):
        try:
            # Set device and get info
            torch.cuda.set_device(i)
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            
            # Check memory
            memory_total = gpu_props.total_memory / 1e9
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            
            print(f"\n📱 GPU {i}: {gpu_name}")
            print(f"   Total Memory: {memory_total:.2f} GB")
            print(f"   Allocated: {memory_allocated:.2f} GB")
            print(f"   Reserved: {memory_reserved:.2f} GB")
            print(f"   Free: {(memory_total - memory_reserved):.2f} GB")
            
            # Test basic operation
            try:
                test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
                result = torch.mm(test_tensor, test_tensor)
                print(f"   ✅ Basic operations: Working")
                del test_tensor, result
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"   ❌ Basic operations: Failed - {e}")
                
        except Exception as e:
            print(f"❌ GPU {i}: Error - {e}")

def test_multi_gpu_assignment():
    """Test that we can assign work to different GPUs."""
    print("\n🎯 Multi-GPU Assignment Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for testing")
        return
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"⚠️  Only {num_gpus} GPU available - cannot test multi-GPU")
        return
    
    def gpu_worker(gpu_id):
        """Worker function that uses a specific GPU."""
        try:
            # Set the device
            torch.cuda.set_device(gpu_id)
            current_device = torch.cuda.current_device()
            
            print(f"🔧 Thread for GPU {gpu_id}: Using device {current_device}")
            
            # Create tensor on this GPU
            device = f'cuda:{gpu_id}'
            tensor = torch.randn(1000, 1000, device=device)
            
            # Do some computation
            for _ in range(10):
                tensor = torch.mm(tensor, tensor.t())
                time.sleep(0.1)  # Simulate work
            
            print(f"✅ GPU {gpu_id}: Completed test computation")
            
            # Clean up
            del tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ GPU {gpu_id}: Error - {e}")
    
    # Start workers for each GPU
    threads = []
    for gpu_id in range(num_gpus):
        thread = Thread(target=gpu_worker, args=(gpu_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    print("🏁 Multi-GPU test completed")

def check_pytorch_lightning_gpu():
    """Test PyTorch Lightning GPU usage."""
    print("\n⚡ PyTorch Lightning GPU Test")
    print("=" * 50)
    
    try:
        import pytorch_lightning as pl
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available for PyTorch Lightning test")
            return
        
        num_gpus = torch.cuda.device_count()
        
        for gpu_id in range(num_gpus):  # Test all GPUs
            print(f"\n🧪 Testing PyTorch Lightning with GPU {gpu_id}")
            
            try:
                # Create a simple trainer
                trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=[gpu_id],
                    fast_dev_run=True,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    logger=False,
                )
                
                print(f"✅ GPU {gpu_id}: PyTorch Lightning trainer created successfully")
                print(f"   Trainer device: {trainer.device_ids}")
                
            except Exception as e:
                print(f"❌ GPU {gpu_id}: PyTorch Lightning failed - {e}")
                
    except ImportError:
        print("❌ PyTorch Lightning not available")

def main():
    """Run all GPU verification tests."""
    print("🚀 GPU Verification")
    print("=" * 60)
    
    # Basic GPU check
    check_gpu_usage()
    
    # Multi-GPU assignment test
    test_multi_gpu_assignment()
    
    # PyTorch Lightning test
    check_pytorch_lightning_gpu()
    
    print("\n" + "=" * 60)
    print("📋 Recommendations:")
    print("1. Check nvidia-smi during parallel processing to see actual GPU usage")
    print("2. Look for any CUDA_VISIBLE_DEVICES environment variable restrictions")
    print("3. Ensure no other processes are blocking GPU 0")
    
    # Check environment variables
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible:
        print(f"⚠️  CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")
    else:
        print("✅ CUDA_VISIBLE_DEVICES is not set (all GPUs should be visible)")

if __name__ == "__main__":
    main()