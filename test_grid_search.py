#!/usr/bin/env python3
"""
Test script for the grid search pipeline
Quick verification that the pipeline components work correctly
"""

import os
import sys
import shutil
import subprocess

def test_pipeline_components():
    """Test basic pipeline components without running the full grid search."""
    
    print("🧪 Testing Grid Search Pipeline Components")
    print("=" * 50)
    
    # Test 1: Check if required files exist
    print("1. Checking required files...")
    required_files = [
        "grid_search_pipeline.py",
        "hard_mining_parallel.py", 
        "train_hard_model_lr.py",
        "controlset.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False
    
    # Test 2: Check if Python imports work
    print("\n2. Testing Python imports...")
    try:
        from grid_search_pipeline import GridSearchPipeline
        print("   ✅ grid_search_pipeline imports successfully")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Test 3: Test parameter modification
    print("\n3. Testing parameter modification...")
    temp_file = None
    try:
        # Create temporary backup
        temp_file = "hard_mining_parallel_test_backup.py"
        shutil.copy2("hard_mining_parallel.py", temp_file)
        
        # Test parameter modification
        pipeline = GridSearchPipeline(
            threshold_values=[0.8],
            cluster_len_values=[10],
            output_dir="test_output"
        )
        
        pipeline.modify_hard_mining_parameters(0.75, 8)
        
        # Check if modification worked
        with open("hard_mining_parallel.py", 'r') as f:
            modified_content = f.read()
        
        if "threshold = 0.75" in modified_content and "cluster_len = 8" in modified_content:
            print("   ✅ Parameter modification works")
        else:
            print("   ❌ Parameter modification failed")
            return False
        
        # Restore original file
        shutil.copy2(temp_file, "hard_mining_parallel.py")
        os.remove(temp_file)
        
    except Exception as e:
        print(f"   ❌ Parameter modification error: {e}")
        # Try to restore file
        if temp_file and os.path.exists(temp_file):
            try:
                shutil.copy2(temp_file, "hard_mining_parallel.py")
                os.remove(temp_file)
            except Exception:
                pass
        return False
    
    # Test 4: Check shell script
    print("\n4. Testing shell script...")
    if os.path.exists("run_grid_search.sh"):
        if os.access("run_grid_search.sh", os.X_OK):
            print("   ✅ run_grid_search.sh is executable")
        else:
            print("   ⚠️  run_grid_search.sh exists but not executable")
            print("      Run: chmod +x run_grid_search.sh")
    else:
        print("   ❌ run_grid_search.sh not found")
        return False
    
    # Test 5: Test help functions
    print("\n5. Testing help functions...")
    try:
        result = subprocess.run([
            sys.executable, "grid_search_pipeline.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "Grid Search Pipeline" in result.stdout:
            print("   ✅ Help function works")
        else:
            print("   ⚠️  Help function has issues")
    except Exception as e:
        print(f"   ⚠️  Help test error: {e}")
    
    print("\n✅ All basic tests passed!")
    print("\n📝 Next steps:")
    print("   1. Run a small test: ./run_grid_search.sh --thresholds \"0.8\" --cluster-lens \"10\" --max-epochs 5")
    print("   2. Check the test results in grid_search_results/")
    print("   3. Run full grid search: ./run_grid_search.sh")
    
    return True

def test_small_grid_search():
    """Run a minimal grid search for testing."""
    print("\n🔬 Running minimal grid search test...")
    
    try:
        result = subprocess.run([
            sys.executable, "grid_search_pipeline.py",
            "--thresholds", "0.8",
            "--cluster-lens", "10", 
            "--max-epochs", "5",
            "--output-dir", "test_grid_search"
        ], timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Minimal grid search test completed successfully!")
            
            # Check if results were created
            if os.path.exists("test_grid_search/grid_search_results.csv"):
                print("✅ Results file created")
                
                # Show results
                try:
                    import pandas as pd
                    df = pd.read_csv("test_grid_search/grid_search_results.csv")
                    print(f"📊 Test results: {len(df)} combinations tested")
                    if len(df) > 0:
                        print(f"    Training success: {df['training_success'].sum()}/{len(df)}")
                        if df['training_success'].any():
                            best_val_loss = df[df['training_success']]['val_loss'].min()
                            print(f"    Best val_loss: {best_val_loss:.6f}")
                except Exception as e:
                    print(f"⚠️  Could not analyze results: {e}")
            
            return True
        else:
            print(f"❌ Minimal grid search failed with code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Minimal grid search timed out")
        return False
    except Exception as e:
        print(f"❌ Minimal grid search error: {e}")
        return False

if __name__ == "__main__":
    print("Grid Search Pipeline Test Suite")
    print("=" * 40)
    
    # Parse arguments
    run_minimal = "--minimal" in sys.argv
    
    # Run basic component tests
    success = test_pipeline_components()
    
    if success and run_minimal:
        print("\n" + "=" * 40)
        success = test_small_grid_search()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)