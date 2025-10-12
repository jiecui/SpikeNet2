"""
Grid Search Pipeline for Hard Mining and Training
Automated pipeline to optimize threshold and cluster_len parameters

This script performs a grid search over threshold and cluster_len parameters
in hard_mining_parallel.py and finds the combination that produces the best
(lowest) validation loss from train_hard_model_lr.py.

Author: Auto-generated pipeline
Date: 2025-10-11
"""

import os
import sys
import subprocess
import itertools
import json
import pandas as pd
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import re
import time


class GridSearchPipeline:
    """
    Automated grid search pipeline for hard mining and training optimization.
    """
    
    def __init__(self, 
                 threshold_values: List[float] = [0.8, 0.75, 0.5, 0.4],
                 cluster_len_values: List[int] = [10, 8, 5, 3],
                 max_epochs: int = 50,
                 patience: int = 5,
                 output_dir: str = "grid_search_results"):
        """
        Initialize the grid search pipeline.
        
        Args:
            threshold_values: List of threshold values to test
            cluster_len_values: List of cluster_len values to test
            max_epochs: Maximum epochs for training (reduced for grid search)
            patience: Early stopping patience
            output_dir: Directory to save results
        """
        self.threshold_values = threshold_values
        self.cluster_len_values = cluster_len_values
        self.max_epochs = max_epochs
        self.patience = patience
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results tracking
        self.results = []
        self.best_result = None
        
        # File paths
        self.hard_mining_file = "hard_mining_parallel.py"
        self.training_file = "train_hard_model_lr.py"
        self.backup_dir = os.path.join(self.output_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        print("🚀 Grid Search Pipeline initialized")
        print(f"📊 Parameter space: {len(threshold_values)} thresholds × {len(cluster_len_values)} cluster_lens = {len(threshold_values) * len(cluster_len_values)} combinations")
        print(f"📁 Results will be saved to: {self.output_dir}")
    
    def backup_original_files(self):
        """Create backups of original files before modification."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup hard_mining_parallel.py
        hard_mining_backup = os.path.join(self.backup_dir, f"hard_mining_parallel_backup_{timestamp}.py")
        shutil.copy2(self.hard_mining_file, hard_mining_backup)
        
        # Backup train_hard_model_lr.py 
        training_backup = os.path.join(self.backup_dir, f"train_hard_model_lr_backup_{timestamp}.py")
        shutil.copy2(self.training_file, training_backup)
        
        print(f"✅ Original files backed up to {self.backup_dir}")
        return hard_mining_backup, training_backup
    
    def restore_original_files(self, hard_mining_backup: str, training_backup: str):
        """Restore original files from backup."""
        shutil.copy2(hard_mining_backup, self.hard_mining_file)
        shutil.copy2(training_backup, self.training_file)
        print("✅ Original files restored")
    
    def modify_hard_mining_parameters(self, threshold: float, cluster_len: int):
        """
        Modify threshold and cluster_len parameters in hard_mining_parallel.py.
        
        Args:
            threshold: New threshold value
            cluster_len: New cluster_len value
        """
        # Read the file
        with open(self.hard_mining_file, 'r') as f:
            content = f.read()
        
        # Replace threshold value
        threshold_pattern = r'threshold = [0-9.]+\s*#.*'
        threshold_replacement = f'threshold = {threshold}  # Grid search value'
        content = re.sub(threshold_pattern, threshold_replacement, content)
        
        # Replace cluster_len value
        cluster_len_pattern = r'cluster_len = [0-9]+\s*#.*'
        cluster_len_replacement = f'cluster_len = {cluster_len}  # Grid search value'
        content = re.sub(cluster_len_pattern, cluster_len_replacement, content)
        
        # Write back to file
        with open(self.hard_mining_file, 'w') as f:
            f.write(content)
        
        print(f"📝 Updated parameters: threshold={threshold}, cluster_len={cluster_len}")
    
    def modify_training_parameters(self):
        """
        Modify training parameters for faster grid search.
        """
        # Read the file
        with open(self.training_file, 'r') as f:
            content = f.read()
        
        # Reduce max_epochs for faster training during grid search
        max_epochs_pattern = r'max_epochs=\d+'
        max_epochs_replacement = f'max_epochs={self.max_epochs}'
        content = re.sub(max_epochs_pattern, max_epochs_replacement, content)
        
        # Reduce patience for faster early stopping
        patience_pattern = r'EarlyStopping\(monitor="val_loss", patience=\d+\)'
        patience_replacement = f'EarlyStopping(monitor="val_loss", patience={self.patience})'
        content = re.sub(patience_pattern, patience_replacement, content)
        
        # Modify to save validation loss to file
        training_code_addition = '''
# Grid search modification: Save validation loss
best_val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
if hasattr(trainer.checkpoint_callback, 'best_model_score'):
    best_val_loss = trainer.checkpoint_callback.best_model_score
    
# Save validation loss to file for grid search
grid_search_results_file = "current_val_loss.txt"
with open(grid_search_results_file, 'w') as f:
    f.write(str(float(best_val_loss)))
print(f"💾 Best validation loss saved: {best_val_loss}")
'''
        
        # Insert before wandb.finish()
        content = content.replace('    wandb.finish()', f'    {training_code_addition}\n    wandb.finish()')
        
        # Write back to file
        with open(self.training_file, 'w') as f:
            f.write(content)
        
        print(f"📝 Training parameters updated: max_epochs={self.max_epochs}, patience={self.patience}")
    
    def run_hard_mining(self, threshold: float, cluster_len: int) -> bool:
        """
        Run hard mining process with given parameters.
        
        Args:
            threshold: Threshold value
            cluster_len: Cluster length value
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🔍 Running hard mining with threshold={threshold}, cluster_len={cluster_len}")
        
        try:
            # Run hard mining
            result = subprocess.run(
                [sys.executable, self.hard_mining_file],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print("✅ Hard mining completed successfully")
                return True
            else:
                print(f"❌ Hard mining failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Hard mining timed out")
            return False
        except Exception as e:
            print(f"❌ Hard mining failed with exception: {e}")
            return False
    
    def run_training(self, threshold: float, cluster_len: int) -> Optional[float]:
        """
        Run training process and extract validation loss.
        
        Args:
            threshold: Threshold value used for this run
            cluster_len: Cluster length value used for this run
            
        Returns:
            float: Best validation loss, or None if failed
        """
        print(f"🚄 Running training with threshold={threshold}, cluster_len={cluster_len}")
        
        try:
            # Remove previous validation loss file if exists
            val_loss_file = "current_val_loss.txt"
            if os.path.exists(val_loss_file):
                os.remove(val_loss_file)
            
            # Run training with LR finder skipped for faster grid search
            result = subprocess.run(
                [sys.executable, self.training_file, "--skip-lr-finder"],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                # Try to read validation loss from file
                if os.path.exists(val_loss_file):
                    with open(val_loss_file, 'r') as f:
                        val_loss = float(f.read().strip())
                    os.remove(val_loss_file)  # Clean up
                    print(f"✅ Training completed. Best val_loss: {val_loss:.6f}")
                    return val_loss
                else:
                    print("⚠️ Training completed but no validation loss file found")
                    return None
            else:
                print(f"❌ Training failed with return code {result.returncode}")
                print(f"Error output: {result.stderr[:1000]}...")  # Truncate long error messages
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰ Training timed out")
            return None
        except Exception as e:
            print(f"❌ Training failed with exception: {e}")
            return None
    
    def run_single_combination(self, threshold: float, cluster_len: int) -> Dict:
        """
        Run complete pipeline for a single parameter combination.
        
        Args:
            threshold: Threshold value
            cluster_len: Cluster length value
            
        Returns:
            dict: Results dictionary
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🧪 Testing combination: threshold={threshold}, cluster_len={cluster_len}")
        print(f"{'='*60}")
        
        result = {
            'threshold': threshold,
            'cluster_len': cluster_len,
            'hard_mining_success': False,
            'training_success': False,
            'val_loss': None,
            'runtime_seconds': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Modify parameters
            self.modify_hard_mining_parameters(threshold, cluster_len)
            
            # Run hard mining
            hard_mining_success = self.run_hard_mining(threshold, cluster_len)
            result['hard_mining_success'] = hard_mining_success
            
            if hard_mining_success:
                # Run training
                val_loss = self.run_training(threshold, cluster_len)
                result['training_success'] = val_loss is not None
                result['val_loss'] = val_loss
            else:
                print("⏭️ Skipping training due to hard mining failure")
            
        except Exception as e:
            print(f"❌ Exception during combination execution: {e}")
        
        finally:
            result['runtime_seconds'] = time.time() - start_time
            print(f"⏱️ Combination completed in {result['runtime_seconds']:.1f} seconds")
            
            # Clean up any temporary files
            self.cleanup_temp_files()
        
        return result
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during the process."""
        temp_files = [
            "current_val_loss.txt",
            "optimal_lr.pkl"
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass  # Ignore cleanup errors
    
    def save_results(self):
        """Save current results to JSON and CSV files."""
        # Save to JSON
        json_file = os.path.join(self.output_dir, "grid_search_results.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save to CSV
        csv_file = os.path.join(self.output_dir, "grid_search_results.csv")
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)
        
        print(f"💾 Results saved to {json_file} and {csv_file}")
    
    def find_best_result(self):
        """Find and update the best result based on validation loss."""
        successful_results = [r for r in self.results if r['training_success'] and r['val_loss'] is not None]
        
        if successful_results:
            self.best_result = min(successful_results, key=lambda x: x['val_loss'])
            print("\n🏆 Best result so far:")
            print(f"   Threshold: {self.best_result['threshold']}")
            print(f"   Cluster Length: {self.best_result['cluster_len']}")
            print(f"   Validation Loss: {self.best_result['val_loss']:.6f}")
            print(f"   Runtime: {self.best_result['runtime_seconds']:.1f}s")
        else:
            print("\n⚠️ No successful results yet")
    
    def print_progress_summary(self, current_idx: int, total: int):
        """Print progress summary."""
        completed = len(self.results)
        successful = len([r for r in self.results if r['training_success']])
        
        print("\n📊 Progress Summary:")
        print(f"   Completed: {completed}/{total} combinations")
        print(f"   Successful: {successful}/{completed} completed combinations")
        print(f"   Current: {current_idx + 1}/{total}")
        
        if self.best_result:
            print(f"   Best Val Loss: {self.best_result['val_loss']:.6f} "
                  f"(threshold={self.best_result['threshold']}, cluster_len={self.best_result['cluster_len']})")
    
    def run_grid_search(self) -> Optional[Dict]:
        """
        Run complete grid search over all parameter combinations.
        
        Returns:
            dict: Best result found
        """
        print("\n🚀 Starting Grid Search Pipeline")
        print(f"🔍 Testing {len(self.threshold_values)} threshold values: {self.threshold_values}")
        print(f"🔍 Testing {len(self.cluster_len_values)} cluster_len values: {self.cluster_len_values}")
        
        # Create parameter combinations
        combinations = list(itertools.product(self.threshold_values, self.cluster_len_values))
        total_combinations = len(combinations)
        
        print(f"📊 Total combinations to test: {total_combinations}")
        
        # Backup original files
        hard_mining_backup, training_backup = self.backup_original_files()
        
        # Modify training file for grid search
        self.modify_training_parameters()
        
        try:
            # Run grid search
            for idx, (threshold, cluster_len) in enumerate(combinations):
                try:
                    # Run single combination
                    result = self.run_single_combination(threshold, cluster_len)
                    self.results.append(result)
                    
                    # Update best result
                    self.find_best_result()
                    
                    # Print progress
                    self.print_progress_summary(idx, total_combinations)
                    
                    # Save intermediate results
                    self.save_results()
                    
                except KeyboardInterrupt:
                    print("\n🛑 Grid search interrupted by user")
                    break
                except Exception as e:
                    print(f"❌ Unexpected error processing combination: {e}")
                    # Continue with next combination
                    continue
            
        finally:
            # Always restore original files
            self.restore_original_files(hard_mining_backup, training_backup)
            
            # Final cleanup
            self.cleanup_temp_files()
        
        # Final results
        self.save_results()
        self.print_final_summary()
        
        return self.best_result
    
    def print_final_summary(self):
        """Print final summary of grid search results."""
        successful_results = [r for r in self.results if r['training_success'] and r['val_loss'] is not None]
        
        print(f"\n{'='*80}")
        print("🎯 GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"📊 Total combinations tested: {len(self.results)}")
        print(f"✅ Successful combinations: {len(successful_results)}")
        print(f"❌ Failed combinations: {len(self.results) - len(successful_results)}")
        
        if successful_results and self.best_result:
            print("\n🏆 BEST RESULT:")
            print(f"   Threshold: {self.best_result['threshold']}")
            print(f"   Cluster Length: {self.best_result['cluster_len']}")
            print(f"   Validation Loss: {self.best_result['val_loss']:.6f}")
            print(f"   Runtime: {self.best_result['runtime_seconds']:.1f} seconds")
            
            # Top 3 results
            sorted_results = sorted(successful_results, key=lambda x: x['val_loss'])
            print("\n🥇 TOP 3 RESULTS:")
            for i, result in enumerate(sorted_results[:3], 1):
                print(f"   {i}. threshold={result['threshold']}, cluster_len={result['cluster_len']}, "
                      f"val_loss={result['val_loss']:.6f}")
            
            # Save best parameters to a separate file
            best_params_file = os.path.join(self.output_dir, "best_parameters.json")
            with open(best_params_file, 'w') as f:
                json.dump(self.best_result, f, indent=2)
            print(f"\n💾 Best parameters saved to: {best_params_file}")
            
        else:
            print("\n❌ No successful combinations found!")
        
        print(f"\n📁 All results saved to: {self.output_dir}")
        print(f"{'='*80}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Grid Search Pipeline for Hard Mining Optimization")
    
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.8, 0.75, 0.5, 0.4],
        help="Threshold values to test (default: 0.8 0.75 0.5 0.4)"
    )
    
    parser.add_argument(
        "--cluster-lens",
        nargs="+",
        type=int,
        default=[10, 8, 5, 3],
        help="Cluster length values to test (default: 10 8 5 3)"
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum epochs for training during grid search (default: 50)"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="grid_search_results",
        help="Output directory for results (default: grid_search_results)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize and run grid search
    pipeline = GridSearchPipeline(
        threshold_values=args.thresholds,
        cluster_len_values=args.cluster_lens,
        max_epochs=args.max_epochs,
        patience=args.patience,
        output_dir=args.output_dir
    )
    
    # Run grid search
    best_result = pipeline.run_grid_search()
    
    # Exit with appropriate code
    if best_result is not None:
        print("✅ Grid search completed successfully!")
        return 0
    else:
        print("❌ Grid search failed to find any successful results!")
        return 1


if __name__ == "__main__":
    sys.exit(main())