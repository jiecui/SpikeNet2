# Grid Search Pipeline for Hard Mining Optimization

This pipeline automatically optimizes the `threshold` and `cluster_len` parameters in the hard mining process to find the combination that produces the lowest validation loss during training.

## Overview

The pipeline performs the following steps:
1. **Parameter Modification**: Updates `threshold` and `cluster_len` in `hard_mining_parallel.py`
2. **Hard Mining**: Runs the hard mining process with the new parameters
3. **Training**: Trains the model using `train_hard_model_lr.py` 
4. **Validation Loss Extraction**: Captures the best validation loss from training
5. **Result Tracking**: Records results and finds the optimal parameter combination

## Files Created

- `grid_search_pipeline.py`: Main pipeline script
- `run_grid_search.sh`: Convenient shell script to run the pipeline
- `README_grid_search.md`: This documentation file

## Usage

### Quick Start

```bash
# Run with default parameters
./run_grid_search.sh
```

### Custom Parameters

```bash
# Run with custom parameter ranges
./run_grid_search.sh --thresholds "0.8 0.75" --cluster-lens "10 8" --max-epochs 30
```

### Python Direct Usage

```bash
# Run directly with Python
python3 grid_search_pipeline.py --thresholds 0.8 0.75 0.5 0.4 --cluster-lens 10 8 5 3
```

## Parameters

### Default Grid Search Space
- **Thresholds**: `[0.8, 0.75, 0.5, 0.4]` (4 values)
- **Cluster Lengths**: `[10, 8, 5, 3]` (4 values)
- **Total combinations**: 16

### Training Parameters (optimized for grid search)
- **Max Epochs**: 50 (reduced from 300 for faster search)
- **Early Stopping Patience**: 5 (reduced from default)
- **Learning Rate Finder**: Skipped during grid search

## Command Line Options

### Shell Script (`run_grid_search.sh`)
```
--thresholds "0.8 0.75 0.5 0.4"    Threshold values to test
--cluster-lens "10 8 5 3"          Cluster length values to test  
--max-epochs 50                     Maximum epochs for training
--patience 5                        Early stopping patience
--output-dir grid_search_results    Output directory
--help, -h                          Show help message
```

### Python Script (`grid_search_pipeline.py`)
```
--thresholds 0.8 0.75 0.5 0.4       Threshold values (space-separated)
--cluster-lens 10 8 5 3             Cluster length values (space-separated)
--max-epochs 50                     Maximum epochs for training
--patience 5                        Early stopping patience
--output-dir grid_search_results    Output directory for results
```

## Output Files

The pipeline creates the following output files in the results directory:

### `grid_search_results.json`
Complete results in JSON format with detailed information for each combination tested.

### `grid_search_results.csv` 
Results in CSV format for easy analysis:
- `threshold`: Threshold value tested
- `cluster_len`: Cluster length value tested
- `hard_mining_success`: Whether hard mining completed successfully
- `training_success`: Whether training completed successfully  
- `val_loss`: Best validation loss achieved (if training succeeded)
- `runtime_seconds`: Time taken for this combination
- `timestamp`: When this combination was tested

### `best_parameters.json`
The optimal parameter combination with the lowest validation loss:
```json
{
  "threshold": 0.75,
  "cluster_len": 8,
  "val_loss": 0.123456,
  "runtime_seconds": 1234.5,
  "timestamp": "2025-10-11T..."
}
```

### `backups/`
Directory containing backups of original files before modification.

## Example Output

```
🚀 Grid Search Pipeline initialized
📊 Parameter space: 4 thresholds × 4 cluster_lens = 16 combinations
📁 Results will be saved to: grid_search_results

🧪 Testing combination: threshold=0.8, cluster_len=10
🔍 Running hard mining with threshold=0.8, cluster_len=10
✅ Hard mining completed successfully
🚄 Running training with threshold=0.8, cluster_len=10
✅ Training completed. Best val_loss: 0.145823

🏆 Best result so far:
   Threshold: 0.8
   Cluster Length: 10
   Validation Loss: 0.145823
   Runtime: 1245.3s

📊 Progress Summary:
   Completed: 1/16 combinations
   Successful: 1/1 completed combinations
   Current: 1/16
   Best Val Loss: 0.145823 (threshold=0.8, cluster_len=10)
```

## Requirements

- Python 3.7+
- All dependencies from `requirements.txt`
- CUDA-capable GPU (recommended)
- Sufficient disk space for model checkpoints and results

## Notes

### Performance Optimization
- Training epochs reduced to 50 for faster grid search
- Early stopping patience reduced to 5
- Learning rate finder skipped during grid search
- Automatic cleanup of temporary files

### Error Handling
- Robust error handling for individual combination failures
- Automatic backup and restoration of original files
- Intermediate results saved after each combination
- Graceful handling of interruptions (Ctrl+C)

### Resource Management
- Automatic cleanup of temporary files
- Memory management for large parameter spaces
- Timeout protection for stuck processes

## Interpreting Results

1. **Success Rate**: Check how many combinations completed successfully
2. **Validation Loss**: Lower values indicate better performance
3. **Runtime**: Consider both performance and computational cost
4. **Parameter Trends**: Look for patterns in successful parameter ranges

## Customization

To modify the parameter space or add new parameters:

1. Edit the default values in `grid_search_pipeline.py`
2. Add new parameter modification logic in `modify_hard_mining_parameters()`
3. Update the results tracking structure if needed

## Troubleshooting

### Common Issues
- **"No .npy files found"**: Hard mining may have failed
- **"Training timed out"**: Increase timeout or reduce max_epochs
- **"Permission denied"**: Make sure `run_grid_search.sh` is executable

### Debug Mode
Add `--verbose` flag or modify logging levels in the pipeline script for more detailed output.

## Future Enhancements

- Parallel execution of multiple combinations
- Bayesian optimization for smarter parameter search
- Real-time visualization of results
- Integration with hyperparameter optimization libraries (Optuna, etc.)