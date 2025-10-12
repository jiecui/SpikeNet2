#!/bin/bash

# Grid Search Runner Script
# Automated script to run the grid search pipeline for hard mining optimization

echo "🚀 Starting Grid Search Pipeline for Hard Mining Optimization"
echo "============================================================="

# Default parameters
THRESHOLDS="0.8 0.75 0.5 0.4"
CLUSTER_LENS="10 8 5 3"
MAX_EPOCHS=50
PATIENCE=5
OUTPUT_DIR="grid_search_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --thresholds)
            THRESHOLDS="$2"
            shift 2
            ;;
        --cluster-lens)
            CLUSTER_LENS="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --thresholds \"0.8 0.75 0.5 0.4\"    Threshold values to test"
            echo "  --cluster-lens \"10 8 5 3\"          Cluster length values to test"
            echo "  --max-epochs 50                     Maximum epochs for training"
            echo "  --patience 5                        Early stopping patience"
            echo "  --output-dir grid_search_results    Output directory"
            echo "  --help, -h                          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --thresholds \"0.8 0.75\" --cluster-lens \"10 8\" --max-epochs 30"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo "Configuration:"
echo "  Thresholds: $THRESHOLDS"
echo "  Cluster lengths: $CLUSTER_LENS"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Patience: $PATIENCE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if required files exist
if [[ ! -f "hard_mining_parallel.py" ]]; then
    echo "❌ Error: hard_mining_parallel.py not found in current directory"
    exit 1
fi

if [[ ! -f "train_hard_model_lr.py" ]]; then
    echo "❌ Error: train_hard_model_lr.py not found in current directory"
    exit 1
fi

if [[ ! -f "grid_search_pipeline.py" ]]; then
    echo "❌ Error: grid_search_pipeline.py not found in current directory"
    exit 1
fi

# Check if required data files exist
if [[ ! -f "controlset.csv" ]]; then
    echo "❌ Error: controlset.csv not found in current directory"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Log start time
START_TIME=$(date)
echo "🕒 Started at: $START_TIME"
echo ""

# Run the grid search pipeline
echo "🔍 Running grid search pipeline..."
python3 grid_search_pipeline.py \
    --thresholds $THRESHOLDS \
    --cluster-lens $CLUSTER_LENS \
    --max-epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --output-dir "$OUTPUT_DIR"

# Capture exit code
EXIT_CODE=$?

# Log end time
END_TIME=$(date)
echo ""
echo "🕒 Completed at: $END_TIME"

# Show results
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Grid search completed successfully!"
    echo ""
    echo "📊 Results saved to: $OUTPUT_DIR"
    
    # Show best parameters if available
    if [[ -f "$OUTPUT_DIR/best_parameters.json" ]]; then
        echo "🏆 Best parameters found:"
        cat "$OUTPUT_DIR/best_parameters.json" | python3 -m json.tool
    fi
    
    # Show summary statistics
    if [[ -f "$OUTPUT_DIR/grid_search_results.csv" ]]; then
        echo ""
        echo "📈 Summary statistics:"
        python3 -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/grid_search_results.csv')
successful = df[df['training_success'] == True]
print(f'Total combinations tested: {len(df)}')
print(f'Successful combinations: {len(successful)}')
if len(successful) > 0:
    best = successful.loc[successful['val_loss'].idxmin()]
    print(f'Best validation loss: {best[\"val_loss\"]:.6f}')
    print(f'Best threshold: {best[\"threshold\"]}')
    print(f'Best cluster_len: {best[\"cluster_len\"]}')
"
    fi
    
else
    echo "❌ Grid search failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check the logs in $OUTPUT_DIR for more details."
fi

echo ""
echo "============================================================="
echo "Grid Search Pipeline Complete"

exit $EXIT_CODE