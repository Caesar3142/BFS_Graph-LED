#!/bin/bash
# Quick start script for backward-facing step problem

set -e  # Exit on error

echo "=========================================="
echo "Graph-LED: Backward-Facing Step"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Create data directory
mkdir -p data
mkdir -p checkpoints
mkdir -p results

echo ""
echo "Choose an option:"
echo "1) Generate synthetic data"
echo "2) Convert OpenFOAM data"
echo "3) Train model (requires data)"
echo "4) Evaluate model (requires trained model)"
echo "5) Full workflow (generate data + train + evaluate)"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "Generating synthetic backward-facing step data..."
        python -c "from generate_synthetic_data import generate_dataset; generate_dataset('backward_step', num_timesteps=200, Re=100)"
        echo "Data generated: data/backward_step_flow.npz"
        ;;
    2)
        read -p "Enter path to OpenFOAM case: " case_path
        echo "Converting OpenFOAM data..."
        python convert_openfoam_data.py "$case_path" --output data/backward_step_flow.npz
        echo "Data converted: data/backward_step_flow.npz"
        ;;
    3)
        if [ ! -f "data/backward_step_flow.npz" ]; then
            echo "Error: data/backward_step_flow.npz not found!"
            echo "Please generate or convert data first (options 1 or 2)"
            exit 1
        fi
        echo "Training model..."
        python train.py --config configs/backward_step.yaml
        echo "Training complete! Check checkpoints/ for saved models"
        ;;
    4)
        if [ ! -f "checkpoints/best_model.pt" ]; then
            echo "Error: checkpoints/best_model.pt not found!"
            echo "Please train the model first (option 3)"
            exit 1
        fi
        echo "Evaluating model..."
        python evaluate.py --config configs/backward_step.yaml \
            --checkpoint checkpoints/best_model.pt \
            --visualize
        echo "Evaluation complete! Check results/ for outputs"
        ;;
    5)
        echo "Running full workflow..."
        echo "Step 1: Generating synthetic data..."
        python -c "from generate_synthetic_data import generate_dataset; generate_dataset('backward_step', num_timesteps=200, Re=100)"
        echo "Step 2: Training model..."
        python train.py --config configs/backward_step.yaml
        echo "Step 3: Evaluating model..."
        python evaluate.py --config configs/backward_step.yaml \
            --checkpoint checkpoints/best_model.pt \
            --visualize
        echo ""
        echo "=========================================="
        echo "Full workflow complete!"
        echo "Check results/ for evaluation outputs"
        echo "=========================================="
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Done!"

