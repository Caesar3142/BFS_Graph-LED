"""
Evaluation script for Graph-LED model.
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from models import GraphLED
from data import create_dataloader
from utils import compute_metrics, visualize_prediction_comparison, set_seed


def main():
    parser = argparse.ArgumentParser(description='Evaluate Graph-LED model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, help='Path to data file (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print('Loading data...')
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config['data']['path']
    
    data = np.load(data_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()
    
    node_positions = data['node_positions']
    flow_fields = data['flow_fields']
    
    # Create dataloaders
    train_loader, val_loader, test_loader, dataset_stats = create_dataloader(
        node_positions=node_positions,
        flow_fields=flow_fields,
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        batch_size=1,
        shuffle=False,
        normalize=config['data'].get('normalize', True),
        k_neighbors=config['data'].get('k_neighbors', 8),
        graph_type=config['data'].get('graph_type', 'knn')
    )
    
    # Create model
    model = GraphLED(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        output_dim=config['model']['output_dim'],
        num_gnn_layers=config['model']['num_gnn_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_attention_layers=config['model']['num_attention_layers'],
        use_edge_features=config['model'].get('use_edge_features', True),
        dropout=config['model'].get('dropout', 0.1),
        aggregation=config['model'].get('aggregation', 'mean')
    ).to(device)
    
    # Load checkpoint
    if args.checkpoint.endswith('.pt'):
        # Full checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    model.eval()
    print('Model loaded successfully')
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            input_sequence = batch['input_sequence'].to(device)
            target_sequence = batch['target_sequence'].to(device)
            edge_index = batch['edge_index'].to(device)
            node_positions = batch['node_positions'].to(device)
            
            # Predict
            predictions = model(
                input_sequence,
                edge_index,
                node_positions=node_positions,
                num_steps=target_sequence.shape[1]
            )
            
            all_predictions.append(predictions.cpu())
            all_targets.append(target_sequence.cpu())
            
            # Visualize first few samples
            if args.visualize and batch_idx < 5:
                pred_np = predictions[0].cpu().numpy()
                target_np = target_sequence[0].cpu().numpy()
                node_pos_np = node_positions[0].cpu().numpy()
                
                for t in range(min(3, pred_np.shape[0])):
                    for f in range(pred_np.shape[2]):
                        save_path = os.path.join(
                            args.output_dir,
                            f'prediction_batch{batch_idx}_timestep{t}_feature{f}.png'
                        )
                        visualize_prediction_comparison(
                            pred_np[t:t+1],
                            target_np[t:t+1],
                            node_pos_np,
                            timestep=0,
                            feature_idx=f,
                            save_path=save_path
                        )
                    
                    # Visualize velocity magnitude if u and v are available
                    if pred_np.shape[2] >= 2:
                        pred_u = pred_np[t, :, 0]
                        pred_v = pred_np[t, :, 1]
                        pred_magnitude = np.sqrt(pred_u**2 + pred_v**2)
                        
                        target_u = target_np[t, :, 0]
                        target_v = target_np[t, :, 1]
                        target_magnitude = np.sqrt(target_u**2 + target_v**2)
                        
                        # Create magnitude arrays with same shape as original
                        pred_mag_array = pred_magnitude.reshape(1, -1, 1)
                        target_mag_array = target_magnitude.reshape(1, -1, 1)
                        
                        save_path = os.path.join(
                            args.output_dir,
                            f'prediction_batch{batch_idx}_timestep{t}_velocity_magnitude.png'
                        )
                        visualize_prediction_comparison(
                            pred_mag_array,
                            target_mag_array,
                            node_pos_np,
                            timestep=0,
                            feature_idx=0,
                            save_path=save_path
                        )
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    # Print results
    print('\n' + '='*50)
    print('Test Set Metrics:')
    print('='*50)
    print(f'MSE:  {metrics["mse"]:.6e}')
    print(f'RMSE: {metrics["rmse"]:.6e}')
    print(f'MAE:  {metrics["mae"]:.6e}')
    print(f'Relative Error: {metrics["relative_error"]:.6e}')
    print('\nPer-feature MSE:')
    for i, mse in enumerate(metrics['per_feature_mse']):
        print(f'  Feature {i}: {mse:.6e}')
    print('\nPer-feature MAE:')
    for i, mae in enumerate(metrics['per_feature_mae']):
        print(f'  Feature {i}: {mae:.6e}')
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.npz')
    np.savez(metrics_path, **{k: v for k, v in metrics.items() if isinstance(v, (int, float, np.ndarray))})
    print(f'\nMetrics saved to {metrics_path}')
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.npz')
    np.savez(
        predictions_path,
        predictions=all_predictions.numpy(),
        targets=all_targets.numpy()
    )
    print(f'Predictions saved to {predictions_path}')


if __name__ == '__main__':
    main()

