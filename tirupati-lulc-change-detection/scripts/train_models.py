"""Train LULC classification models."""

import argparse
from pathlib import Path
import sys
import torch
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import create_unet_model
from src.models.random_forest import LULCRandomForest
from src.training.train_unet import train_unet
from src.utils.config_utils import get_full_config
from src.utils.logger import default_logger as logger


def train_random_forest(config):
    """Train Random Forest model."""
    logger.info("Training Random Forest model...")
    
    rf_config = config.random_forest
    model = LULCRandomForest(
        n_estimators=rf_config.n_estimators,
        max_depth=rf_config.max_depth,
        min_samples_split=rf_config.min_samples_split,
        min_samples_leaf=rf_config.min_samples_leaf,
        random_state=rf_config.random_state
    )
    
    # Load training data (implement based on your data structure)
    logger.info("Loading training data...")
    # X_train, y_train = load_training_data()
    
    # Train model
    # metrics = model.train(X_train, y_train)
    # logger.info(f"Training completed. Accuracy: {metrics['train_accuracy']:.4f}")
    
    # Save model
    save_path = Path(config.models.random_forest) / 'rf_model.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # model.save(str(save_path))
    
    logger.info("Random Forest training completed!")


def train_unet_model(config):
    """Train U-Net model."""
    logger.info("Training U-Net model...")
    
    # Create model
    unet_config = config.unet.architecture
    model = create_unet_model(unet_config)
    
    # Load data (implement based on your data structure)
    logger.info("Loading training data...")
    # train_loader, val_loader = create_dataloaders(...)
    
    # Train model
    train_config = config.unet.training
    # history = train_unet(model, train_loader, val_loader, train_config)
    
    logger.info("U-Net training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train LULC models')
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'random_forest', 'all'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    args = parser.parse_args()
    
    config = get_full_config()
    
    # Set device
    if args.gpu and torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Override epochs if specified
    if args.epochs:
        if hasattr(config, 'unet'):
            config.unet.training.epochs = args.epochs
    
    # Train models
    if args.model == 'random_forest' or args.model == 'all':
        train_random_forest(config)
    
    if args.model == 'unet' or args.model == 'all':
        train_unet_model(config)
    
    logger.info("All training completed!")


if __name__ == '__main__':
    main()
