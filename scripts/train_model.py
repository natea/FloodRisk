#!/usr/bin/env python3
"""
Training script for flood prediction model.
Simplified entry point for the ml-training branch.
"""

import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train flood prediction model based on APPROACH.md"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        choices=["baseline", "multiscale", "generalization"],
        default="baseline",
        help="Training phase per APPROACH.md"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing training data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs",
        help="Output directory for models and logs"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with dummy data"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🌊 FloodRisk ML Training - Based on APPROACH.md")
    logger.info("=" * 60)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"W&B logging: {args.wandb}")
    logger.info(f"Debug mode: {args.debug}")
    
    try:
        # Import training module
        from src.ml.training.train import train_model
        from omegaconf import OmegaConf
        
        # Load configuration
        config = OmegaConf.load(args.config)
        
        # Override config with command line arguments
        if args.phase:
            config.phase.current = args.phase
            
        if args.data_dir:
            config.data.data_dir = str(args.data_dir)
            
        if args.output_dir:
            config.logging.save_dir = str(args.output_dir)
            
        if args.wandb:
            config.logging.wandb.enabled = True
            
        if args.debug:
            logger.warning("Debug mode enabled - using dummy data")
            config.data.use_dummy_data = True
        
        # Phase-specific configuration
        if args.phase == "baseline":
            logger.info("🎯 Phase 1: Baseline Implementation")
            logger.info("Target: IoU ≥ 0.70 on held-out neighborhoods")
            config.model.in_channels = 2  # [DEM, Rain]
            config.model.use_fpn = False
            config.loss.type = "bce_dice"
            
        elif args.phase == "multiscale":
            logger.info("🎯 Phase 2: Multi-scale & Features")
            logger.info("Adding FPN + derived features")
            config.model.in_channels = 6  # [DEM, Rain, Slope, Hand, Flow, Curvature]
            config.model.use_fpn = True
            config.data.input_features = [
                "dem", "rainfall", "slope", "hand", 
                "flow_accumulation", "plan_curvature"
            ]
            
        elif args.phase == "generalization":
            logger.info("🎯 Phase 3: Generalization")
            logger.info("Cross-city validation & few-shot adaptation")
            config.validation.cross_city_validation = True
            config.phase.generalization.enable_cross_city = True
        
        logger.info("Starting training...")
        
        # Run training with Hydra
        import hydra
        from hydra import compose, initialize
        from hydra.core.config_store import ConfigStore
        
        # Register config
        cs = ConfigStore.instance()
        cs.store(name="config", node=config)
        
        # Initialize and run
        with initialize(version_base=None, config_path=None):
            cfg = compose(config_name="config")
            trainer, model = train_model(cfg)
            
        logger.info("✅ Training completed successfully!")
        
        # Print phase results
        if args.phase == "baseline":
            logger.info("📊 Phase 1 Results:")
            logger.info("Check validation IoU ≥ 0.70 target")
            
        elif args.phase == "multiscale":
            logger.info("📊 Phase 2 Results:")
            logger.info("Compare multi-scale performance vs baseline")
            
        elif args.phase == "generalization":
            logger.info("📊 Phase 3 Results:")
            logger.info("Evaluate cross-city generalization")
        
        return trainer, model
        
    except ImportError as e:
        logger.error(f"Import error - install ML dependencies: pip install -r requirements-ml.txt")
        logger.error(f"Error: {e}")
        return None, None
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()