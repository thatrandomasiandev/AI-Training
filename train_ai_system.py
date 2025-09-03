#!/usr/bin/env python3
"""
Training script for the AI Engineering System.
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from ai_engineering_system.training import TrainingPipeline, TrainingConfig
from ai_engineering_system.utils.logger import setup_logger


def setup_training_logging():
    """Setup logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train the AI Engineering System')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (cpu/cuda/auto)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--output-dir', type=str, default='trained_models', help='Output directory for trained models')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint file')
    parser.add_argument('--quick', action='store_true', help='Quick training with reduced samples and epochs')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_training_logging()
    logger.info("Starting AI Engineering System Training")
    logger.info(f"Arguments: {args}")
    
    # Create training configuration
    if args.quick:
        # Quick training configuration
        config = TrainingConfig(
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=min(args.epochs, 10),  # Reduced epochs for quick training
            num_training_samples=min(args.samples, 1000),  # Reduced samples
            num_validation_samples=200,
            num_test_samples=100,
            output_dir=args.output_dir,
            early_stopping_patience=5,
            checkpoint_frequency=5
        )
        logger.info("Using quick training configuration")
    else:
        # Full training configuration
        config = TrainingConfig(
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            num_training_samples=args.samples,
            num_validation_samples=args.samples // 5,
            num_test_samples=args.samples // 10,
            output_dir=args.output_dir,
            early_stopping_patience=10,
            checkpoint_frequency=10
        )
        logger.info("Using full training configuration")
    
    try:
        # Create training pipeline
        pipeline = TrainingPipeline(config)
        
        # Resume training if checkpoint provided
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            await pipeline.resume_training(args.resume)
        
        # Run complete training
        logger.info("Starting complete training pipeline...")
        result = await pipeline.run_complete_training()
        
        if result.success:
            logger.info("🎉 Training completed successfully!")
            logger.info(f"Training time: {result.training_time:.2f} seconds")
            logger.info(f"Models trained: {result.models_trained}")
            logger.info(f"Best models: {result.best_models}")
            
            # Save final results
            results_file = Path(args.output_dir) / "final_training_results.json"
            pipeline.save_pipeline_results(str(results_file))
            logger.info(f"Results saved to: {results_file}")
            
            # Print summary
            print("\n" + "="*80)
            print("🎯 AI ENGINEERING SYSTEM TRAINING COMPLETE!")
            print("="*80)
            print(f"⏱️  Training Time: {result.training_time:.2f} seconds")
            print(f"🤖 Models Trained: {sum(len(models) for models in result.models_trained.values())}")
            print(f"🏆 Best Models:")
            for model_type, best_model in result.best_models.items():
                print(f"   {model_type.upper()}: {best_model}")
            print(f"📊 Performance Metrics Available: {len(result.performance_metrics)} model types")
            print(f"💾 Models Saved to: {args.output_dir}")
            print("="*80)
            print("🚀 Your AI Engineering System is now trained and ready to solve engineering problems!")
            print("="*80)
            
        else:
            logger.error(f"❌ Training failed: {result.error}")
            print(f"\n❌ Training failed: {result.error}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        print(f"\n❌ Training failed with exception: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
