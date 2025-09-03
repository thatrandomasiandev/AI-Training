#!/usr/bin/env python3
"""
Quick training script for testing the AI Engineering System.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from ai_engineering_system.training import TrainingPipeline, TrainingConfig


async def quick_train():
    """Quick training for testing purposes."""
    print("🚀 Starting Quick AI Engineering System Training...")
    print("="*60)
    
    # Quick training configuration
    config = TrainingConfig(
        device="cpu",  # Use CPU for quick training
        batch_size=16,
        learning_rate=0.01,
        num_epochs=5,  # Very few epochs for quick training
        num_training_samples=100,  # Very few samples
        num_validation_samples=20,
        num_test_samples=10,
        output_dir="quick_trained_models",
        early_stopping_patience=3,
        checkpoint_frequency=2
    )
    
    # Setup simple logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Create training pipeline
        pipeline = TrainingPipeline(config)
        
        print("📊 Generating training data...")
        print("🤖 Training AI models...")
        print("🔍 Validating models...")
        print("⚡ Optimizing models...")
        print("🔗 Testing integration...")
        print("✅ Final validation...")
        
        # Run training
        result = await pipeline.run_complete_training()
        
        if result.success:
            print("\n" + "="*60)
            print("🎉 QUICK TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"⏱️  Training Time: {result.training_time:.2f} seconds")
            print(f"🤖 Models Trained: {sum(len(models) for models in result.models_trained.values())}")
            print(f"🏆 Best Models:")
            for model_type, best_model in result.best_models.items():
                print(f"   {model_type.upper()}: {best_model}")
            print("="*60)
            print("✅ Your AI system is ready for testing!")
            print("="*60)
            return True
        else:
            print(f"\n❌ Training failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"\n❌ Training failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(quick_train())
    if success:
        print("\n🎯 Quick training completed! You can now test the AI system.")
    else:
        print("\n💥 Quick training failed. Check the logs for details.")
        sys.exit(1)
