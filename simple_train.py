#!/usr/bin/env python3
"""
Simplified training script for the AI Engineering System.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import json
from pathlib import Path


class SimpleAITrainer:
    """Simplified AI trainer for demonstration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}
    
    def generate_training_data(self, n_samples=1000):
        """Generate simple training data."""
        # Generate random engineering-like data
        X = np.random.rand(n_samples, 10)  # 10 features
        y = np.random.randint(0, 3, n_samples)  # 3 classes
        
        # Split data
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_ml_model(self, X_train, y_train, X_test, y_test):
        """Train a simple ML model."""
        self.logger.info("Training ML model...")
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['ml'] = model
        self.results['ml'] = {'accuracy': accuracy}
        
        return model, accuracy
    
    def train_neural_model(self, X_train, y_train, X_test, y_test):
        """Train a simple neural network."""
        self.logger.info("Training Neural Network...")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create simple neural network
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(10, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3)
                )
            
            def forward(self, x):
                return self.network(x)
        
        model = SimpleNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean().item()
        
        self.models['neural'] = model
        self.results['neural'] = {'accuracy': accuracy}
        
        return model, accuracy
    
    def save_models(self, output_dir="trained_models"):
        """Save trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save ML model
        if 'ml' in self.models:
            import joblib
            joblib.dump(self.models['ml'], output_path / 'ml_model.pkl')
        
        # Save Neural model
        if 'neural' in self.models:
            torch.save(self.models['neural'].state_dict(), output_path / 'neural_model.pth')
        
        # Save results
        with open(output_path / 'training_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Models saved to {output_dir}")
    
    async def train_all(self):
        """Train all models."""
        self.logger.info("Starting AI Engineering System Training...")
        start_time = time.time()
        
        # Generate data
        X_train, X_test, y_train, y_test = self.generate_training_data()
        
        # Train models
        ml_model, ml_accuracy = self.train_ml_model(X_train, y_train, X_test, y_test)
        neural_model, neural_accuracy = self.train_neural_model(X_train, y_train, X_test, y_test)
        
        # Save models
        self.save_models()
        
        training_time = time.time() - start_time
        
        return {
            'success': True,
            'training_time': training_time,
            'ml_accuracy': ml_accuracy,
            'neural_accuracy': neural_accuracy,
            'models_trained': ['ml', 'neural']
        }


async def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🚀 Starting Simplified AI Engineering System Training...")
    print("="*60)
    
    try:
        # Create trainer
        trainer = SimpleAITrainer()
        
        # Train models
        result = await trainer.train_all()
        
        if result['success']:
            print("\n" + "="*60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"⏱️  Training Time: {result['training_time']:.2f} seconds")
            print(f"🤖 Models Trained: {len(result['models_trained'])}")
            print(f"📊 ML Model Accuracy: {result['ml_accuracy']:.3f}")
            print(f"🧠 Neural Network Accuracy: {result['neural_accuracy']:.3f}")
            print("💾 Models saved to: trained_models/")
            print("="*60)
            print("✅ Your AI system is ready for testing!")
            print("="*60)
            return True
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎯 Training completed! You can now test the AI system.")
    else:
        print("\n💥 Training failed. Check the logs for details.")
