"""
Basic usage examples for the AI Engineering System.
"""

import asyncio
import numpy as np
from typing import Dict, Any

from ..core.main import EngineeringAI
from ..core.orchestrator import SystemConfig, EngineeringTask


async def basic_example():
    """Basic usage example."""
    print("=" * 60)
    print("AI ENGINEERING SYSTEM - BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize the AI system
    ai_system = EngineeringAI(device="cpu")
    
    try:
        # Example 1: Structural Analysis
        print("\n1. Structural Analysis Example")
        print("-" * 40)
        
        structural_data = {
            "numerical_data": np.random.rand(100, 10),
            "complex_data": np.random.rand(100, 10)
        }
        
        result = await ai_system.analyze_engineering_problem(
            structural_data,
            modules=["ml", "neural"]
        )
        
        print(f"Structural analysis completed: {result}")
        
        # Example 2: Document Analysis
        print("\n2. Document Analysis Example")
        print("-" * 40)
        
        document_data = {
            "text_data": "Engineering analysis report: The beam shows stress concentration at the midspan. Material properties: E=200GPa, yield strength=250MPa."
        }
        
        result = await ai_system.analyze_engineering_problem(
            document_data,
            modules=["nlp"]
        )
        
        print(f"Document analysis completed: {result}")
        
        # Example 3: Image Processing
        print("\n3. Image Processing Example")
        print("-" * 40)
        
        image_data = {
            "image_data": np.random.rand(224, 224, 3)
        }
        
        result = await ai_system.analyze_engineering_problem(
            image_data,
            modules=["vision"]
        )
        
        print(f"Image processing completed: {result}")
        
        # Example 4: Design Optimization
        print("\n4. Design Optimization Example")
        print("-" * 40)
        
        optimization_data = {
            "numerical_data": np.random.rand(100, 10),
            "optimization_data": {"objective": "minimize", "constraints": []},
            "complex_data": np.random.rand(100, 10)
        }
        
        result = await ai_system.analyze_engineering_problem(
            optimization_data,
            modules=["rl", "ml", "neural"]
        )
        
        print(f"Design optimization completed: {result}")
        
        # Example 5: Multi-modal Analysis
        print("\n5. Multi-modal Analysis Example")
        print("-" * 40)
        
        multimodal_data = {
            "numerical_data": np.random.rand(100, 10),
            "text_data": "Engineering specifications and requirements",
            "image_data": np.random.rand(224, 224, 3),
            "optimization_data": {"objective": "minimize", "constraints": []},
            "complex_data": np.random.rand(100, 10)
        }
        
        result = await ai_system.analyze_engineering_problem(
            multimodal_data,
            modules=["ml", "nlp", "vision", "rl", "neural"]
        )
        
        print(f"Multi-modal analysis completed: {result}")
        
        # System status
        print("\n6. System Status")
        print("-" * 40)
        status = ai_system.get_system_status()
        print(f"System status: {status}")
        
    finally:
        # Shutdown the system
        await ai_system.shutdown()
        print("\nAI system shutdown complete.")


async def advanced_example():
    """Advanced usage example with custom configuration."""
    print("\n" + "=" * 60)
    print("AI ENGINEERING SYSTEM - ADVANCED USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize with custom configuration
    ai_system = EngineeringAI(device="cpu")
    
    try:
        # Direct orchestrator usage
        orchestrator = ai_system.orchestrator
        
        # Create custom engineering task
        task = EngineeringTask(
            task_id="custom_analysis_001",
            task_type="structural_analysis",
            input_data={
                "numerical_data": np.random.rand(200, 15),
                "material_properties": {
                    "E": 200e9,  # Young's modulus in Pa
                    "nu": 0.3,   # Poisson's ratio
                    "rho": 7850  # Density in kg/m³
                },
                "geometry": {
                    "length": 10.0,  # meters
                    "width": 0.5,    # meters
                    "height": 0.3    # meters
                },
                "loading": {
                    "distributed_load": 10000,  # N/m
                    "point_loads": [{"position": 5.0, "magnitude": 50000}]  # N
                }
            },
            requirements={
                "modules": ["ml", "neural"],
                "accuracy_threshold": 0.95,
                "max_processing_time": 30.0
            }
        )
        
        print("\nProcessing custom engineering task...")
        result = await orchestrator.process_engineering_task(task)
        
        print(f"Task completed successfully: {result.success}")
        print(f"Task confidence: {result.confidence:.3f}")
        print(f"Processing time: {result.processing_time:.3f} seconds")
        
        if result.success:
            print(f"Result data: {result.result_data}")
        else:
            print(f"Error: {result.error}")
        
        # System monitoring
        print("\nSystem monitoring:")
        status = orchestrator.get_system_status()
        print(f"Active modules: {status.get('active_modules', [])}")
        print(f"System health: {status.get('system_health', 'unknown')}")
        
    finally:
        await ai_system.shutdown()
        print("\nAdvanced example completed.")


async def performance_example():
    """Performance testing example."""
    print("\n" + "=" * 60)
    print("AI ENGINEERING SYSTEM - PERFORMANCE EXAMPLE")
    print("=" * 60)
    
    ai_system = EngineeringAI(device="cpu")
    
    try:
        # Performance test with multiple tasks
        tasks = []
        for i in range(5):
            task_data = {
                "numerical_data": np.random.rand(50, 8),
                "text_data": f"Engineering analysis task {i+1}",
                "complex_data": np.random.rand(50, 8)
            }
            tasks.append(task_data)
        
        print(f"\nProcessing {len(tasks)} tasks concurrently...")
        start_time = asyncio.get_event_loop().time()
        
        # Process tasks concurrently
        results = await asyncio.gather(*[
            ai_system.analyze_engineering_problem(task_data, modules=["ml", "nlp"])
            for task_data in tasks
        ])
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        print(f"All tasks completed in {total_time:.3f} seconds")
        print(f"Average time per task: {total_time/len(tasks):.3f} seconds")
        
        # Analyze results
        successful_tasks = sum(1 for r in results if r.get("success", False))
        print(f"Successful tasks: {successful_tasks}/{len(tasks)}")
        
    finally:
        await ai_system.shutdown()
        print("\nPerformance example completed.")


async def testing_example():
    """Testing example."""
    print("\n" + "=" * 60)
    print("AI ENGINEERING SYSTEM - TESTING EXAMPLE")
    print("=" * 60)
    
    ai_system = EngineeringAI(device="cpu")
    
    try:
        # Import test runner
        from ..tests.test_runner import TestRunner
        
        # Initialize test runner
        test_runner = TestRunner(ai_system.orchestrator)
        
        print("\nRunning unit tests...")
        unit_results = await test_runner.run_test_suite("unit")
        print(f"Unit tests: {unit_results['passed_tests']}/{unit_results['total_tests']} passed")
        
        print("\nRunning integration tests...")
        integration_results = await test_runner.run_test_suite("integration")
        print(f"Integration tests: {integration_results['passed_tests']}/{integration_results['total_tests']} passed")
        
        print("\nRunning validation tests...")
        validation_results = await test_runner.run_test_suite("validation")
        print(f"Validation tests: {validation_results['passed_tests']}/{validation_results['total_tests']} passed")
        
        # Generate test report
        print("\nTest Report:")
        print("-" * 40)
        test_runner.print_test_report()
        
    except ImportError as e:
        print(f"Test runner not available: {e}")
    
    finally:
        await ai_system.shutdown()
        print("\nTesting example completed.")


def main():
    """Main function to run all examples."""
    print("Starting AI Engineering System Examples...")
    
    # Run basic example
    asyncio.run(basic_example())
    
    # Run advanced example
    asyncio.run(advanced_example())
    
    # Run performance example
    asyncio.run(performance_example())
    
    # Run testing example
    asyncio.run(testing_example())
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()
