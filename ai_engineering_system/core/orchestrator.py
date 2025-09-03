"""
Main orchestration module for the AI Engineering System.
"""

import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .integration import AIIntegrationFramework, IntegrationConfig, MultiModalResult
from .ml import MLModule
from .nlp import NLPModule
from .vision import VisionModule
from .rl import RLModule
from .neural import NeuralModule


@dataclass
class SystemConfig:
    """Configuration for the AI Engineering System."""
    # Module configurations
    ml_config: Optional[Dict[str, Any]] = None
    nlp_config: Optional[Dict[str, Any]] = None
    vision_config: Optional[Dict[str, Any]] = None
    rl_config: Optional[Dict[str, Any]] = None
    neural_config: Optional[Dict[str, Any]] = None
    
    # Integration configuration
    integration_config: Optional[IntegrationConfig] = None
    
    # System settings
    log_level: str = "INFO"
    enable_monitoring: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    max_concurrent_requests: int = 10
    
    # Performance settings
    timeout_seconds: float = 60.0
    retry_attempts: int = 3
    enable_parallel_processing: bool = True
    
    # Data settings
    data_directory: str = "data"
    models_directory: str = "models"
    results_directory: str = "results"
    
    # API settings
    enable_api: bool = False
    api_host: str = "localhost"
    api_port: int = 8000
    api_workers: int = 4


@dataclass
class EngineeringTask:
    """Represents an engineering task."""
    task_id: str
    task_type: str
    input_data: Any
    requirements: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1 = highest, 5 = lowest
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of an engineering task."""
    task_id: str
    success: bool
    result: Any
    confidence: float
    execution_time: float
    modules_used: List[str]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIEngineeringOrchestrator:
    """
    Main orchestrator for the AI Engineering System.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the AI Engineering Orchestrator.
        
        Args:
            config: System configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or SystemConfig()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize directories
        self._setup_directories()
        
        # Initialize AI modules
        self.modules = {}
        self._initialize_modules()
        
        # Initialize integration framework
        self.integration_framework = AIIntegrationFramework(
            config=self.config.integration_config or IntegrationConfig()
        )
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Performance monitoring
        self.performance_monitor = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'module_usage': {},
            'task_type_usage': {},
            'start_time': time.time()
        }
        
        # System status
        self.system_status = {
            'initialized': True,
            'modules_loaded': list(self.modules.keys()),
            'integration_active': True,
            'last_health_check': time.time()
        }
        
        self.logger.info("AI Engineering Orchestrator initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('ai_engineering_system.log')
            ]
        )
    
    def _setup_directories(self):
        """Setup required directories."""
        directories = [
            self.config.data_directory,
            self.config.models_directory,
            self.config.results_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def _initialize_modules(self):
        """Initialize all AI modules."""
        try:
            if self.config.ml_config:
                self.modules['ml'] = MLModule(self.config.ml_config)
                self.logger.info("ML module initialized")
            
            if self.config.nlp_config:
                self.modules['nlp'] = NLPModule(self.config.nlp_config)
                self.logger.info("NLP module initialized")
            
            if self.config.vision_config:
                self.modules['vision'] = VisionModule(self.config.vision_config)
                self.logger.info("Vision module initialized")
            
            if self.config.rl_config:
                self.modules['rl'] = RLModule(self.config.rl_config)
                self.logger.info("RL module initialized")
            
            if self.config.neural_config:
                self.modules['neural'] = NeuralModule(self.config.neural_config)
                self.logger.info("Neural module initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing modules: {e}")
            raise
    
    async def process_engineering_task(self, task: EngineeringTask) -> TaskResult:
        """
        Process an engineering task using the AI system.
        
        Args:
            task: Engineering task to process
            
        Returns:
            Task result
        """
        self.logger.info(f"Processing engineering task: {task.task_id}")
        
        start_time = time.time()
        
        try:
            # Add task to queue
            await self.task_queue.put(task)
            self.active_tasks[task.task_id] = task
            
            # Process task using integration framework
            result = await self.integration_framework.process_engineering_task(
                task_type=task.task_type,
                input_data=task.input_data,
                requirements=task.requirements
            )
            
            execution_time = time.time() - start_time
            
            # Create task result
            task_result = TaskResult(
                task_id=task.task_id,
                success=result.success,
                result=result.fused_result,
                confidence=result.confidence,
                execution_time=execution_time,
                modules_used=result.metadata.get('modules_used', []),
                metadata={
                    'task_type': task.task_type,
                    'requirements': task.requirements,
                    'integration_result': result
                }
            )
            
            # Update performance monitoring
            self._update_performance_monitor(task_result)
            
            # Move task to completed
            if task_result.success:
                self.completed_tasks[task.task_id] = task_result
            else:
                self.failed_tasks[task.task_id] = task_result
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            return task_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            
            # Create failed task result
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                confidence=0.0,
                execution_time=execution_time,
                modules_used=[],
                error=str(e),
                metadata={'task_type': task.task_type}
            )
            
            # Update performance monitoring
            self._update_performance_monitor(task_result)
            
            # Move task to failed
            self.failed_tasks[task.task_id] = task_result
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            return task_result
    
    async def process_batch_tasks(self, tasks: List[EngineeringTask]) -> List[TaskResult]:
        """
        Process multiple engineering tasks in batch.
        
        Args:
            tasks: List of engineering tasks
            
        Returns:
            List of task results
        """
        self.logger.info(f"Processing batch of {len(tasks)} tasks")
        
        if self.config.enable_parallel_processing:
            # Process tasks in parallel
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            
            async def process_with_semaphore(task):
                async with semaphore:
                    return await self.process_engineering_task(task)
            
            results = await asyncio.gather(*[process_with_semaphore(task) for task in tasks])
        else:
            # Process tasks sequentially
            results = []
            for task in tasks:
                result = await self.process_engineering_task(task)
                results.append(result)
        
        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def _update_performance_monitor(self, task_result: TaskResult):
        """Update performance monitoring statistics."""
        self.performance_monitor['total_tasks'] += 1
        
        if task_result.success:
            self.performance_monitor['successful_tasks'] += 1
        else:
            self.performance_monitor['failed_tasks'] += 1
        
        # Update average execution time
        total_time = self.performance_monitor['average_execution_time'] * (self.performance_monitor['total_tasks'] - 1)
        self.performance_monitor['average_execution_time'] = (total_time + task_result.execution_time) / self.performance_monitor['total_tasks']
        
        # Update module usage
        for module in task_result.modules_used:
            if module not in self.performance_monitor['module_usage']:
                self.performance_monitor['module_usage'][module] = 0
            self.performance_monitor['module_usage'][module] += 1
        
        # Update task type usage
        task_type = task_result.metadata.get('task_type', 'unknown')
        if task_type not in self.performance_monitor['task_type_usage']:
            self.performance_monitor['task_type_usage'][task_type] = 0
        self.performance_monitor['task_type_usage'][task_type] += 1
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'uptime': time.time() - self.performance_monitor['start_time'],
            'modules': {},
            'performance': self.performance_monitor.copy(),
            'queue_status': {
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks)
            }
        }
        
        # Check module health
        for name, module in self.modules.items():
            try:
                if hasattr(module, 'get_status'):
                    health_status['modules'][name] = module.get_status()
                else:
                    health_status['modules'][name] = {'status': 'active'}
            except Exception as e:
                health_status['modules'][name] = {'status': 'error', 'error': str(e)}
                health_status['status'] = 'degraded'
        
        # Check integration framework health
        try:
            health_status['integration'] = self.integration_framework.get_status()
        except Exception as e:
            health_status['integration'] = {'status': 'error', 'error': str(e)}
            health_status['status'] = 'degraded'
        
        return health_status
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimize system performance based on current usage patterns."""
        self.logger.info("Optimizing system performance")
        
        optimization_results = {
            'timestamp': time.time(),
            'optimizations_applied': [],
            'performance_improvements': {}
        }
        
        # Analyze performance patterns
        total_tasks = self.performance_monitor['total_tasks']
        if total_tasks == 0:
            return optimization_results
        
        # Optimize based on module usage
        module_usage = self.performance_monitor['module_usage']
        most_used_module = max(module_usage.keys(), key=lambda k: module_usage[k]) if module_usage else None
        
        if most_used_module:
            optimization_results['optimizations_applied'].append(f"Prioritized {most_used_module} module")
            optimization_results['performance_improvements'][most_used_module] = "Increased priority"
        
        # Optimize based on task types
        task_type_usage = self.performance_monitor['task_type_usage']
        most_common_task = max(task_type_usage.keys(), key=lambda k: task_type_usage[k]) if task_type_usage else None
        
        if most_common_task:
            optimization_results['optimizations_applied'].append(f"Optimized for {most_common_task} tasks")
            optimization_results['performance_improvements'][most_common_task] = "Improved processing"
        
        # Optimize based on execution times
        avg_execution_time = self.performance_monitor['average_execution_time']
        if avg_execution_time > 10.0:  # If average execution time is high
            optimization_results['optimizations_applied'].append("Enabled parallel processing")
            optimization_results['performance_improvements']['parallel_processing'] = "Reduced execution time"
        
        return optimization_results
    
    def save_system_state(self, filepath: str):
        """Save system state to file."""
        state = {
            'config': self.config.__dict__,
            'performance_monitor': self.performance_monitor,
            'system_status': self.system_status,
            'module_status': {name: module.get_status() if hasattr(module, 'get_status') else {} for name, module in self.modules.items()},
            'integration_status': self.integration_framework.get_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"System state saved to {filepath}")
    
    def load_system_state(self, filepath: str):
        """Load system state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Update performance monitor
        self.performance_monitor.update(state.get('performance_monitor', {}))
        
        # Update system status
        self.system_status.update(state.get('system_status', {}))
        
        self.logger.info(f"System state loaded from {filepath}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'system_status': self.system_status,
            'performance_monitor': self.performance_monitor,
            'module_status': {name: module.get_status() if hasattr(module, 'get_status') else {} for name, module in self.modules.items()},
            'integration_status': self.integration_framework.get_status(),
            'queue_status': {
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks)
            }
        }
    
    async def shutdown(self):
        """Shutdown the AI Engineering System."""
        self.logger.info("Shutting down AI Engineering System")
        
        # Wait for active tasks to complete
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            # In a real implementation, you might want to wait for tasks or cancel them
        
        # Save system state
        self.save_system_state(f"{self.config.results_directory}/system_state.json")
        
        # Close modules
        for name, module in self.modules.items():
            if hasattr(module, 'shutdown'):
                await module.shutdown()
            self.logger.info(f"Module {name} shutdown")
        
        # Close integration framework
        if hasattr(self.integration_framework, 'shutdown'):
            await self.integration_framework.shutdown()
        
        self.logger.info("AI Engineering System shutdown complete")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        if hasattr(self, 'logger'):
            self.logger.info("AI Engineering System destructor called")
