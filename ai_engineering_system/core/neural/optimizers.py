"""
Custom optimizers for engineering applications.
"""

import logging
import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union, Tuple
import math


class EngineeringOptimizer:
    """
    Base engineering optimizer with common engineering features.
    """
    
    def __init__(self, params, lr: float = 0.001, weight_decay: float = 1e-4,
                 momentum: float = 0.9, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, amsgrad: bool = False):
        """
        Initialize engineering optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            weight_decay: Weight decay
            momentum: Momentum
            beta1: Beta1 for Adam
            beta2: Beta2 for Adam
            eps: Epsilon
            amsgrad: Whether to use AMSGrad
        """
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        
        # State tracking
        self.state = {}
        self.step_count = 0
        
        # Engineering-specific parameters
        self.adaptive_lr = True
        self.gradient_clipping = True
        self.clip_value = 1.0
        
    def step(self, closure=None):
        """Perform optimization step."""
        raise NotImplementedError("Subclasses must implement step method")
    
    def zero_grad(self):
        """Zero gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def state_dict(self):
        """Get optimizer state."""
        return {
            'state': self.state,
            'step_count': self.step_count,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'amsgrad': self.amsgrad
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.state = state_dict['state']
        self.step_count = state_dict['step_count']
        self.lr = state_dict['lr']
        self.weight_decay = state_dict['weight_decay']
        self.momentum = state_dict['momentum']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.eps = state_dict['eps']
        self.amsgrad = state_dict['amsgrad']
    
    def _clip_gradients(self):
        """Clip gradients."""
        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_value)
    
    def _adaptive_learning_rate(self):
        """Adapt learning rate based on training progress."""
        if self.adaptive_lr:
            # Simple adaptive learning rate based on step count
            decay_factor = 1.0 / (1.0 + 0.1 * self.step_count)
            return self.lr * decay_factor
        return self.lr


class AdamW(EngineeringOptimizer):
    """
    AdamW optimizer with engineering-specific features.
    """
    
    def __init__(self, params, lr: float = 0.001, weight_decay: float = 1e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 amsgrad: bool = False):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            weight_decay: Weight decay
            beta1: Beta1 for Adam
            beta2: Beta2 for Adam
            eps: Epsilon
            amsgrad: Whether to use AMSGrad
        """
        super(AdamW, self).__init__(params, lr, weight_decay, beta1=beta1, beta2=beta2, eps=eps, amsgrad=amsgrad)
        
        # Initialize state for each parameter
        for param in self.params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data)
            }
            if self.amsgrad:
                self.state[param]['max_exp_avg_sq'] = torch.zeros_like(param.data)
    
    def step(self, closure=None):
        """Perform optimization step."""
        if closure is not None:
            loss = closure()
        else:
            loss = None
        
        # Clip gradients
        self._clip_gradients()
        
        # Adaptive learning rate
        current_lr = self._adaptive_learning_rate()
        
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[param]
            
            # Update step count
            state['step'] += 1
            self.step_count += 1
            
            # Decay the first and second moment running average coefficient
            state['exp_avg'] = self.beta1 * state['exp_avg'] + (1 - self.beta1) * grad
            state['exp_avg_sq'] = self.beta2 * state['exp_avg_sq'] + (1 - self.beta2) * grad * grad
            
            if self.amsgrad:
                # Maintains the maximum of all 2nd moment running avg. until now
                torch.max(state['max_exp_avg_sq'], state['exp_avg_sq'], out=state['max_exp_avg_sq'])
                # Use the max. for normalizing running avg. of gradient
                denom = state['max_exp_avg_sq'].sqrt().add_(self.eps)
            else:
                denom = state['exp_avg_sq'].sqrt().add_(self.eps)
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            step_size = current_lr * math.sqrt(bias_correction2) / bias_correction1
            
            # Update parameters
            param.data.mul_(1 - current_lr * self.weight_decay)
            param.data.addcdiv_(state['exp_avg'], denom, value=-step_size)
        
        return loss


class RAdam(EngineeringOptimizer):
    """
    RAdam optimizer with engineering-specific features.
    """
    
    def __init__(self, params, lr: float = 0.001, weight_decay: float = 1e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        Initialize RAdam optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            weight_decay: Weight decay
            beta1: Beta1 for Adam
            beta2: Beta2 for Adam
            eps: Epsilon
        """
        super(RAdam, self).__init__(params, lr, weight_decay, beta1=beta1, beta2=beta2, eps=eps)
        
        # Initialize state for each parameter
        for param in self.params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data)
            }
    
    def step(self, closure=None):
        """Perform optimization step."""
        if closure is not None:
            loss = closure()
        else:
            loss = None
        
        # Clip gradients
        self._clip_gradients()
        
        # Adaptive learning rate
        current_lr = self._adaptive_learning_rate()
        
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[param]
            
            # Update step count
            state['step'] += 1
            self.step_count += 1
            
            # Decay the first and second moment running average coefficient
            state['exp_avg'] = self.beta1 * state['exp_avg'] + (1 - self.beta1) * grad
            state['exp_avg_sq'] = self.beta2 * state['exp_avg_sq'] + (1 - self.beta2) * grad * grad
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            
            # Rectified Adam
            rho_inf = 2 / (1 - self.beta2) - 1
            rho_t = rho_inf - 2 * state['step'] * (self.beta2 ** state['step']) / (1 - self.beta2 ** state['step'])
            
            if rho_t > 4:
                # Compute the variance rectification term
                r_t = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                
                # Compute the effective step size
                step_size = current_lr * r_t * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                param.data.mul_(1 - current_lr * self.weight_decay)
                param.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(self.eps), value=-step_size)
            else:
                # Use unrectified update
                step_size = current_lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                param.data.mul_(1 - current_lr * self.weight_decay)
                param.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(self.eps), value=-step_size)
        
        return loss


class AdaBelief(EngineeringOptimizer):
    """
    AdaBelief optimizer with engineering-specific features.
    """
    
    def __init__(self, params, lr: float = 0.001, weight_decay: float = 1e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        Initialize AdaBelief optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            weight_decay: Weight decay
            beta1: Beta1 for Adam
            beta2: Beta2 for Adam
            eps: Epsilon
        """
        super(AdaBelief, self).__init__(params, lr, weight_decay, beta1=beta1, beta2=beta2, eps=eps)
        
        # Initialize state for each parameter
        for param in self.params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data)
            }
    
    def step(self, closure=None):
        """Perform optimization step."""
        if closure is not None:
            loss = closure()
        else:
            loss = None
        
        # Clip gradients
        self._clip_gradients()
        
        # Adaptive learning rate
        current_lr = self._adaptive_learning_rate()
        
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[param]
            
            # Update step count
            state['step'] += 1
            self.step_count += 1
            
            # Decay the first and second moment running average coefficient
            state['exp_avg'] = self.beta1 * state['exp_avg'] + (1 - self.beta1) * grad
            state['exp_avg_sq'] = self.beta2 * state['exp_avg_sq'] + (1 - self.beta2) * (grad - state['exp_avg']) ** 2
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            step_size = current_lr * math.sqrt(bias_correction2) / bias_correction1
            
            # Update parameters
            param.data.mul_(1 - current_lr * self.weight_decay)
            param.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(self.eps), value=-step_size)
        
        return loss


class AdaBound(EngineeringOptimizer):
    """
    AdaBound optimizer with engineering-specific features.
    """
    
    def __init__(self, params, lr: float = 0.001, weight_decay: float = 1e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 final_lr: float = 0.1, gamma: float = 1e-3):
        """
        Initialize AdaBound optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            weight_decay: Weight decay
            beta1: Beta1 for Adam
            beta2: Beta2 for Adam
            eps: Epsilon
            final_lr: Final learning rate
            gamma: Gamma parameter
        """
        super(AdaBound, self).__init__(params, lr, weight_decay, beta1=beta1, beta2=beta2, eps=eps)
        
        self.final_lr = final_lr
        self.gamma = gamma
        
        # Initialize state for each parameter
        for param in self.params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data)
            }
    
    def step(self, closure=None):
        """Perform optimization step."""
        if closure is not None:
            loss = closure()
        else:
            loss = None
        
        # Clip gradients
        self._clip_gradients()
        
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[param]
            
            # Update step count
            state['step'] += 1
            self.step_count += 1
            
            # Decay the first and second moment running average coefficient
            state['exp_avg'] = self.beta1 * state['exp_avg'] + (1 - self.beta1) * grad
            state['exp_avg_sq'] = self.beta2 * state['exp_avg_sq'] + (1 - self.beta2) * grad * grad
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            
            # AdaBound learning rate bounds
            lower_bound = self.final_lr * (1 - 1 / (self.gamma * state['step'] + 1))
            upper_bound = self.final_lr * (1 + 1 / (self.gamma * state['step']))
            
            # Clamp learning rate
            current_lr = torch.clamp(self.lr / math.sqrt(bias_correction2), lower_bound, upper_bound)
            
            # Update parameters
            param.data.mul_(1 - self.lr * self.weight_decay)
            param.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(self.eps), value=-current_lr)
        
        return loss


class AdaMod(EngineeringOptimizer):
    """
    AdaMod optimizer with engineering-specific features.
    """
    
    def __init__(self, params, lr: float = 0.001, weight_decay: float = 1e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 beta3: float = 0.999):
        """
        Initialize AdaMod optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            weight_decay: Weight decay
            beta1: Beta1 for Adam
            beta2: Beta2 for Adam
            eps: Epsilon
            beta3: Beta3 for AdaMod
        """
        super(AdaMod, self).__init__(params, lr, weight_decay, beta1=beta1, beta2=beta2, eps=eps)
        
        self.beta3 = beta3
        
        # Initialize state for each parameter
        for param in self.params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data),
                'exp_avg_mod': torch.zeros_like(param.data)
            }
    
    def step(self, closure=None):
        """Perform optimization step."""
        if closure is not None:
            loss = closure()
        else:
            loss = None
        
        # Clip gradients
        self._clip_gradients()
        
        # Adaptive learning rate
        current_lr = self._adaptive_learning_rate()
        
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[param]
            
            # Update step count
            state['step'] += 1
            self.step_count += 1
            
            # Decay the first and second moment running average coefficient
            state['exp_avg'] = self.beta1 * state['exp_avg'] + (1 - self.beta1) * grad
            state['exp_avg_sq'] = self.beta2 * state['exp_avg_sq'] + (1 - self.beta2) * grad * grad
            
            # AdaMod modification
            mod = torch.abs(grad - state['exp_avg'])
            state['exp_avg_mod'] = self.beta3 * state['exp_avg_mod'] + (1 - self.beta3) * mod
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            bias_correction3 = 1 - self.beta3 ** state['step']
            
            # Compute effective learning rate
            effective_lr = current_lr * math.sqrt(bias_correction2) / bias_correction1
            effective_lr = effective_lr / (state['exp_avg_mod'] / bias_correction3 + self.eps)
            
            # Update parameters
            param.data.mul_(1 - current_lr * self.weight_decay)
            param.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(self.eps), value=-effective_lr)
        
        return loss
