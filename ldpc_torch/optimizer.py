"""TensorFlow v1 Adam uses epsilon before variance bias correction."""
import math
import torch


class LegacyAdam(torch.optim.Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state['step'] = 0
                    state['first_moment'] = torch.zeros_like(parameter)
                    state['second_moment'] = torch.zeros_like(parameter)
                state['step'] += 1
                first, second = state['first_moment'], state['second_moment']
                first.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                corrected_rate = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                parameter.addcdiv_(first, second.sqrt().add_(group['eps']), value=-corrected_rate)
        return loss
