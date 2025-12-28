import torch
from torch.nn import functional as functional
from .quantization import differentiable_sign


def decoding_loss(iteration_llrs, labels, start_iteration=0, loss_function='frame_error', decay=0.0):
    """Weighted mean over [start_iteration, end); legacy eta=0 means last only."""
    if not 0 <= start_iteration < len(iteration_llrs) or decay < 0:
        raise ValueError('Invalid loss window or decay')
    labels = labels[:, :iteration_llrs.shape[-1]]
    if loss_function != 'binary_cross_entropy' and torch.any(labels != 0):
        raise ValueError('Soft BER and FER surrogate require all-zero codewords')
    weighted_losses = []
    coefficients = []
    for iteration in range(len(iteration_llrs) - 1, start_iteration - 1, -1):
        logits = iteration_llrs[iteration]
        if loss_function == 'binary_cross_entropy':
            loss = functional.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
        elif loss_function == 'soft_bit_error':
            loss = logits.sigmoid().mean()
        elif loss_function == 'frame_error':
            loss = (0.5 * (1 - differentiable_sign((-logits).amin(1)))).mean()
        else:
            raise ValueError('Unknown loss function')
        coefficient = decay ** (len(iteration_llrs) - 1 - iteration)
        weighted_losses.append(loss * coefficient)
        coefficients.append(coefficient)
    return torch.stack(weighted_losses).sum() / sum(coefficients)
