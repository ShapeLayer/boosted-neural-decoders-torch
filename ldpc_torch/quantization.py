"""Legacy quantizer forward values and clipped straight-through derivatives."""
import torch

QUANTIZERS = {6: (1.0, 15.5), 5: (0.5, 7.5), -5: (1.0, 15.0),
              4: (1.0, 7.0), 3: (2.0, 6.0)}


class _InclusiveClip(torch.autograd.Function):
    @staticmethod
    def forward(context, values, lower, upper):
        context.save_for_backward(values)
        context.lower, context.upper = lower, upper
        return values.clamp(lower, upper)

    @staticmethod
    def backward(context, gradient):
        (values,) = context.saved_tensors
        return gradient * ((values >= context.lower) & (values <= context.upper)), None, None


def clip_llr(values, lower, upper):
    """TensorFlow clip_by_value keeps gradient 1 at both exact boundaries."""
    return _InclusiveClip.apply(values, lower, upper)


def quantize_llr(values: torch.Tensor, bits: int) -> torch.Tensor:
    step, limit = QUANTIZERS[bits]
    rounded = (torch.round(values / step) * step).clamp(-limit, limit)
    surrogate = clip_llr(values, -limit, limit)
    return surrogate + (rounded - surrogate).detach()


def differentiable_sign(values: torch.Tensor) -> torch.Tensor:
    surrogate = 2 * torch.sigmoid(values) - 1
    return surrogate + (torch.sign(values) - surrogate).detach()
