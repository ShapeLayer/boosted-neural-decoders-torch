from dataclasses import dataclass
import torch


@dataclass
class ErrorMetrics:
    bit_error_rate_last: float
    frame_error_rate_last: float
    frame_error_rate: float
    uncorrected_frames: torch.Tensor
    bit_errors_per_frame: torch.Tensor


def error_metrics(iteration_llrs, labels, legacy_bit_error_rate=True):
    """FER is failure at *every* iteration; FER_last uses the last iteration.

    Legacy BER uses signed error cancellation and the full label width. Keep it
    by default for reproducibility; opt out for standard BER on nonzero words.
    """
    decisions = (iteration_llrs >= 0).to(torch.int64)
    differences = decisions - labels[None, :, :decisions.shape[-1]]
    frame_errors = differences.abs().sum(-1) > 0
    uncorrected = frame_errors.all(0)
    signed_errors = differences[-1].sum(-1)
    if legacy_bit_error_rate:
        bit_error_rate = signed_errors.sum().abs().float() / labels.numel()
    else:
        bit_error_rate = differences[-1].abs().float().mean()
    return ErrorMetrics(bit_error_rate.item(), frame_errors[-1].float().mean().item(),
                        uncorrected.float().mean().item(), uncorrected, signed_errors)
