"""Runtime device selection shared by every experiment entry point."""
import torch


def accelerator_availability() -> dict[str, bool]:
    """Check both backends, including CPU-only PyTorch builds."""
    mps_backend = getattr(torch.backends, 'mps', None)
    return {
        'cuda': torch.cuda.is_available(),
        'mps': mps_backend is not None and mps_backend.is_available(),
    }


def resolve_device(requested: str = 'auto', availability=None) -> torch.device:
    availability = accelerator_availability() if availability is None else availability
    if requested == 'auto':
        return torch.device('cuda' if availability['cuda'] else 'mps' if availability['mps'] else 'cpu')
    try:
        selected = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f'Invalid device {requested!r}; use auto, cpu, cuda[:index], or mps') from error
    if selected.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError(f'Unsupported device: {requested}')
    if selected.type in availability and not availability[selected.type]:
        raise RuntimeError(f'Requested device {requested!r} is unavailable; use --device auto or --device cpu')
    if selected.type == 'cuda' and selected.index is not None and selected.index >= torch.cuda.device_count():
        raise ValueError(f'CUDA device index {selected.index} is out of range')
    if selected.type == 'mps' and selected.index not in (None, 0):
        raise ValueError('MPS supports only device index 0')
    return selected
