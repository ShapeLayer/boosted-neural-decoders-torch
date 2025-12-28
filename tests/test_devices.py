import pytest
import torch
from ldpc_torch.devices import accelerator_availability, resolve_device


@pytest.mark.parametrize('cuda,mps,expected', [(True, True, 'cuda'), (True, False, 'cuda'),
                                              (False, True, 'mps'), (False, False, 'cpu')])
def test_auto_checks_both_backends(monkeypatch, cuda, mps, expected):
    checked = []
    def cuda_available():
        checked.append('cuda')
        return cuda
    def mps_available():
        checked.append('mps')
        return mps
    monkeypatch.setattr(torch.cuda, 'is_available', cuda_available)
    monkeypatch.setattr(torch.backends.mps, 'is_available', mps_available)
    assert resolve_device().type == expected
    assert checked == ['cuda', 'mps']


def test_cpu_override():
    assert str(resolve_device('cpu', {'cuda': True, 'mps': True})) == 'cpu'


@pytest.mark.parametrize('requested', ['cuda', 'cuda:0', 'mps'])
def test_unavailable_explicit_accelerator(requested):
    with pytest.raises(RuntimeError, match='unavailable'):
        resolve_device(requested, {'cuda': False, 'mps': False})


def test_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    assert str(resolve_device('cuda:1', {'cuda': True, 'mps': False})) == 'cuda:1'
    with pytest.raises(ValueError, match='out of range'):
        resolve_device('cuda:2', {'cuda': True, 'mps': False})


def test_missing_mps_backend(monkeypatch):
    monkeypatch.delattr(torch.backends, 'mps')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert accelerator_availability() == {'cuda': False, 'mps': False}


def test_explicit_mps_overrides_cuda():
    assert resolve_device('mps', {'cuda': True, 'mps': True}).type == 'mps'
