import numpy as np
import pytest
import torch
from ldpc_torch import Protograph
from ldpc_torch.coding import generator_matrix, parity_check_matrix
from ldpc_torch.losses import decoding_loss
from ldpc_torch.metrics import error_metrics


def test_generator_and_lifting_agree():
    graph = Protograph([[0, 1, 2], [1, 0, 0]], 3)
    parity = parity_check_matrix(graph)
    generator = generator_matrix(parity)
    messages = np.random.RandomState(9).randint(0, 2, (40, len(generator)))
    codewords = messages @ generator % 2
    assert not np.any(codewords @ parity.T % 2)
    lifted = graph.lift(torch.tensor(codewords).reshape(-1, 3, 3)[:, graph.edge_variables])
    for check in range(graph.check_nodes):
        assert torch.all(lifted[:, graph.edge_checks == check].sum(1) % 2 == 0)
    assert generator.shape[0] > 0
    assert np.any(codewords)


def test_redundant_parity_checks():
    parity = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    generator = generator_matrix(parity)
    assert generator.shape == (2, 3)
    assert not (generator @ parity.T % 2).any()


def test_nonzero_words_use_real_ber_and_label_aware_loss():
    labels = torch.tensor([[0, 1, 0, 1]])
    swapped = torch.tensor([[[1., -1., -1., 1.]]])
    assert error_metrics(swapped, labels, legacy_bit_error_rate=False).bit_error_rate_last == .5
    assert error_metrics(swapped, labels).bit_error_rate_last == 0  # historical signed cancellation
    for loss in ('frame_error', 'soft_bit_error'):
        with pytest.raises(ValueError, match='all-zero'):
            decoding_loss(swapped, labels, loss_function=loss)
    assert decoding_loss(swapped, labels, loss_function='binary_cross_entropy') > 0
