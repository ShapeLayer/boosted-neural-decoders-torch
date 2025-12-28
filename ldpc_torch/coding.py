"""Binary linear-code helpers for validation on nonzero codewords.

The nullspace encoder is for correctness experiments, not a high-throughput
standards-specific systematic encoder. Columns retain the decoder's bit order.
"""
import numpy as np


def parity_check_matrix(graph):
    lifting = graph.lifting_factor
    shifts = graph.shifts.detach().cpu().numpy()
    parity = np.zeros((graph.check_nodes * lifting, graph.variable_nodes * lifting), dtype=np.uint8)
    positions = np.arange(lifting)
    for check, variable in zip(*np.where(shifts >= 0)):
        parity[check * lifting + positions,
               variable * lifting + (positions + shifts[check, variable]) % lifting] = 1
    return parity


def generator_matrix(parity):
    """Return a full-rank basis of ker(H) over GF(2), one basis vector per row."""
    parity = np.asarray(parity)
    if parity.ndim != 2 or not np.isin(parity, (0, 1)).all():
        raise ValueError('Parity-check matrix must be a binary two-dimensional array')
    reduced = parity.astype(np.uint8, copy=True)
    pivots = []
    rank = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[rank:, column])
        if not len(candidates):
            continue
        selected = rank + candidates[0]
        reduced[[rank, selected]] = reduced[[selected, rank]]
        other_rows = np.flatnonzero(reduced[:, column])
        other_rows = other_rows[other_rows != rank]
        reduced[other_rows] ^= reduced[rank]
        pivots.append(column)
        rank += 1
        if rank == reduced.shape[0]:
            break
    free_columns = np.setdiff1d(np.arange(reduced.shape[1]), pivots)
    if not len(free_columns):
        raise ValueError('This code has no nonzero information dimension')
    generator = np.zeros((len(free_columns), reduced.shape[1]), dtype=np.uint8)
    generator[np.arange(len(free_columns)), free_columns] = 1
    generator[:, pivots] = reduced[:rank, free_columns].T
    if np.any((generator.astype(np.int64) @ parity.T) % 2):
        raise AssertionError('Generated basis violates the parity checks')
    return generator
