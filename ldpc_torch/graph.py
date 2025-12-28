"""QC graph topology without quadratic lifting or check-update matrices."""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class Protograph(nn.Module):
    def __init__(self, shifts, lifting_factor: int):
        super().__init__()
        if lifting_factor < 1:
            raise ValueError('lifting_factor must be positive')
        shifts = torch.as_tensor(shifts, dtype=torch.long)
        if shifts.ndim != 2 or torch.any(shifts < -1):
            raise ValueError('A protograph must be a matrix of -1 or nonnegative shifts')
        self.check_nodes, self.variable_nodes = shifts.shape
        self.lifting_factor = lifting_factor
        check_indices, variable_indices = torch.where(shifts >= 0)
        self.edges = len(check_indices)
        if not self.edges:
            raise ValueError('A protograph must contain at least one edge')
        self.register_buffer('shifts', shifts)
        self.register_buffer('edge_checks', check_indices)
        self.register_buffer('edge_variables', variable_indices)
        edge_shifts = shifts[check_indices, variable_indices] % lifting_factor
        positions = torch.arange(lifting_factor)[None, :]
        self.register_buffer('to_check_positions', (positions + edge_shifts[:, None]) % lifting_factor)
        self.register_buffer('to_variable_positions', (positions - edge_shifts[:, None]) % lifting_factor)
        for name, indices in [('check_neighbors', check_indices), ('variable_neighbors', variable_indices)]:
            neighborhoods = [torch.where((indices == indices[edge]) & (torch.arange(self.edges) != edge))[0]
                             for edge in range(self.edges)]
            # One extra absent entry reproduces the legacy zero-mask sentinel.
            width = max(len(neighbors) for neighbors in neighborhoods) + 1
            table = torch.full((self.edges, width), self.edges, dtype=torch.long)
            for edge, neighbors in enumerate(neighborhoods):
                table[edge, :len(neighbors)] = neighbors
            self.register_buffer(name, table)
        output_incidence = torch.zeros(self.edges, self.variable_nodes)
        output_incidence[torch.arange(self.edges), variable_indices] = 1
        self.register_buffer('output_incidence', output_incidence)

    @classmethod
    def from_file(cls, path: str | Path, lifting_factor: int):
        return cls(np.loadtxt(path, dtype=np.int64, ndmin=2), lifting_factor)

    def lift(self, messages, inverse=False):
        positions = self.to_variable_positions if inverse else self.to_check_positions
        return messages.gather(2, positions[None].expand(messages.shape[0], -1, -1))

    def neighbors(self, messages, table):
        padded = torch.cat((messages, messages.new_zeros(messages.shape[0], 1, self.lifting_factor)), dim=1)
        return padded[:, table, :]
