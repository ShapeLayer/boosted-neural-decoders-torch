"""Legacy text weight interoperability and portable native checkpoints."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .config import DecoderConfig
from .graph import Protograph
from .decoder import NeuralLDPCDecoder


def read_legacy_weights(path):
    lines = Path(path).read_text().splitlines()
    sharing = tuple(int(value) for value in lines[0].split())
    if len(sharing) != 3 or any(mode not in range(6) for mode in sharing):
        raise ValueError('Invalid legacy weight sharing header')
    blocks, current = [], []
    for line in lines[1:] + ['']:
        if line.strip():
            values = np.asarray([float(value) for value in line.split()], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('Weights must be finite')
            current.append(torch.from_numpy(values))
        elif current:
            blocks.append(current)
            current = []
    if len(blocks) != sum(mode != 0 for mode in sharing):
        raise ValueError('Weight block count does not match sharing header')
    groups, block_index = [], 0
    for mode in sharing:
        groups.append(blocks[block_index] if mode else [])
        block_index += bool(mode)
    return sharing, groups


@torch.no_grad()
def load_legacy_weights(model, path, start_iteration=0, end_iteration=None):
    """Copy matching semantic groups, never positional rows across groups.

    A newly enabled UCN group copies the satisfied group if absent in source.
    Scalar/check-node weights can be expanded when transferring sharing modes.
    """
    source_sharing, source_groups = read_legacy_weights(path)
    end_iteration = model.config.iterations if end_iteration is None else end_iteration
    if not 0 <= start_iteration <= end_iteration <= model.config.iterations:
        raise ValueError('Invalid weight import interval')
    for group_index, (mode, parameters) in enumerate(zip(model.config.sharing, model.weight_groups)):
        source_index = 0 if group_index == 1 and not source_groups[1] else group_index
        source = source_groups[source_index]
        for iteration, parameter in enumerate(parameters):
            if not start_iteration <= iteration < end_iteration:
                continue
            if iteration >= len(source):
                raise ValueError(f'{path}: missing group {source_index}, iteration {iteration}')
            values = source[iteration].to(parameter)
            if values.numel() == 1:
                values = values.expand_as(parameter)
            elif source_sharing[source_index] in (2, 5) and mode in (1, 4) and group_index < 2:
                values = values[model.graph.edge_checks]
            if values.shape != parameter.shape:
                raise ValueError(f'{path}: incompatible weight shape in group {group_index}')
            parameter.copy_(values)


@torch.no_grad()
def save_legacy_weights(model, path, iterations=None):
    iterations = model.config.iterations if iterations is None else iterations
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as output:
        output.write(' '.join(map(str, model.config.sharing)) + '\n\n')
        for mode, parameters in zip(model.config.sharing, model.weight_groups):
            if not mode:
                continue
            for iteration in range(iterations):
                index = min(iteration, model.config.frozen_iterations) if mode in (4, 5) else iteration
                output.write('\t'.join(str(value) for value in parameters[index].detach().cpu().tolist()) + '\n')
            output.write('\n')


def save_checkpoint(model, path, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'decoder_config': asdict(model.config), 'state_dict': model.state_dict(),
                'metadata': metadata}, path)


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = DecoderConfig(**saved['decoder_config'])
    graph = Protograph(saved['state_dict']['graph.shifts'].cpu(), config.lifting_factor)
    model = NeuralLDPCDecoder(graph, config).to(device)
    model.load_state_dict(saved['state_dict'])
    return model, saved['metadata']
