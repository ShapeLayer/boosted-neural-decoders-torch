"""AWGN and failure-vector input; NumPy RandomState preserves legacy streams."""
from pathlib import Path
import numpy as np
import torch
from .quantization import QUANTIZERS


def code_rate(variable_nodes, check_nodes, lifting_factor, puncturing=(0, 0), shortening=(0, 0), legacy=True):
    length = variable_nodes * lifting_factor
    for start, end in (puncturing, shortening):
        if (start, end) != (0, 0) and not 1 <= start <= end <= length:
            raise ValueError('Puncturing/shortening must be (0,0) or one-based inclusive intervals')
    punctured = puncturing[1] - puncturing[0] + 1 if legacy or puncturing[0] > 0 else 0
    shortened = shortening[1] - shortening[0] + 1 if legacy or shortening[0] > 0 else 0
    transmitted = length - punctured - shortened
    information = (variable_nodes - check_nodes) * lifting_factor - shortened
    if transmitted <= 0 or not 0 < information <= transmitted:
        raise ValueError('Invalid effective code rate')
    return information / transmitted


class AWGNSampler:
    def __init__(self, graph, decoder_config, experiment_config):
        self.graph = graph
        self.decoder_config = decoder_config
        self.experiment_config = experiment_config
        self.word_random = np.random.RandomState(2042 + experiment_config.seed)
        self.noise_random = np.random.RandomState(1074 + experiment_config.seed)
        self.rate = code_rate(graph.variable_nodes, graph.check_nodes, graph.lifting_factor,
                              (experiment_config.puncturing_start, experiment_config.puncturing_end),
                              (experiment_config.shortening_start, experiment_config.shortening_end),
                              experiment_config.legacy_code_rate)

    def sample(self, batch_size, snr_db=None, generator_matrix=None):
        settings, decoder = self.experiment_config, self.decoder_config
        snrs = settings.signal_to_noise_ratios_db if snr_db is None else snr_db
        if batch_size < 1 or len(snrs) == 0:
            raise ValueError('A batch and at least one SNR are required')
        sigmas = np.sqrt(1 / (2 * 10 ** (np.asarray(snrs, dtype=float) / 10) * self.rate))
        received, labels = [], []
        length = self.graph.variable_nodes * self.graph.lifting_factor
        for index in range(batch_size):
            sigma = sigmas[index % len(sigmas)]
            if generator_matrix is None:
                bits = 0 * self.word_random.randint(0, 2, size=(1, length))
            else:
                matrix = np.asarray(generator_matrix)
                if matrix.ndim != 2 or matrix.shape[1] != length:
                    raise ValueError('Generator matrix has incompatible shape')
                information = self.word_random.randint(0, 2, size=(1, matrix.shape[0]))
                bits = information @ matrix % 2
            noisy_symbols = self.noise_random.normal(0, 1, bits.shape) * sigma + (-1) ** (1 - bits)
            llrs = 2 * noisy_symbols / sigma ** 2
            if decoder.decoding_algorithm == 'quantized_min_sum':
                step, limit = QUANTIZERS[decoder.quantization_bits]
                llrs = np.clip(np.round(llrs / step) * step, -limit, limit)
            if settings.puncturing_start > 0:
                llrs[:, settings.puncturing_start - 1:settings.puncturing_end] = (
                    0.001 if decoder.decoding_algorithm == 'sum_product' else 0)
            if settings.shortening_start > 0:
                llrs[:, settings.shortening_start - 1:settings.shortening_end] = -decoder.llr_clip
            received.append(llrs)
            labels.append(bits)
        return (torch.as_tensor(np.concatenate(received).reshape(batch_size, self.graph.variable_nodes,
                                                                self.graph.lifting_factor), dtype=torch.float32),
                torch.as_tensor(np.concatenate(labels), dtype=torch.int64))


class UncorrectedDataset:
    def __init__(self, path, graph, samples=None):
        rows = np.loadtxt(path, dtype=np.float32, ndmin=2)
        length = graph.variable_nodes * graph.lifting_factor
        if rows.shape[1] != length + 3 or not np.isfinite(rows).all():
            raise ValueError(f'{path}: expected three metadata columns and {length} finite LLR values')
        if samples is not None and len(rows) < samples:
            raise ValueError(f'{path}: requested {samples} rows, found {len(rows)}')
        self.channel_llrs = -torch.from_numpy(rows[:samples, 3:].copy()).reshape(-1, graph.variable_nodes, graph.lifting_factor)
        self.labels = torch.zeros(len(self.channel_llrs), length, dtype=torch.int64)

    def batch(self, batch_index, batch_size):
        start = batch_index * batch_size
        return self.channel_llrs[start:start + batch_size], self.labels[start:start + batch_size]


def write_uncorrected(path, channel_llrs, uncorrected_frames):
    selected = -channel_llrs[uncorrected_frames].flatten(1).detach().cpu().numpy()
    if not len(selected):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as output:
        np.savetxt(output, np.concatenate((np.zeros((len(selected), 3)), selected), axis=1), fmt='%.1f', delimiter='\t')
