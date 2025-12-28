"""Explicit experiment settings; iteration intervals are zero-based, end-exclusive."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DecoderConfig:
    lifting_factor: int = 24
    iterations: int = 20
    decoding_algorithm: str = 'quantized_min_sum'
    check_weight_sharing: int = 3
    unsatisfied_weight_sharing: int = 0
    channel_weight_sharing: int = 3
    frozen_iterations: int = 0
    quantization_bits: int = 5
    llr_clip: float = 20.0
    initial_check_weight: float = 1.0
    initial_channel_weight: float = 1.0
    minimum_weight: float = 0.0
    maximum_weight: float = 2.0
    target_variable_nodes: int = 0

    @property
    def sharing(self):
        return (self.check_weight_sharing, self.unsatisfied_weight_sharing,
                self.channel_weight_sharing)

    def __post_init__(self):
        if self.lifting_factor < 1 or self.iterations < 1:
            raise ValueError('lifting_factor and iterations must be positive')
        if not 0 <= self.frozen_iterations <= self.iterations:
            raise ValueError('frozen_iterations must be within the decoder')
        if self.decoding_algorithm not in {'sum_product', 'min_sum', 'quantized_min_sum', 'legacy_min_sum'}:
            raise ValueError('Unknown decoding_algorithm')
        if self.quantization_bits not in (6, 5, -5, 4, 3):
            raise ValueError('Supported quantization_bits: 6, 5, -5, 4, 3')
        if any(mode not in range(6) for mode in self.sharing):
            raise ValueError('Weight sharing must be between 0 and 5')
        if self.channel_weight_sharing in (1, 4):
            raise ValueError('Channel weights support node/scalar sharing, not edge sharing')
        if self.unsatisfied_weight_sharing not in (0, self.check_weight_sharing):
            raise ValueError('Unsatisfied and satisfied check weights must use matching sharing')
        if self.llr_clip <= 0 or self.minimum_weight > self.maximum_weight:
            raise ValueError('Invalid clipping bounds')
        for initial in (self.initial_check_weight, self.initial_channel_weight):
            if initial != -1 and not self.minimum_weight <= initial <= self.maximum_weight:
                raise ValueError('Initial weights must lie within the bounds, or be -1 for random initialization')


@dataclass(frozen=True)
class ExperimentConfig:
    graph_name: str = 'wman_N0576_R34_z24'
    data_directory: str = str(Path(__file__).resolve().parents[1])
    output_directory: str = 'runs/base'
    sampling_mode: str = 'awgn'
    signal_to_noise_ratios_db: tuple = (2, 2.5, 3, 3.5, 4)
    batch_size: int = 20
    training_samples: int = 10000
    validation_samples: int = 10000
    test_samples: int = 0
    epochs: int = 200
    iteration_block_size: int = 20
    retrain_previous_iterations: int = 0
    loss_function: str = 'frame_error'
    loss_decay: float = 0.0
    loss_decay_multiplier: float = 0.0
    loss_decay_interval: int = 0
    learning_rate: float = 0.001
    learning_rate_multiplier: float = 0.0
    learning_rate_interval: int = 0
    selection_metric: str = 'frame_error_rate_last'
    seed: int = 2
    device: str = 'auto'
    puncturing_start: int = 0
    puncturing_end: int = 0
    shortening_start: int = 0
    shortening_end: int = 0
    legacy_code_rate: bool = True
    systematic_only: bool = False
    initial_weights: str | None = None
    frozen_weights: str | None = None

    def __post_init__(self):
        if self.sampling_mode not in ('awgn', 'uncorrected', 'collect'):
            raise ValueError('Unknown sampling_mode')
        if self.batch_size < 1 or self.iteration_block_size < 1 or self.epochs < 0:
            raise ValueError('Invalid batch size, block size, or epochs')
        if self.retrain_previous_iterations < 0 or self.loss_decay < 0 or self.learning_rate <= 0:
            raise ValueError('Invalid training schedule')
        if not self.signal_to_noise_ratios_db:
            raise ValueError('At least one SNR is required')
        if self.sampling_mode == 'collect' and len(self.signal_to_noise_ratios_db) != 1:
            raise ValueError('Collection requires exactly one SNR')
        if self.loss_function not in ('binary_cross_entropy', 'soft_bit_error', 'frame_error'):
            raise ValueError('Unknown loss_function')
        if self.selection_metric not in ('bit_error_rate_last', 'frame_error_rate_last', 'frame_error_rate', 'loss'):
            raise ValueError('Unknown selection_metric')
        for count in (self.training_samples, self.validation_samples, self.test_samples):
            if count < 0:
                raise ValueError('Sample counts cannot be negative')
