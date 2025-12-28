"""Trainable flooding decoder. Internal LLRs are log(P(bit=1)/P(bit=0))."""
from dataclasses import dataclass
import torch
from torch import nn
from .config import DecoderConfig
from .graph import Protograph
from .quantization import quantize_llr, clip_llr


@dataclass
class DecoderOutput:
    posterior_llrs: torch.Tensor  # [iteration, batch, variable * lifting]
    target_llrs: torch.Tensor
    check_messages: torch.Tensor  # [batch, edge (check-major), lifting]

    @property
    def hard_decisions(self):
        return self.target_llrs[-1] >= 0


class NeuralLDPCDecoder(nn.Module):
    def __init__(self, graph: Protograph, config: DecoderConfig):
        super().__init__()
        if graph.lifting_factor != config.lifting_factor:
            raise ValueError('Graph and decoder lifting factors differ')
        if not 0 <= config.target_variable_nodes <= graph.variable_nodes:
            raise ValueError('target_variable_nodes exceeds the graph')
        self.graph = graph
        self.config = config
        self.satisfied_check_weights = nn.ParameterList()
        self.unsatisfied_check_weights = nn.ParameterList()
        self.channel_weights = nn.ParameterList()
        for group_index, (mode, parameters) in enumerate(zip(config.sharing, self.weight_groups)):
            if mode == 0:
                continue
            width = (graph.edges if mode in (1, 4) else
                     (graph.variable_nodes if group_index == 2 else graph.check_nodes)
                     if mode in (2, 5) else 1)
            count = min(config.iterations, config.frozen_iterations + 1) if mode in (4, 5) else config.iterations
            initial = config.initial_channel_weight if group_index == 2 else config.initial_check_weight
            for iteration in range(count):
                values = torch.full((width,), float(initial))
                if initial == -1:
                    mean = (config.minimum_weight + config.maximum_weight) / 2
                    nn.init.trunc_normal_(values, mean=mean, std=0.1, a=mean - 0.2, b=mean + 0.2)
                parameters.append(nn.Parameter(values, requires_grad=iteration >= config.frozen_iterations))

    @property
    def weight_groups(self):
        return (self.satisfied_check_weights, self.unsatisfied_check_weights, self.channel_weights)

    def _weight(self, group_index, iteration):
        mode = self.config.sharing[group_index]
        if mode == 0:
            return 1.0
        index = min(iteration, self.config.frozen_iterations) if mode in (4, 5) else iteration
        weights = self.weight_groups[group_index][index]
        if group_index < 2 and mode in (2, 5):
            weights = weights[self.graph.edge_checks]
        return weights[None, :, None]

    def set_training_window(self, start: int, end: int):
        if not 0 <= start < end <= self.config.iterations:
            raise ValueError('Invalid training window')
        for mode, parameters in zip(self.config.sharing, self.weight_groups):
            for index, parameter in enumerate(parameters):
                parameter.requires_grad_(index >= self.config.frozen_iterations and
                                         (mode in (4, 5) or start <= index < end))
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def constrain_weights(self):
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.clamp_(self.config.minimum_weight, self.config.maximum_weight)

    def forward(self, channel_llrs: torch.Tensor, iterations: int | None = None) -> DecoderOutput:
        graph, config = self.graph, self.config
        iterations = config.iterations if iterations is None else iterations
        if not 1 <= iterations <= config.iterations:
            raise ValueError('Requested iterations must be within the decoder')
        if channel_llrs.ndim != 3 or channel_llrs.shape[1:] != (graph.variable_nodes, graph.lifting_factor):
            raise ValueError('channel_llrs must have shape [batch, variable_nodes, lifting_factor]')
        if not channel_llrs.is_floating_point():
            raise ValueError('channel_llrs must be floating point')
        check_messages = channel_llrs.new_zeros(channel_llrs.shape[0], graph.edges, graph.lifting_factor)
        posteriors = []
        quantized = config.decoding_algorithm == 'quantized_min_sum'
        original_channel = quantize_llr(channel_llrs, config.quantization_bits) if quantized else channel_llrs
        for iteration in range(iterations):
            weighted_channel = channel_llrs * self._weight(2, iteration)
            if quantized:
                weighted_channel = quantize_llr(weighted_channel, config.quantization_bits)
            if config.unsatisfied_weight_sharing:
                previous_posterior = weighted_channel if iteration == 0 else posteriors[-1]
                hard_signs = torch.where(previous_posterior < 0, 1.0, -1.0)
                lifted_signs = graph.lift(hard_signs[:, graph.edge_variables, :])
                neighbor_signs = graph.neighbors(lifted_signs, graph.check_neighbors)
                neighbor_signs = torch.where(neighbor_signs == 0, 1.0, neighbor_signs)
                parity_signs = neighbor_signs.prod(2) * lifted_signs
                unsatisfied_checks = graph.lift((parity_signs < 0).to(channel_llrs.dtype), inverse=True)
            else:
                unsatisfied_checks = 0.0
            variable_messages = weighted_channel[:, graph.edge_variables, :] + graph.neighbors(
                check_messages, graph.variable_neighbors).sum(2)
            lifted_messages = graph.lift(variable_messages)
            lifted_messages = (quantize_llr(lifted_messages, config.quantization_bits) if quantized
                               else clip_llr(lifted_messages, -config.llr_clip, config.llr_clip))
            if config.decoding_algorithm in ('min_sum', 'quantized_min_sum'):
                lifted_messages = lifted_messages + 0.0001 * (lifted_messages == 0).to(channel_llrs.dtype)
            neighbor_messages = graph.neighbors(lifted_messages, graph.check_neighbors)
            if config.decoding_algorithm == 'sum_product':
                hyperbolic_messages = torch.tanh(-0.5 * neighbor_messages)
                # The original also replaces connected exact zeros with 1.
                hyperbolic_messages = hyperbolic_messages + (hyperbolic_messages == 0).to(channel_llrs.dtype)
                product = clip_llr(hyperbolic_messages.prod(2), -1 + 1e-7, 1 - 1e-7)
                updated_messages = -2 * torch.atanh(product)
            else:
                magnitudes = neighbor_messages.abs() + 10000 * (neighbor_messages == 0).to(channel_llrs.dtype)
                minimum = magnitudes.amin(2)
                minimum = minimum - 0.0001 * (minimum <= 0.0001).to(channel_llrs.dtype)
                signs = 1 - 2 * (neighbor_messages > 0).to(channel_llrs.dtype)
                updated_messages = -minimum * signs.prod(2)
            updated_messages = graph.lift(updated_messages, inverse=True)
            magnitudes = updated_messages.abs() * self._weight(0, iteration)
            if config.unsatisfied_weight_sharing:
                unsatisfied_magnitudes = updated_messages.abs() * self._weight(1, iteration)
                magnitudes = magnitudes * (1 - unsatisfied_checks) + unsatisfied_magnitudes * unsatisfied_checks
            magnitudes = magnitudes * (magnitudes > 0).to(channel_llrs.dtype)
            magnitudes = (quantize_llr(magnitudes, config.quantization_bits) if quantized
                          else clip_llr(magnitudes, -config.llr_clip, config.llr_clip))
            check_messages = magnitudes * updated_messages.sign()
            summed_messages = (check_messages.transpose(1, 2) @ graph.output_incidence.to(channel_llrs.dtype)).transpose(1, 2)
            posterior = clip_llr(original_channel + summed_messages, -config.llr_clip, config.llr_clip)
            posteriors.append(posterior)
        stacked = torch.stack(posteriors)
        target_count = config.target_variable_nodes or graph.variable_nodes
        return DecoderOutput(stacked.flatten(2), stacked[:, :, :target_count].flatten(2), check_messages)
