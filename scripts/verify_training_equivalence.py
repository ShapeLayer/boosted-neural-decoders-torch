"""Audit real 20-iteration training and saved weights against original TensorFlow.

Run with uv run --extra reference python scripts/verify_training_equivalence.py.
This writes an audit report only; it never updates the user's training outputs.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ldpc_torch import DecoderConfig, ExperimentConfig, NeuralLDPCDecoder, Protograph, decoding_loss
from ldpc_torch.data import AWGNSampler
from ldpc_torch.optimizer import LegacyAdam
from ldpc_torch.weights import load_legacy_weights

spec = importlib.util.spec_from_file_location('tensorflow_oracle', ROOT / 'tests/reference/tensorflow_oracle.py')
oracle_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_module)


def parameter_arrays(model):
    return [parameter.detach().cpu().numpy().copy() for parameter in model.parameters()]


def maximum_difference(actual, expected):
    return max(float(np.max(np.abs(left - right))) for left, right in zip(actual, expected))


def compare_outputs(actual, expected, iterations, batch_size):
    expected = expected.reshape(iterations, batch_size, -1)
    return {'posterior_max_abs_difference': float(np.max(np.abs(actual - expected))),
            'hard_decision_mismatches_all_iterations': int(np.count_nonzero((actual >= 0) != (expected >= 0))),
            'hard_decision_mismatches_last_iteration': int(np.count_nonzero((actual[-1] >= 0) != (expected[-1] >= 0))),
            'frame_error_count_difference_last': int(np.any(actual[-1] >= 0, axis=1).sum()) - int(np.any(expected[-1] >= 0, axis=1).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--frames-per-snr', type=int, default=200)
    parser.add_argument('--weights', type=Path, default=ROOT / 'runs/base/best_weights_end20.txt')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    config = DecoderConfig()
    settings = ExperimentConfig(device=args.device)
    if args.steps < 1 or args.frames_per_snr < settings.batch_size or args.frames_per_snr % settings.batch_size:
        parser.error('steps must be positive; frames-per-snr must be a positive multiple of 20')
    torch.set_num_threads(1)
    model = NeuralLDPCDecoder(Protograph.from_file(ROOT / 'base_graphs/wman_N0576_R34_z24.txt', 24), config).to(args.device)
    reference = oracle_module.TensorFlowOracle(model, settings.batch_size, settings.learning_rate)
    optimizer = LegacyAdam(model.parameters(), lr=settings.learning_rate)
    report = {'torch_version': torch.__version__, 'tensorflow_version': oracle_module.tf.__version__,
              'device': args.device, 'iterations': config.iterations, 'batch_size': settings.batch_size,
              'learning_rate': settings.learning_rate, 'loss': 'frame_error', 'loss_decay': 0,
              'training_steps': [], 'fixed_weights_evaluations': {}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        args.report.write_text(json.dumps(report, indent=2))
    # Reproduce the RNG consumption of epoch 0 validation before epoch 1 training.
    sampler = AWGNSampler(model.graph, config, settings)
    for _ in range(settings.validation_samples // settings.batch_size):
        for snr in settings.signal_to_noise_ratios_db:
            sampler.sample(settings.batch_size, (snr,))
    for step in range(args.steps):
        channel, labels = sampler.sample(settings.batch_size)
        reference_outputs, reference_loss, reference_gradients = reference.evaluate(channel.numpy(), gradients=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(channel.to(args.device))
        loss = decoding_loss(output.target_llrs, labels.to(args.device))
        loss.backward()
        record = {'step': step + 1, 'torch_loss': loss.item(), 'tensorflow_loss': float(reference_loss),
                  'gradient_max_abs_difference': maximum_difference(
                      [parameter.grad.detach().cpu().numpy() for parameter in model.parameters()], reference_gradients)}
        record.update(compare_outputs(output.target_llrs.detach().cpu().numpy(), reference_outputs, config.iterations, settings.batch_size))
        optimizer.step()
        model.constrain_weights()
        updated_reference = reference.step(channel.numpy())
        record['updated_weight_max_abs_difference'] = maximum_difference(parameter_arrays(model), updated_reference)
        report['training_steps'].append(record)
        if step == 0 or (step + 1) % 8 == 0:
            persist()
            print(json.dumps(record), flush=True)
    report['mean_training_loss'] = {
        'torch': float(np.mean([row['torch_loss'] for row in report['training_steps']])),
        'tensorflow': float(np.mean([row['tensorflow_loss'] for row in report['training_steps']]))}
    persist()
    # Equal-parameter tests distinguish forward/backend effects from training drift.
    for label in ('initial', 'trained'):
        if label == 'initial':
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(1.0)
        else:
            load_legacy_weights(model, args.weights)
            report['weights_file'] = str(args.weights.resolve())
            report['weights_sha256'] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
        reference.assign(parameter_arrays(model))
        eval_sampler = AWGNSampler(model.graph, config, replace(settings, seed=12345))
        summaries = []
        for snr in settings.signal_to_noise_ratios_db:
            summary = {'snr_db': snr, 'frames': args.frames_per_snr, 'posterior_max_abs_difference': 0.,
                       'hard_decision_mismatches_all_iterations': 0, 'hard_decision_mismatches_last_iteration': 0,
                       'torch_frame_errors_last': 0, 'tensorflow_frame_errors_last': 0, 'loss_max_abs_difference': 0.}
            for _ in range(args.frames_per_snr // settings.batch_size):
                channel, labels = eval_sampler.sample(settings.batch_size, (snr,))
                ref_outputs, ref_loss = reference.evaluate(channel.numpy())
                with torch.no_grad():
                    output = model(channel.to(args.device))
                    loss = decoding_loss(output.target_llrs, labels.to(args.device)).item()
                actual = output.target_llrs.cpu().numpy()
                differences = compare_outputs(actual, ref_outputs, config.iterations, settings.batch_size)
                summary['posterior_max_abs_difference'] = max(summary['posterior_max_abs_difference'], differences['posterior_max_abs_difference'])
                for name in ('hard_decision_mismatches_all_iterations', 'hard_decision_mismatches_last_iteration'):
                    summary[name] += differences[name]
                summary['torch_frame_errors_last'] += int(np.any(actual[-1] >= 0, axis=1).sum())
                summary['tensorflow_frame_errors_last'] += int(np.any(ref_outputs[-settings.batch_size:] >= 0, axis=1).sum())
                summary['loss_max_abs_difference'] = max(summary['loss_max_abs_difference'], abs(loss - float(ref_loss)))
            summaries.append(summary)
            report['fixed_weights_evaluations'][label] = summaries
            persist()
            print(label, json.dumps(summary), flush=True)
    reference.close()
    persist()


if __name__ == '__main__':
    main()
