"""Audit post updates, frozen prefix, legacy weight import, and failure collection."""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ldpc_torch import ExperimentConfig, NeuralLDPCDecoder, Protograph, decoding_loss, error_metrics
from ldpc_torch.coding import parity_check_matrix, generator_matrix
from ldpc_torch.data import AWGNSampler, UncorrectedDataset, write_uncorrected
from ldpc_torch.optimizer import LegacyAdam
from ldpc_torch.weights import load_checkpoint, load_legacy_weights, save_legacy_weights


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def weights(model):
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def maximum_difference(first, second):
    return max(float(np.abs(a - b).max()) for a, b in zip(first, second))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--post-steps', type=int, default=32)
    parser.add_argument('--mixed-training-steps', type=int, default=8)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    oracle_module = module('oracle', ROOT / 'tests/reference/tensorflow_oracle.py')
    legacy_print = module('legacy_print', ROOT / 'tests/reference/legacy_print.py')
    base, _ = load_checkpoint(ROOT / 'runs/base/best_decoder_end20.pt')
    config = replace(base.config, iterations=30, frozen_iterations=20, unsatisfied_weight_sharing=3)
    post = NeuralLDPCDecoder(Protograph(base.graph.shifts, config.lifting_factor), config)
    base_path = ROOT / 'runs/base/best_weights_end20.txt'
    load_legacy_weights(post, base_path, end_iteration=20)
    report = {'device': 'cpu', 'batch_size': 20, 'post_steps': [], 'mixed_bce_steps': []}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        args.report.write_text(json.dumps(report, indent=2))
    # Exercise the original weight_init, including its positional three-group assumption.
    with tempfile.TemporaryDirectory(prefix='ldpc-legacy-weight-import-') as temporary:
        directory = Path(temporary)
        (directory / 'Weights').mkdir()
        fixed_file = directory / 'Weights/audit_Opt_Weight_End20.txt'
        def original_import():
            with oracle_module.tf.Graph().as_default():
                old_directory = Path.cwd()
                try:
                    os.chdir(directory)
                    net = oracle_module.legacy.weight_init({}, 0, 'audit', 30, 20, config.sharing,
                        post.graph.edges, post.graph.check_nodes, post.graph.variable_nodes,
                        0., 2., 1., 1., 30, 20)
                finally:
                    os.chdir(old_directory)
                with oracle_module.tf.Session() as session:
                    session.run(oracle_module.tf.global_variables_initializer())
                    return session.run([net[f'var_{group}_{iteration}'] for group in range(3) for iteration in range(30)])
        shutil.copyfile(base_path, fixed_file)
        try:
            raw = original_import()
            report['legacy_raw_base_to_post_import'] = {'loads': True, 'max_weight_difference': maximum_difference(weights(post), raw)}
        except (ValueError, IndexError, TypeError) as error:
            report['legacy_raw_base_to_post_import'] = {'loads': False, 'error_type': type(error).__name__, 'error': str(error)[:400]}
        save_legacy_weights(post, fixed_file, iterations=20)
        expanded = original_import()
        report['legacy_expanded_base_to_post_import_max_difference'] = maximum_difference(weights(post), expanded)
        assert report['legacy_expanded_base_to_post_import_max_difference'] == 0
    parameters = post.set_training_window(20, 30)
    frozen_before = [p.detach().clone() for p in post.parameters() if not p.requires_grad]
    reference = oracle_module.TensorFlowOracle(post, 20)
    optimizer = LegacyAdam(parameters)
    dataset = UncorrectedDataset(ROOT / 'uncorrected_samples/[Uncor]_wman_N0576_R34_z24.txt', post.graph)
    for step in range(args.post_steps):
        channel, labels = dataset.batch(step, 20)
        expected, ref_loss, ref_gradients = reference.evaluate(channel.numpy(), gradients=True)
        optimizer.zero_grad(set_to_none=True)
        actual = post(channel).target_llrs
        loss = decoding_loss(actual, labels, 20)
        loss.backward()
        row = {'step': step + 1, 'torch_loss': loss.item(), 'tensorflow_loss': float(ref_loss),
               'posterior_max_abs_difference': float(np.abs(actual.detach().flatten(0, 1).numpy() - expected).max()),
               'decision_mismatches': int(np.count_nonzero((actual.detach().flatten(0, 1).numpy() >= 0) != (expected >= 0))),
               'gradient_max_abs_difference': maximum_difference([p.grad.numpy() for p in parameters], ref_gradients)}
        optimizer.step()
        post.constrain_weights()
        ref_weights = reference.step(channel.numpy())
        row['updated_weight_max_abs_difference'] = maximum_difference(weights(post), ref_weights)
        report['post_steps'].append(row)
        if (step + 1) % 8 == 0:
            persist()
            print('post', json.dumps(row), flush=True)
    report['frozen_prefix_max_change'] = max((before - after).abs().max().item() for before, after in
        zip(frozen_before, [p.detach() for p in post.parameters() if not p.requires_grad]))
    reference.close()
    assert report['frozen_prefix_max_change'] == 0
    # Collect failures with both implementations and compare file content + reload signs.
    reference = oracle_module.TensorFlowOracle(base, 20)
    sampler = AWGNSampler(base.graph, base.config, ExperimentConfig(seed=389))
    collected = 0
    with tempfile.TemporaryDirectory(prefix='ldpc-collection-parity-') as temporary:
        directory = Path(temporary)
        current_file = directory / 'current.txt'
        for _ in range(4):
            channel, labels = sampler.sample(20, (2.5,))
            expected, _ = reference.evaluate(channel.numpy())
            with torch.no_grad():
                actual = base(channel).target_llrs
            expected_metrics = legacy_print.calc_ber_fer(expected, 20, labels.numpy(), 20)
            actual_metrics = error_metrics(actual, labels)
            np.testing.assert_array_equal(actual_metrics.uncorrected_frames.numpy(), expected_metrics[3])
            np.testing.assert_allclose([actual_metrics.bit_error_rate_last, actual_metrics.frame_error_rate_last,
                                       actual_metrics.frame_error_rate], expected_metrics[:3], atol=1e-7)
            write_uncorrected(current_file, channel, actual_metrics.uncorrected_frames)
            old_directory = Path.cwd()
            try:
                os.chdir(directory)
                legacy_print.write_uncor_file(expected_metrics[3], channel.numpy(), 576)
            finally:
                os.chdir(old_directory)
            collected += int(actual_metrics.uncorrected_frames.sum())
        assert collected > 20
        report['failure_collection'] = {'candidate_frames': 80, 'collected_frames': collected,
                                       'byte_identical_files': current_file.read_bytes() == (directory / 'Uncor.txt').read_bytes()}
        restored = UncorrectedDataset(current_file, base.graph)
        raw = np.loadtxt(directory / 'Uncor.txt', ndmin=2)
        expected_channel, expected_labels = legacy_print.read_uncor_llr(raw[:, 3:], np.zeros_like(raw[:, 3:]), 0, 20, 24, 24)
        channel, labels = restored.batch(0, 20)
        np.testing.assert_array_equal(channel.numpy(), expected_channel)
        np.testing.assert_array_equal(labels.numpy(), expected_labels)
        report['failure_collection']['reloaded_samples_exact_match'] = True
    reference.close()
    # Label-aware mixed-word training is tested with BCE, never the zero-only FER surrogate.
    base, _ = load_checkpoint(ROOT / 'runs/base/best_decoder_end20.pt')
    reference = oracle_module.TensorFlowOracle(base, 20, loss_function='binary_cross_entropy')
    optimizer = LegacyAdam(base.parameters())
    generator = generator_matrix(parity_check_matrix(base.graph))
    sampler = AWGNSampler(base.graph, base.config, ExperimentConfig(seed=471))
    for step in range(args.mixed_training_steps):
        channel, labels = sampler.sample(20, generator_matrix=generator)
        expected, ref_loss, ref_gradients = reference.evaluate(channel.numpy(), gradients=True, labels=labels.numpy())
        optimizer.zero_grad(set_to_none=True)
        actual = base(channel).target_llrs
        loss = decoding_loss(actual, labels, loss_function='binary_cross_entropy')
        loss.backward()
        row = {'step': step + 1, 'torch_loss': loss.item(), 'tensorflow_loss': float(ref_loss),
               'posterior_max_abs_difference': float(np.abs(actual.detach().flatten(0, 1).numpy() - expected).max()),
               'gradient_max_abs_difference': maximum_difference([p.grad.numpy() for p in base.parameters()], ref_gradients)}
        optimizer.step()
        base.constrain_weights()
        row['updated_weight_max_abs_difference'] = maximum_difference(weights(base), reference.step(channel.numpy(), labels=labels.numpy()))
        report['mixed_bce_steps'].append(row)
    reference.close()
    persist()
    print('complete', json.dumps({k: v for k, v in report.items() if k not in ('post_steps', 'mixed_bce_steps')}), flush=True)


if __name__ == '__main__':
    main()
