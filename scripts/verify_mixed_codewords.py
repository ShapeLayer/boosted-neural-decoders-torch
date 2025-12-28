"""Test saved zero-word-trained decoders on valid random nonzero LDPC words."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ldpc_torch import ExperimentConfig, decoding_loss
from ldpc_torch.coding import parity_check_matrix, generator_matrix
from ldpc_torch.data import AWGNSampler
from ldpc_torch.weights import load_checkpoint


def load_reference(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests/reference' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wilson_interval(errors, samples):
    z = 1.959963984540054
    rate = errors / samples
    denominator = 1 + z * z / samples
    center = (rate + z * z / (2 * samples)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / samples + z * z / (4 * samples * samples)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def counts(posterior, labels):
    decisions = posterior >= 0
    errors = decisions != labels
    return {'bit_errors': int(errors.sum()), 'frame_errors': int(errors.any(1).sum()),
            'predicted_ones': int(decisions.sum()), 'all_zero_output_frames': int((~decisions.any(1)).sum()),
            'zero_bit_errors': int((errors & (labels == 0)).sum()),
            'one_bit_errors': int((errors & (labels == 1)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--frames-per-snr', type=int, default=1000)
    parser.add_argument('--reference-frames-per-snr', type=int, default=100)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    batch_size = 20
    if args.frames_per_snr % batch_size or args.frames_per_snr < batch_size:
        parser.error('frames-per-snr must be a positive multiple of 20')
    if args.reference_frames_per_snr % batch_size or not 0 <= args.reference_frames_per_snr <= args.frames_per_snr:
        parser.error('reference frames must be a multiple of 20 within the sample count')
    torch.set_num_threads(1)
    original_data = load_reference('original_data', 'legacy_print.py')
    oracle_class = load_reference('oracle', 'tensorflow_oracle.py').TensorFlowOracle if args.reference_frames_per_snr else None
    report = {'device': args.device, 'seed': 73921, 'frames_per_snr': args.frames_per_snr,
              'reference_frames_per_snr': args.reference_frames_per_snr,
              'ber_definition': 'mean(abs(hard_decision - actual_codeword)); no signed cancellation',
              'models': {}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        args.report.write_text(json.dumps(report, indent=2))
    for name, iterations in [('base', 20), ('post', 30)]:
        path = ROOT / f'runs/{name}/best_decoder_end{iterations}.pt'
        model, metadata = load_checkpoint(path, args.device)
        model.eval()
        parity = parity_check_matrix(model.graph)
        generator = generator_matrix(parity)
        length = parity.shape[1]
        assert generator.shape == (432, 576)
        model_report = {'checkpoint_metadata': metadata, 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'length': length, 'dimension': len(generator), 'parity_rank': length - len(generator),
                        'generator_syndrome_nonzeros': int(np.count_nonzero(generator.astype(np.int64) @ parity.T % 2)),
                        'snr_results': []}
        report['models'][name] = model_report
        settings = ExperimentConfig(seed=73921, device=args.device)
        sampler = AWGNSampler(model.graph, model.config, settings)
        # Original encoder/channel implementation must generate the exact same samples.
        original_words = np.random.RandomState(2042 + settings.seed)
        original_noise = np.random.RandomState(1074 + settings.seed)
        sigmas = np.sqrt(1 / (2 * 10 ** (np.asarray(settings.signal_to_noise_ratios_db) / 10) * sampler.rate))
        received, labels = sampler.sample(batch_size, generator_matrix=generator)
        expected, expected_labels = original_data.create_mix_epoch(sigmas, original_words, original_noise, batch_size,
            model.graph.variable_nodes, model.graph.variable_nodes - model.graph.check_nodes,
            model.graph.lifting_factor, generator, False, 2, 0, 0, 0, 0, model.config.quantization_bits, model.config.llr_clip)
        np.testing.assert_array_equal(received.numpy(), expected.astype(np.float32))
        np.testing.assert_array_equal(labels.numpy(), expected_labels)
        model_report['original_nonzero_encoder_and_awgn_exact_match'] = True
        with torch.no_grad():
            noiseless = ((2 * labels - 1) * 7.5).reshape(batch_size, model.graph.variable_nodes, model.graph.lifting_factor).to(args.device)
            decoded = model(noiseless).target_llrs[-1].cpu().numpy()
        model_report['noiseless'] = counts(decoded, labels.numpy())
        assert model_report['noiseless']['bit_errors'] == 0
        reference = oracle_class(model, batch_size, loss_function='binary_cross_entropy') if oracle_class else None
        for snr in settings.signal_to_noise_ratios_db:
            # A fresh, deterministic stream makes base/post use exactly the same paired frames.
            sampler = AWGNSampler(model.graph, model.config, replace(settings, seed=settings.seed + int(snr * 100)))
            result = {'snr_db': snr, 'frames': args.frames_per_snr, 'bits': args.frames_per_snr * length,
                      'label_ones': 0, 'label_zero_frames': 0, 'invalid_codewords': 0,
                      'zero_word_frame_errors': 0, 'channel_bit_errors': 0,
                      'posterior_symmetry_max_abs_difference': 0., 'posterior_symmetry_mismatched_values': 0,
                      'hard_symmetry_mismatches_excluding_ties': 0,
                      'mixed_only_frame_failures': 0, 'zero_only_frame_failures': 0,
                      'reference_max_posterior_difference': 0., 'reference_decision_mismatches': 0,
                      'reference_bce_max_difference': 0.,
                      **{key: 0 for key in counts(np.zeros((1, 1)), np.zeros((1, 1)))}}
            for offset in range(0, args.frames_per_snr, batch_size):
                channel, labels = sampler.sample(batch_size, (snr,), generator_matrix=generator)
                label_array = labels.numpy()
                result['invalid_codewords'] += int(np.any(label_array @ parity.T % 2, axis=1).sum())
                result['label_ones'] += int(label_array.sum())
                result['label_zero_frames'] += int((label_array.sum(1) == 0).sum())
                signs = (1 - 2 * labels).reshape_as(channel)
                zero_channel = channel * signs
                with torch.no_grad():
                    mixed_output = model(channel.to(args.device)).target_llrs.cpu()
                    zero_output = model(zero_channel.to(args.device)).target_llrs.cpu()
                actual = mixed_output.numpy()
                zero_actual = zero_output.numpy()
                flipped = zero_actual * signs.flatten(1).numpy()[None]
                difference = np.abs(actual - flipped)
                result['posterior_symmetry_max_abs_difference'] = max(result['posterior_symmetry_max_abs_difference'], float(difference.max()))
                result['posterior_symmetry_mismatched_values'] += int(np.count_nonzero(difference))
                result['hard_symmetry_mismatches_excluding_ties'] += int(np.count_nonzero(((actual >= 0) != (flipped >= 0)) & (actual != 0) & (flipped != 0)))
                zero_errors = (zero_actual[-1] >= 0).any(1)
                mixed_errors = ((actual[-1] >= 0) != label_array).any(1)
                result['zero_word_frame_errors'] += int(zero_errors.sum())
                result['mixed_only_frame_failures'] += int((mixed_errors & ~zero_errors).sum())
                result['zero_only_frame_failures'] += int((zero_errors & ~mixed_errors).sum())
                result['channel_bit_errors'] += int(((channel.flatten(1).numpy() >= 0) != label_array).sum())
                for key, value in counts(actual[-1], label_array).items():
                    result[key] += value
                if reference and offset < args.reference_frames_per_snr:
                    for inputs, outputs, targets in [(channel, actual, labels), (zero_channel, zero_actual, torch.zeros_like(labels))]:
                        expected, expected_loss = reference.evaluate(inputs.numpy(), labels=targets.numpy())
                        expected = expected.reshape(iterations, batch_size, length)
                        result['reference_max_posterior_difference'] = max(result['reference_max_posterior_difference'], float(np.max(np.abs(expected - outputs))))
                        result['reference_decision_mismatches'] += int(np.count_nonzero((expected >= 0) != (outputs >= 0)))
                        actual_loss = decoding_loss(torch.from_numpy(outputs), targets, loss_function='binary_cross_entropy').item()
                        result['reference_bce_max_difference'] = max(result['reference_bce_max_difference'], abs(float(expected_loss) - actual_loss))
            assert result['invalid_codewords'] == 0 and result['label_zero_frames'] == 0
            result['ber'] = result['bit_errors'] / result['bits']
            result['fer'] = result['frame_errors'] / result['frames']
            result['fer_95pct_wilson_interval'] = wilson_interval(result['frame_errors'], result['frames'])
            result['zero_word_fer'] = result['zero_word_frame_errors'] / result['frames']
            result['channel_ber'] = result['channel_bit_errors'] / result['bits']
            result['always_zero_ber'] = result['label_ones'] / result['bits']
            result['always_zero_fer'] = 1.0
            result['predicted_one_fraction'] = result['predicted_ones'] / result['bits']
            result['zero_bit_error_rate'] = result['zero_bit_errors'] / (result['bits'] - result['label_ones'])
            result['one_bit_error_rate'] = result['one_bit_errors'] / result['label_ones']
            result['paired_fer_difference_standard_error'] = math.sqrt(max(0.,
                ((result['mixed_only_frame_failures'] + result['zero_only_frame_failures']) / result['frames'] -
                 (result['fer'] - result['zero_word_fer']) ** 2) / result['frames']))
            model_report['snr_results'].append(result)
            persist()
            print(name, json.dumps(result), flush=True)
        if reference:
            reference.close()
    persist()


if __name__ == '__main__':
    main()
