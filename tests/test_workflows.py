from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ldpc_torch import DecoderConfig, ExperimentConfig, NeuralLDPCDecoder, Protograph
from ldpc_torch.data import AWGNSampler, UncorrectedDataset, code_rate, write_uncorrected
from ldpc_torch.metrics import error_metrics
from ldpc_torch.training import run_experiment
from ldpc_torch.weights import load_checkpoint, load_legacy_weights, save_checkpoint, save_legacy_weights

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('legacy_print', ROOT / 'tests/reference/legacy_print.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def small_model(**kwargs):
    config = DecoderConfig(lifting_factor=2, iterations=4, **kwargs)
    return NeuralLDPCDecoder(Protograph([[0, 1, 0], [1, 0, 1]], 2), config)


@pytest.mark.parametrize('algorithm,index', [('sum_product', 0), ('min_sum', 1), ('quantized_min_sum', 2)])
@pytest.mark.parametrize('intervals', [(0, 0, 0, 0), (1, 2, 4, 4)])
def test_seeded_channel_matches_original(algorithm, index, intervals):
    graph = Protograph([[0, 1, 0], [1, 0, 1]], 2)
    config = DecoderConfig(lifting_factor=2, decoding_algorithm=algorithm)
    settings = ExperimentConfig(puncturing_start=intervals[0], puncturing_end=intervals[1],
                                shortening_start=intervals[2], shortening_end=intervals[3])
    sampler = AWGNSampler(graph, config, settings)
    reference_words = np.random.RandomState(2042 + settings.seed)
    reference_noise = np.random.RandomState(1074 + settings.seed)
    sigmas = np.sqrt(1 / (2 * 10 ** (np.asarray(settings.signal_to_noise_ratios_db) / 10) * sampler.rate))
    for _ in range(2):
        received, labels = sampler.sample(7)
        expected, bits = legacy.create_mix_epoch(sigmas, reference_words, reference_noise, 7, 3, 1, 2,
                                                  [], True, index, *intervals, config.quantization_bits, config.llr_clip)
        np.testing.assert_array_equal(received.numpy(), expected.astype(np.float32))
        np.testing.assert_array_equal(labels.numpy(), bits)


def test_metrics_preserve_original_and_offer_standard_ber():
    predictions = torch.tensor([[[-1., 1, 1], [-1, 1, -1]], [[1., -1, 1], [1, -1, 1]]])
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    actual = error_metrics(predictions, labels)
    expected = legacy.calc_ber_fer(predictions.flatten(0, 1).numpy(), 2, labels.numpy(), 2)
    np.testing.assert_allclose([actual.bit_error_rate_last, actual.frame_error_rate_last, actual.frame_error_rate], expected[:3])
    np.testing.assert_array_equal(actual.uncorrected_frames.numpy(), expected[3])
    assert error_metrics(predictions, labels, False).bit_error_rate_last == pytest.approx(0.5)


def test_code_rate_compatibility():
    assert code_rate(24, 6, 24) == 431 / 574
    assert code_rate(24, 6, 24, legacy=False) == 0.75


@pytest.mark.parametrize('sharing', [(1, 1, 2), (2, 2, 3), (3, 3, 3), (4, 4, 5), (5, 5, 5)])
def test_text_and_checkpoint_roundtrip(tmp_path, sharing):
    model = small_model(check_weight_sharing=sharing[0], unsatisfied_weight_sharing=sharing[1],
                        channel_weight_sharing=sharing[2], frozen_iterations=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(0.2, 1.8)
    text_path, checkpoint = tmp_path / 'weights.txt', tmp_path / 'decoder.pt'
    save_legacy_weights(model, text_path)
    restored = NeuralLDPCDecoder(Protograph(model.graph.shifts, 2), model.config)
    load_legacy_weights(restored, text_path)
    for source, target in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(source, target, rtol=0, atol=0)
    save_checkpoint(model, checkpoint, epoch=3)
    restored, metadata = load_checkpoint(checkpoint)
    channel = torch.randn(2, 3, 2)
    torch.testing.assert_close(restored(channel).target_llrs, model(channel).target_llrs)
    assert metadata == {'epoch': 3}


def test_new_unsatisfied_group_copies_base_weights(tmp_path):
    base, post = small_model(), small_model(unsatisfied_weight_sharing=3)
    save_legacy_weights(base, tmp_path / 'weights.txt')
    load_legacy_weights(post, tmp_path / 'weights.txt')
    for source, target in zip(base.satisfied_check_weights, post.unsatisfied_check_weights):
        torch.testing.assert_close(source, target)


def test_single_row_failure_file_roundtrip(tmp_path):
    model = small_model()
    channel = torch.tensor([[[-0.5, 1], [1.5, -2], [0, 0.5]]])
    path = tmp_path / 'uncorrected.txt'
    write_uncorrected(path, channel, torch.tensor([True]))
    dataset = UncorrectedDataset(path, model.graph, 1)
    torch.testing.assert_close(dataset.channel_llrs, channel)
    with pytest.raises(ValueError):
        UncorrectedDataset(path, model.graph, 2)


def test_frozen_prefix_and_overlap_training():
    model = small_model(frozen_iterations=1)
    parameters = model.set_training_window(2, 4)
    assert len(parameters) == 4
    assert not model.satisfied_check_weights[0].requires_grad
    assert not model.satisfied_check_weights[1].requires_grad
    assert model.satisfied_check_weights[2].requires_grad
    model.set_training_window(1, 4)
    assert model.satisfied_check_weights[1].requires_grad


def test_base_post_collection_end_to_end(tmp_path):
    data = tmp_path / 'data'
    (data / 'base_graphs').mkdir(parents=True)
    np.savetxt(data / 'base_graphs/tiny.txt', [[0, 1, 0], [1, 0, 1]], fmt='%d', delimiter='\t')
    decoder = DecoderConfig(lifting_factor=2, iterations=4)
    settings = ExperimentConfig(device='cpu', graph_name='tiny', data_directory=str(data), output_directory=str(tmp_path / 'base'),
                                signal_to_noise_ratios_db=(0.,), batch_size=2, training_samples=4,
                                validation_samples=2, epochs=1, iteration_block_size=2, retrain_previous_iterations=1)
    run_experiment(decoder, settings)
    records = [json.loads(line) for line in (tmp_path / 'base/history.jsonl').read_text().splitlines()]
    assert len(records) == 4
    checkpoint = tmp_path / 'base/best_decoder_end4.pt'
    restored, _ = load_checkpoint(checkpoint)
    assert restored.config.iterations == 4
    for suffix in ('', '_Valid', '_Test'):
        (data / 'uncorrected_samples').mkdir(exist_ok=True)
        np.savetxt(data / f'uncorrected_samples/[Uncor]_tiny{suffix}.txt', np.zeros((4, 9)), delimiter='\t')
    frozen_path = tmp_path / 'base/best_weights_end4.txt'
    post_config = replace(decoder, iterations=6, frozen_iterations=4, unsatisfied_weight_sharing=3)
    post_settings = replace(settings, sampling_mode='uncorrected', frozen_weights=str(frozen_path),
                            test_samples=2, output_directory=str(tmp_path / 'post'))
    post = run_experiment(post_config, post_settings)
    for index in range(4):
        torch.testing.assert_close(post.satisfied_check_weights[index], restored.satisfied_check_weights[index])
    collect_settings = replace(settings, sampling_mode='collect', initial_weights=str(frozen_path),
                               signal_to_noise_ratios_db=(-10.,), validation_samples=10,
                               output_directory=str(tmp_path / 'collect'))
    run_experiment(decoder, collect_settings)
    assert (tmp_path / 'collect/evaluation.json').is_file()
    assert (tmp_path / 'collect/Uncor.txt').is_file()


def test_existing_weights_and_failure_vectors():
    config = DecoderConfig(iterations=20, unsatisfied_weight_sharing=3)
    graph = Protograph.from_file(ROOT / 'base_graphs/wman_N0576_R34_z24.txt', 24)
    model = NeuralLDPCDecoder(graph, config)
    load_legacy_weights(model, ROOT / 'pretrained_weights/C0_wman_N0576_R34_z24_Opt_Weight_End20.txt')
    dataset = UncorrectedDataset(ROOT / 'uncorrected_samples/[Uncor]_wman_N0576_R34_z24.txt', graph, 2)
    output = model(dataset.channel_llrs)
    assert output.target_llrs.shape == (20, 2, 576)
    assert torch.isfinite(output.target_llrs).all()


@pytest.mark.parametrize('path', sorted((ROOT / 'reference_results').rglob('*.txt')))
def test_published_weight_files(path):
    from ldpc_torch.weights import read_legacy_weights
    sharing, groups = read_legacy_weights(path)
    assert all(len(group) == 50 for group in groups if group)
    if path.parent.name == 'wimax':
        graph_name, lifting = 'wman_N0576_R34_z24', 24
    elif path.parent.name == 'wifi':
        graph_name, lifting = '802_11n_N648_R56_z27', 27
    else:
        graph_name = path.name.removesuffix('_Weight_End50.txt')
        lifting = int(graph_name.split('_z')[1].split('_')[0])
    graph = Protograph.from_file(ROOT / 'base_graphs' / f'{graph_name}.txt', lifting)
    config = DecoderConfig(lifting_factor=lifting, iterations=50, check_weight_sharing=sharing[0],
                           unsatisfied_weight_sharing=sharing[1], channel_weight_sharing=sharing[2])
    model = NeuralLDPCDecoder(graph, config)
    load_legacy_weights(model, path)
    with torch.no_grad():
        output = model(torch.full((1, graph.variable_nodes, lifting), -2.0), iterations=2)
    assert torch.isfinite(output.target_llrs).all()
