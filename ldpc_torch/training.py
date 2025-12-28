"""Base training, failure sampling, and block-wise post-decoder training."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import torch
from .config import DecoderConfig, ExperimentConfig
from .data import AWGNSampler, UncorrectedDataset, write_uncorrected
from .decoder import NeuralLDPCDecoder
from .devices import accelerator_availability, resolve_device
from .graph import Protograph
from .losses import decoding_loss
from .metrics import error_metrics
from .optimizer import LegacyAdam
from .weights import load_legacy_weights, save_legacy_weights, save_checkpoint


@torch.no_grad()
def evaluate(model, sampler, settings, samples, dataset=None, iterations=None,
             loss_start=0, loss_decay=None, collection_path=None):
    if samples < settings.batch_size:
        raise ValueError('Evaluation requires at least one full batch')
    model.eval()
    device = model.graph.shifts.device
    iterations = model.config.iterations if iterations is None else iterations
    snrs = (0.0,) if dataset is not None else settings.signal_to_noise_ratios_db
    names = ('bit_error_rate_last', 'frame_error_rate_last', 'frame_error_rate', 'loss')
    results = {name: [0.0] * len(snrs) for name in names}
    batches = samples // settings.batch_size
    for batch_index in range(batches):
        for snr_index, snr in enumerate(snrs):
            channel, labels = (dataset.batch(batch_index, settings.batch_size) if dataset is not None
                               else sampler.sample(settings.batch_size, (snr,)))
            channel, labels = channel.to(device), labels.to(device)
            output = model(channel, iterations)
            metrics = error_metrics(output.target_llrs, labels)
            for name in names[:-1]:
                results[name][snr_index] += getattr(metrics, name) / batches
            if collection_path is not None:
                write_uncorrected(collection_path, channel, metrics.uncorrected_frames)
            elif loss_start < iterations:
                loss = decoding_loss(output.target_llrs, labels, loss_start, settings.loss_function,
                                     settings.loss_decay if loss_decay is None else loss_decay)
                results['loss'][snr_index] += loss.item() / batches
    return results


def run_experiment(decoder_config: DecoderConfig, settings: ExperimentConfig):
    availability = accelerator_availability()
    requested_device = settings.device
    selected_device = resolve_device(requested_device, availability)
    settings = replace(settings, device=str(selected_device))
    print(f"Device: {selected_device} (requested={requested_device}, "
          f"CUDA available={availability['cuda']}, MPS available={availability['mps']})", flush=True)
    torch.manual_seed(settings.seed)
    directory = Path(settings.data_directory)
    output_directory = Path(settings.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    graph = Protograph.from_file(directory / 'base_graphs' / f'{settings.graph_name}.txt', decoder_config.lifting_factor)
    if settings.systematic_only:
        decoder_config = replace(decoder_config, target_variable_nodes=graph.variable_nodes - graph.check_nodes)
    sampler = AWGNSampler(graph, decoder_config, settings)
    datasets = {}
    if settings.sampling_mode == 'uncorrected':
        for split, count, suffix in [('train', settings.training_samples, ''),
                                     ('validation', settings.validation_samples, '_Valid'),
                                     ('test', settings.test_samples, '_Test')]:
            if count:
                datasets[split] = UncorrectedDataset(directory / 'uncorrected_samples' / f'[Uncor]_{settings.graph_name}{suffix}.txt', graph, count)
    configuration_path = output_directory / 'configuration.json'
    configuration_path.write_text(json.dumps({'decoder': asdict(decoder_config), 'experiment': asdict(settings),
                                              'device_selection': {'requested': requested_device,
                                                                   'selected': str(selected_device),
                                                                   'availability': availability}}, indent=2))
    if settings.sampling_mode == 'collect' or decoder_config.frozen_iterations == decoder_config.iterations:
        model = NeuralLDPCDecoder(graph, decoder_config).to(settings.device)
        if settings.initial_weights or settings.frozen_weights:
            load_legacy_weights(model, settings.initial_weights or settings.frozen_weights)
        results = evaluate(model, sampler, settings, settings.validation_samples,
                           dataset=datasets.get('validation'), loss_start=decoder_config.iterations,
                           collection_path=output_directory / 'Uncor.txt' if settings.sampling_mode == 'collect' else None)
        (output_directory / 'evaluation.json').write_text(json.dumps(results, indent=2))
        save_checkpoint(model, output_directory / 'decoder.pt')
        print(json.dumps(results), flush=True)
        return model
    if settings.training_samples < settings.batch_size:
        raise ValueError('Training requires at least one full batch')
    remaining = decoder_config.iterations - decoder_config.frozen_iterations
    if remaining % settings.iteration_block_size:
        raise ValueError('The trainable iteration count must be divisible by iteration_block_size')
    previous_best = settings.frozen_weights
    if decoder_config.frozen_iterations and previous_best is None:
        candidate = directory / 'pretrained_weights' / f'C0_{settings.graph_name}_Opt_Weight_End{decoder_config.frozen_iterations}.txt'
        if not candidate.is_file():
            raise ValueError('A frozen_weights file is required for a pretrained prefix')
        previous_best = str(candidate)
    history_path = output_directory / 'history.jsonl'
    with open(history_path, 'w') as history:
        for block_start in range(decoder_config.frozen_iterations, decoder_config.iterations, settings.iteration_block_size):
            block_end = block_start + settings.iteration_block_size
            active_config = replace(decoder_config, iterations=block_end)
            model = NeuralLDPCDecoder(graph, active_config).to(settings.device)
            if settings.initial_weights:
                load_legacy_weights(model, settings.initial_weights, block_start, block_end)
            if previous_best:
                load_legacy_weights(model, previous_best, 0, block_start)
            loss_start = max(block_start - settings.retrain_previous_iterations, decoder_config.frozen_iterations)
            parameters = model.set_training_window(loss_start, block_end)
            if not parameters:
                raise ValueError('Training requires at least one trainable weight')
            optimizer = LegacyAdam(parameters, lr=settings.learning_rate)
            best_score = float('inf')
            learning_rate, decay = settings.learning_rate, settings.loss_decay
            best_path = output_directory / f'best_weights_end{block_end}.txt'
            for epoch in range(settings.epochs + 1):
                model.train()
                average_loss = 0.0
                if epoch > 0:
                    batches = settings.training_samples // settings.batch_size
                    for batch_index in range(batches):
                        channel, labels = (datasets['train'].batch(batch_index, settings.batch_size)
                                           if 'train' in datasets else sampler.sample(settings.batch_size))
                        channel, labels = channel.to(settings.device), labels.to(settings.device)
                        optimizer.zero_grad(set_to_none=True)
                        output = model(channel)
                        loss = decoding_loss(output.target_llrs, labels, loss_start, settings.loss_function, decay)
                        loss.backward()
                        optimizer.step()
                        model.constrain_weights()
                        average_loss += loss.item() / batches
                save_legacy_weights(model, output_directory / f'weights_end{block_end}.txt')
                record = dict(block_start=block_start, block_end=block_end, epoch=epoch,
                              training_loss=average_loss, learning_rate=learning_rate, loss_decay=decay)
                if settings.validation_samples:
                    results = evaluate(model, sampler, settings, settings.validation_samples,
                                       datasets.get('validation'), loss_start=loss_start, loss_decay=decay)
                    record['validation'] = results
                    score = sum(results[settings.selection_metric])
                    improved = score < best_score
                else:
                    # No validation: hand off the latest epoch instead of looking for a nonexistent file.
                    score, improved = average_loss, True
                if settings.test_samples:
                    record['test'] = evaluate(model, sampler, settings, settings.test_samples,
                                               datasets.get('test'), loss_start=loss_start, loss_decay=decay)
                if improved:
                    best_score = score
                    save_legacy_weights(model, best_path)
                    save_checkpoint(model, output_directory / f'best_decoder_end{block_end}.pt', epoch=epoch, score=score)
                record['best_score'] = best_score
                history.write(json.dumps(record) + '\n')
                history.flush()
                print(json.dumps(record), flush=True)
                if settings.loss_decay_multiplier and settings.loss_decay_interval and (epoch + 1) % settings.loss_decay_interval == 0:
                    decay *= settings.loss_decay_multiplier
                if settings.learning_rate_multiplier and settings.learning_rate_interval and (epoch + 1) % settings.learning_rate_interval == 0:
                    learning_rate *= settings.learning_rate_multiplier
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate
            previous_best = str(best_path)
    return model
