"""JSON configuration keeps experiment scripts free of mutable global state."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from .config import DecoderConfig, ExperimentConfig
from .training import run_experiment


def main(default_mode='base', argv=None):
    parser = argparse.ArgumentParser(description='PyTorch LDPC base/post training and failure collection')
    parser.add_argument('--mode', choices=['base', 'post', 'collect'], default=default_mode)
    parser.add_argument('--config', type=Path, help='JSON with decoder and experiment sections')
    parser.add_argument('--write-config', type=Path, help='Write resolved configuration and exit')
    parser.add_argument('--output-directory')
    parser.add_argument('--batch-size', type=int, help='Override experiment batch size (changes updates per epoch)')
    parser.add_argument('--device', help='auto (default): CUDA, then MPS, then CPU; or cpu/cuda[:index]/mps')
    parser.add_argument('--smoke', action='store_true', help='Two iterations, one training batch and one validation batch')
    arguments = parser.parse_args(argv)
    decoder = DecoderConfig()
    experiment = ExperimentConfig()
    if arguments.mode == 'post':
        decoder = replace(decoder, iterations=30, frozen_iterations=20, unsatisfied_weight_sharing=3)
        experiment = replace(experiment, sampling_mode='uncorrected', iteration_block_size=10,
                             validation_samples=5000, test_samples=5000,
                             signal_to_noise_ratios_db=(2.0, 2.1, 2.2, 2.3, 2.4, 2.5), output_directory='runs/post')
    elif arguments.mode == 'collect':
        experiment = replace(experiment, sampling_mode='collect', signal_to_noise_ratios_db=(4.0,), output_directory='runs/collect')
    if arguments.config:
        contents = json.loads(arguments.config.read_text())
        decoder = replace(decoder, **contents.get('decoder', {}))
        experiment = replace(experiment, **contents.get('experiment', {}))
    if arguments.output_directory:
        experiment = replace(experiment, output_directory=arguments.output_directory)
    if arguments.batch_size is not None:
        experiment = replace(experiment, batch_size=arguments.batch_size)
    if arguments.device:
        experiment = replace(experiment, device=arguments.device)
    if arguments.smoke:
        if decoder.frozen_iterations:
            decoder = replace(decoder, iterations=decoder.frozen_iterations + 2)
        else:
            decoder = replace(decoder, iterations=2)
        experiment = replace(experiment, batch_size=2, training_samples=2, validation_samples=2,
                             test_samples=2 if experiment.test_samples else 0, epochs=1, iteration_block_size=2)
    if arguments.write_config:
        arguments.write_config.parent.mkdir(parents=True, exist_ok=True)
        arguments.write_config.write_text(json.dumps({'decoder': asdict(decoder), 'experiment': asdict(experiment)}, indent=2))
        return
    run_experiment(decoder, experiment)
