"""Run the unmodified TensorFlow source as a numerical oracle (optional extra)."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from ldpc_torch import DecoderConfig, NeuralLDPCDecoder, Protograph, decoding_loss
from ldpc_torch.optimizer import LegacyAdam
from ldpc_torch.quantization import quantize_llr

# TensorFlow is never imported by the runtime package.
tf = pytest.importorskip('tensorflow.compat.v1')
tf.disable_v2_behavior()
spec = importlib.util.spec_from_file_location('legacy_main', Path(__file__).parent / 'reference/legacy_main.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def compare_reference(shifts, config, channel, loss_kind='binary_cross_entropy', decay=0.7, tolerance=3e-5):
    model = NeuralLDPCDecoder(Protograph(shifts, config.lifting_factor), config)
    with torch.no_grad():
        for group in model.weight_groups:
            for iteration, parameter in enumerate(group):
                parameter.copy_(torch.linspace(0.67 + 0.03 * iteration, 1.23 + 0.03 * iteration, len(parameter)))
    tensor = torch.tensor(channel, requires_grad=True)
    labels = torch.zeros(channel.shape[0], shifts.shape[1] * config.lifting_factor)
    output = model(tensor)
    loss = decoding_loss(output.target_llrs, labels, config.frozen_iterations, loss_kind, decay)
    loss.backward()
    tf.reset_default_graph()
    checks, variables, base, check_degrees, variable_degrees, edges, _, _ = legacy.init_parameter(
        shifts, np.array([2.0]), config.lifting_factor, 0, 0, 0, 0)
    matrices = legacy.init_connecting_matrix(shifts, base, variables, checks, edges, config.lifting_factor,
                                             variable_degrees, check_degrees, 0, 0)
    net = {'xa': tf.placeholder(tf.float32, channel.shape), 'ya': tf.constant(labels.numpy()),
           'etha': tf.constant(decay), 'learn_rate': tf.constant(0.001),
           'LLRa0': tf.zeros((channel.shape[0], config.lifting_factor, edges))}
    reference_variables = []
    actual_parameters = []
    for group_index, group in enumerate(model.weight_groups):
        for iteration, parameter in enumerate(group):
            variable = tf.Variable(parameter.detach().numpy(), name=f'var_{group_index}_{iteration}')
            net[f'var_{group_index}_{iteration}'] = variable
            if parameter.requires_grad:
                reference_variables.append(variable)
                actual_parameters.append(parameter)
    arguments = dict(sharing=config.sharing,
                     decoding_type=['sum_product', 'min_sum', 'quantized_min_sum', 'legacy_min_sum'].index(config.decoding_algorithm),
                     sampling_type=0, loss_type=['binary_cross_entropy', 'soft_bit_error', 'frame_error'].index(loss_kind),
                     target_node=config.target_variable_nodes, iters_max=config.iterations,
                     fixed_iter=config.frozen_iterations, fixed_init=0,
                     training_iter_start=config.frozen_iterations, training_iter_end=config.iterations,
                     N_proto=variables, M_proto=checks, Num_edge_proto=edges, z_value=config.lifting_factor,
                     batch_size=channel.shape[0], q_bit=config.quantization_bits, clip_LLR=config.llr_clip)
    arguments.update(dict(zip(('Lift_Matrix1', 'Lift_Matrix2', 'W_odd2even', 'W_skipconn2even',
                               'W_even2odd', 'W_output', 'W_skipconn2odd', 'W_even2odd_with_self'), matrices)))
    for iteration in range(config.iterations):
        legacy.build_neural_network(net, curr_iter=iteration, **arguments)
    reference_gradients = tf.gradients(net['lossa'], [net['xa']] + reference_variables)
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as session:
        session.run(tf.global_variables_initializer())
        reference_outputs, reference_loss, gradients, messages = session.run(
            [net['ya_output_all'], net['lossa'], reference_gradients, net[f'LLRa{config.iterations}']],
            feed_dict={net['xa']: channel})
    np.testing.assert_allclose(output.target_llrs.detach().flatten(0, 1).numpy(), reference_outputs, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(output.check_messages.detach().transpose(1, 2).numpy(), messages, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(loss.item(), reference_loss, rtol=tolerance, atol=tolerance)
    for actual, expected in zip([tensor.grad] + [parameter.grad for parameter in actual_parameters], gradients):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=tolerance * 5, atol=tolerance * 5)


@pytest.mark.parametrize('algorithm', ['sum_product', 'min_sum', 'quantized_min_sum', 'legacy_min_sum'])
@pytest.mark.parametrize('sharing', [(1, 0, 2), (1, 1, 3), (2, 2, 2), (3, 3, 3), (4, 0, 2), (0, 0, 3)])
def test_decoder_forward_and_gradients(algorithm, sharing):
    shifts = np.array([[0, 1, -1, 2], [2, -1, 1, 0], [-1, 2, 0, 1]])
    config = DecoderConfig(lifting_factor=3, iterations=3, decoding_algorithm=algorithm,
                           check_weight_sharing=sharing[0], unsatisfied_weight_sharing=sharing[1],
                           channel_weight_sharing=sharing[2], frozen_iterations=1 if sharing[0] == 4 else 0)
    channel = np.random.RandomState(10).normal(-0.5, 2, (2, 4, 3)).astype(np.float32)
    channel[0, 0, 0] = 0
    compare_reference(shifts, config, channel)


@pytest.mark.parametrize('loss_kind', ['binary_cross_entropy', 'soft_bit_error', 'frame_error'])
@pytest.mark.parametrize('decay', [0.0, 1.0])
def test_losses_on_quantized_ties(loss_kind, decay):
    shifts = np.array([[0, 1, 2], [1, 0, 1]])
    config = DecoderConfig(lifting_factor=2, iterations=2, unsatisfied_weight_sharing=3,
                           target_variable_nodes=2)
    channel = np.array([[[0, -0.5], [1, -2], [0.5, 0]], [[-2, -1], [0, 0], [2, 0.5]]], dtype=np.float32)
    compare_reference(shifts, config, channel, loss_kind, decay)


@pytest.mark.parametrize('bits', [6, 5, -5, 4, 3])
def test_quantizer_forward_and_clipped_gradient(bits):
    values = np.arange(-40, 40.25, 0.25, dtype=np.float32)
    actual = torch.tensor(values, requires_grad=True)
    quantized = quantize_llr(actual, bits)
    quantized.sum().backward()
    tf.reset_default_graph()
    reference_input = tf.constant(values)
    reference = legacy.Cal_MSA_Q_TF(reference_input, bits)
    with tf.Session() as session:
        expected, derivative = session.run([reference, tf.gradients(tf.reduce_sum(reference), reference_input)[0]])
    np.testing.assert_array_equal(quantized.detach().numpy(), expected)
    np.testing.assert_array_equal(actual.grad.numpy(), derivative)


def test_tensorflow_adam_update():
    tf.reset_default_graph()
    reference = tf.Variable([0.5, 0.8], dtype=tf.float32)
    loss = tf.reduce_sum(reference ** 2 * [0.001, 3.0])
    update = tf.train.AdamOptimizer(0.001).minimize(loss)
    actual = torch.nn.Parameter(torch.tensor([0.5, 0.8]))
    optimizer = LegacyAdam([actual])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for _ in range(5):
            session.run(update)
            optimizer.zero_grad()
            (actual.square() * torch.tensor([0.001, 3.0])).sum().backward()
            optimizer.step()
            np.testing.assert_allclose(actual.detach().numpy(), session.run(reference), atol=2e-7)


@pytest.mark.parametrize('filename,lifting', [('wman_N0576_R34_z24', 24), ('802_11n_N648_R56_z27', 27)])
def test_real_protographs(filename, lifting):
    shifts = np.loadtxt(Path(__file__).parents[1] / 'base_graphs' / f'{filename}.txt', dtype=int)
    config = DecoderConfig(lifting_factor=lifting, iterations=2, unsatisfied_weight_sharing=3)
    channel = np.random.RandomState(12).normal(-2, 3, (1, shifts.shape[1], lifting)).astype(np.float32)
    compare_reference(shifts, config, channel)


@pytest.mark.parametrize('algorithm', ['sum_product', 'min_sum', 'quantized_min_sum'])
def test_degree_one_and_disconnected_nodes(algorithm):
    shifts = np.array([[0, -1, -1, -1], [1, 0, -1, -1], [-1, -1, -1, -1]])
    config = DecoderConfig(lifting_factor=2, iterations=2, decoding_algorithm=algorithm,
                           unsatisfied_weight_sharing=3)
    channel = np.array([[[-1, 0], [0.5, -3], [1, 2], [-20, 20]]], dtype=np.float32)
    # tanh/atanh near saturation amplifies backend float32 transcendental error.
    compare_reference(shifts, config, channel, tolerance=1e-3 if algorithm == 'sum_product' else 3e-5)
