"""Session adapter around the unmodified original, used only for validation."""
import importlib.util
from pathlib import Path
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
_spec = importlib.util.spec_from_file_location('original_ldpc', Path(__file__).with_name('legacy_main.py'))
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)


class TensorFlowOracle:
    def __init__(self, model, batch_size, learning_rate=0.001, loss_function='frame_error'):
        config = model.config
        shifts = model.graph.shifts.cpu().numpy()
        self.graph = tf.Graph()
        with self.graph.as_default():
            checks, variables, base, check_degrees, variable_degrees, edges, _, _ = legacy.init_parameter(
                shifts, np.array([2.0]), config.lifting_factor, 0, 0, 0, 0)
            matrices = legacy.init_connecting_matrix(shifts, base, variables, checks, edges, config.lifting_factor,
                                                     variable_degrees, check_degrees, 0, 0)
            self.inputs = tf.placeholder(tf.float32, (batch_size, variables, config.lifting_factor))
            self.labels = tf.placeholder_with_default(tf.zeros((batch_size, variables * config.lifting_factor)),
                                                       (batch_size, variables * config.lifting_factor))
            net = {'xa': self.inputs, 'ya': self.labels,
                   'etha': tf.constant(0.0), 'learn_rate': tf.constant(learning_rate),
                   'LLRa0': tf.zeros((batch_size, config.lifting_factor, edges))}
            self.parameters = []
            self.trainable_parameters = []
            for group_index, group in enumerate(model.weight_groups):
                for iteration, parameter in enumerate(group):
                    variable = tf.get_variable(f'var_{group_index}_{iteration}', initializer=parameter.detach().cpu().numpy(),
                                               constraint=lambda values: tf.clip_by_value(values, config.minimum_weight, config.maximum_weight))
                    net[f'var_{group_index}_{iteration}'] = variable
                    self.parameters.append(variable)
                    if parameter.requires_grad:
                        self.trainable_parameters.append(variable)
            arguments = dict(sharing=config.sharing,
                             decoding_type=['sum_product', 'min_sum', 'quantized_min_sum', 'legacy_min_sum'].index(config.decoding_algorithm),
                             sampling_type=0, loss_type=['binary_cross_entropy', 'soft_bit_error', 'frame_error'].index(loss_function), target_node=config.target_variable_nodes,
                             iters_max=config.iterations, fixed_iter=config.frozen_iterations, fixed_init=0,
                             training_iter_start=config.frozen_iterations, training_iter_end=config.iterations,
                             N_proto=variables, M_proto=checks, Num_edge_proto=edges, z_value=config.lifting_factor,
                             batch_size=batch_size, q_bit=config.quantization_bits, clip_LLR=config.llr_clip)
            arguments.update(dict(zip(('Lift_Matrix1', 'Lift_Matrix2', 'W_odd2even', 'W_skipconn2even',
                                       'W_even2odd', 'W_output', 'W_skipconn2odd', 'W_even2odd_with_self'), matrices)))
            for iteration in range(config.iterations):
                legacy.build_neural_network(net, curr_iter=iteration, **arguments)
            self.outputs, self.loss, self.update = net['ya_output_all'], net['lossa'], net['train_stepa']
            self.gradients = tf.gradients(self.loss, self.trainable_parameters)
            self.assignment_inputs = [tf.placeholder(tf.float32, variable.shape) for variable in self.parameters]
            self.assignments = [tf.assign(variable, values) for variable, values in zip(self.parameters, self.assignment_inputs)]
            initialization = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        self.session.run(initialization)

    def assign(self, values):
        self.session.run(self.assignments, feed_dict=dict(zip(self.assignment_inputs, values)))

    def evaluate(self, channel, gradients=False, labels=None):
        fetches = [self.outputs, self.loss]
        if gradients:
            fetches.append(self.gradients)
        feed = {self.inputs: channel}
        if labels is not None:
            feed[self.labels] = labels
        return self.session.run(fetches, feed_dict=feed)

    def step(self, channel, labels=None):
        feed = {self.inputs: channel}
        if labels is not None:
            feed[self.labels] = labels
        self.session.run(self.update, feed_dict=feed)
        return self.session.run(self.parameters)

    def close(self):
        self.session.close()
