import random, os, itertools, sys, gc, warnings, logging, json


sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, Input, Model
from tensorflow.keras.layers import Layer, Dense, SimpleRNNCell, LSTMCell, GRUCell, Dropout, GaussianNoise, Concatenate, Reshape, Add
from tensorflow.keras.losses import MSE, MAE, logcosh 

from utils.evaluator import compute_anomaly_scores
from utils.experiment import core_seed, running_seeds, save_model_configs, save_results
from utils.experiment import data_loaders, seq_lengths, strides, n_epochs, batch_size, clear_memory

from tqdm import tqdm


# THESE LINES ARE FOR REPRODUCIBILITY
random.seed(core_seed)
np.random.seed(core_seed)
tf.random.set_seed(core_seed)


# Compute Cell Operations at Time t
def _compute_current_timestep(t, xt, t_states, t_outputs, cell, arch_connections, btw_connection, layer_configs, n_features, en_states = None, is_decoder = False):
    n_layers = len(layer_configs)
    n_timesteps = arch_connections.shape[0]
    cell_name = cell.__name__
    l_outputs, l_states = [], [] # shape: [n_layers, n_units]
        
    for l, configs in enumerate(layer_configs):
        n_units, activation, dropout, connection = configs["n_units"], configs["activation"], configs["dropout"], configs["connection"]
        
        # create initial hidden states for current layer
        if cell_name == 'LSTMCell':
            # hidden states and carry states
            init_states = [tf.zeros([n_timesteps, n_units])[None, t], tf.zeros([n_timesteps, n_units])[None, t]]
        else:
            init_states = tf.zeros([n_timesteps, n_units])[None, t]
        
        new_state = None
        if t > 0:
            # between-layer connectivity patterns (in addition to the "inputs")
            l_state = init_states # default with no additional states (inputs)
            if btw_connection == "feedback":
                if l < n_layers - 1: # not last layer
                    l_state = t_states[t - 1][l + arch_connections[t][l][0]]

            elif btw_connection == "skip":
                if l > 0: # not first layer
                    l_state = t_states[t - 1][l + arch_connections[t][l][0]]

            elif btw_connection == "full":
                if n_layers == 1:
                    l_state = init_states
                elif l == 0: # first layer
                    l_state = t_states[t - 1][l + arch_connections[t][l][0]]
                elif l == n_layers - 1: # last layer
                    l_state = t_states[t - 1][l + arch_connections[t][l][0]]
                else:
                    if cell_name == "LSTMCell":
                        l_state = [
                            [t_states[t - 1][l + arch_connections[t][l][0]][0], t_states[t - 1][l + arch_connections[t][l][1]][0]],
                            [t_states[t - 1][l + arch_connections[t][l][0]][1], t_states[t - 1][l + arch_connections[t][l][1]][1]]
                        ]
                        l_state = [Concatenate()(l_state[0]), Concatenate()(l_state[1])]
                    else:
                        l_state = [t_states[t - 1][l + arch_connections[t][l][0]], t_states[t - 1][l + arch_connections[t][l][1]]]
                        l_state = Concatenate()(l_state)

            # within-layer temporal connectivity (main previous states)
            if connection == "uniform_skip":
                skip_length = 2 ** l
                s_idx = t - skip_length if t - skip_length >= 0 else 0
                # layer-wise + timestep-wise states
                if cell_name == "LSTMCell":
                    new_state = [Dense(n_units, trainable = False)(l_state[0]) + t_states[s_idx][l][0], Dense(n_units, trainable = False)(l_state[1]) + t_states[s_idx][l][1]]
                else:
                    new_state = Dense(n_units, trainable = False)(l_state) + t_states[s_idx][l]

            elif connection == "dense_random_skip":
                skip2 = arch_connections[t][l][3]
                s1_idx, s2_idx = t - 1 if t - 1 >= 0 else 0, t + skip2 if t + skip2 >= 0 else 0
                # previous state (fixed) + skip state (from random sampling)
                if cell_name == "LSTMCell":
                    new_state = [Dense(n_units, trainable = False)(l_state[0]) + (t_states[s1_idx][l][0] + t_states[s2_idx][l][0]) / 2., Dense(n_units, trainable = False)(l_state[1]) + (t_states[s1_idx][l][1] + t_states[s2_idx][l][1]) / 2.]
                else:
                    new_state = Dense(n_units, trainable = False)(l_state) + (t_states[s1_idx][l] + t_states[s2_idx][l]) / 2.

            elif connection == "sparse_random_skip":
                skip1, skip2 = arch_connections[t][l][2], arch_connections[t][l][3]
                w1, w2 = 1. if arch_connections[t][l][2] != 0 else 0., 1. if arch_connections[t][l][3] != 0 else 0.  # randomed sparseness weights
                s1_idx, s2_idx = t + skip1 if t + skip1 >= 0 else 0, t + skip2 if t + skip2 >= 0 else 0
                s1_idx, s2_idx = 0 if w1 == 0. else s1_idx, 0 if w2 == 0. else s2_idx # avoid index out of range error for weight == 0 case
                # previous state * w1 + skip state * w2
                if cell_name == "LSTMCell":
                    new_state = [Dense(n_units, trainable = False)(l_state[0]) + (w1 * t_states[s1_idx][l][0] + w2 * t_states[s2_idx][l][0]) / (w1 + w2), Dense(n_units, trainable = False)(l_state[1]) + (w1 * t_states[s1_idx][l][1] + w2 * t_states[s2_idx][l][1]) / (w1 + w2)] 
                else:
                    new_state = Dense(n_units, trainable = False)(l_state) + (w1 * t_states[s1_idx][l] + w2 * t_states[s2_idx][l]) / (w1 + w2)

            else:
                # default case: (always t - 1) get previous state or initial state
                s_idx = t - 1
                if cell_name == "LSTMCell":
                    new_state = [Dense(n_units, trainable = False)(l_state[0]) + t_states[s_idx][l][0], Dense(n_units, trainable = False)(l_state[1]) + t_states[s_idx][l][1]]
                else:
                    new_state = Dense(n_units, trainable = False)(l_state) + t_states[s_idx][l]
        
        else:
            if is_decoder:
                # get the "last" timestep's hiddien states from the encoder of the same level
                if cell_name == "LSTMCell":
                    new_state = [Dense(n_units, trainable = False)(en_states[-1][l][0]), Dense(n_units, trainable = False)(en_states[-1][l][1])]
                else:
                    new_state = Dense(n_units, trainable = False)(en_states[-1][l]) 
        
        # set inputs
        if l > 0: # not the first layer
            xt = l_outputs[l - 1] # from the pervious layer at the same time step
        elif t > 0 and l == 0 and is_decoder: # each timestep of the decoder's first layer
            xt = t_outputs[t - 1][-1] # output from the last layer of the previous time step
        # else: original input xt provided above

        new_state = init_states if new_state == None else new_state
        # compute outputs and new states (share learnable weights across all time steps)
        output, state = cell(units=n_units, activation=activation, dropout=dropout, trainable = t == 0)(xt, new_state)
        if is_decoder and l == n_layers - 1: # the last layer of decoder
            h_state = state[0] if cell_name == "LSTMCell" else state
            output = Dense(n_features)(h_state) # predicted y at timestep t of the last layer based on computed hidden states
        
        l_outputs.append(output)
        l_states.append(state)
                
    return l_outputs, l_states


def build_graph(input_shape, graph_configs, arch_connections):
    tf.keras.backend.clear_session()
    
    main_configs, layer_configs = graph_configs
    
    cells = {
        'RNN': SimpleRNNCell,
        'LSTM': LSTMCell,
        'GRU': GRUCell
    }
    z_dim = main_configs["z_dim"]
    add_noise = main_configs["noise"]
    cell = cells[main_configs["cell"]]
    n_ae = main_configs["n_ae"]

    n_timesteps, n_features = input_shape[1], input_shape[2]
    
    print(f'\nBuilding: n_ae: {n_ae}, cell: {cell.__name__}, z_dim:{z_dim}, scoring:{main_configs["scoring"]}, n_layers: {len(layer_configs["encoder"])}, \n layer configs: {layer_configs}')    
    
    # BUILD ENCODER NETWORK
    encoder_states = [] # collect the (computed) hidden states of each layer
    encoder_outputs = [] # shared hidden states
    encoder_btw = layer_configs["encoder_btw"]
    encoder_in = Input(shape=[n_timesteps, n_features])
    X = encoder_in
    if add_noise:
        X = GaussianNoise(0.2)(X)

    for n in range(n_ae):
        # shape: [n_timesteps, n_layers]
        t_outputs, t_states = [], []
        for t in range(n_timesteps):
            # generate graph for model training
            xt = X[:, t, :]
            outputs, states = _compute_current_timestep(t, xt, t_states, t_outputs, cell, arch_connections[0][n], encoder_btw, layer_configs["encoder"], n_features)

            t_outputs.append(outputs)
            t_states.append(states)

        encoder_states.append(t_states)
        encoder_outputs.append(t_outputs[-1][-1]) # last timestep of the last layer --> latent representation of each encoder

    # latent hidden states
    Z = Concatenate()(encoder_outputs)
    Z = Dense(z_dim, activation="relu")(Z)

    # BUILD DECODER NETWORK
    decoder_btw = layer_configs["decoder_btw"]
    decoder_states = [] # layer-wise "additional" hidden states
    decoder_outputs = [] # predicted output Y at each timestep for the last layer
    for n in range(n_ae):
        # shape: [n_timesteps, n_layers]
        t_outputs, t_states = [], []
        for t in range(n_timesteps):
            if t == 0:
                xt = Z
            outputs, states = _compute_current_timestep(t, xt, t_states, t_outputs, cell, arch_connections[1][n], decoder_btw, layer_configs["decoder"], n_features, en_states = encoder_states[n], is_decoder = True)

            t_outputs.append(outputs)
            t_states.append(states)

        Y = np.array(t_outputs)[:, -1].tolist() # get output of all timesteps for the last layer
        Y = Concatenate()(Y)
        Y = Reshape([n_timesteps, n_features])(Y)

        decoder_states.append(t_states)
        decoder_outputs.append(Y)

    model = Model(inputs = encoder_in, outputs = decoder_outputs)
    return model