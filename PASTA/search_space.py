import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv1D, SimpleRNNCell, LSTMCell, GRUCell, Dropout, MaxPool2D, GaussianNoise
from tensorflow.keras.losses import MSE, MAE, logcosh 

from PASTA.global_settings import get_random_settings
from PASTA.networks import get_random_networks
from PASTA.temporal_connections import get_random_temporal_connections


names = {
    'global_settings': {
        'anomaly_scoring_function': ['Absolute Errors', 'Squared Errors', 'Gaussian Distribution', 'Mahalanobis Distance', 'Max Normalized Errors'],
        # 'multivariate_learning': ['Unified Modeling', 'Separated Modeling'],
        'output_direction': ['Same as Input', 'Reverse of Input'],
        'loss_function': ['MSE', 'MAE', 'logcosh'],
#         'regularization': ['None', 'L1', 'L2', 'L1L2'],
        'noise_injection': ['None', 'Gaussian Noise'],
        'autoencoder': [n for n in range(1, 6)],
        'layer_type': ['RNN', 'LSTM', 'GRU'],
        'z_dim': [8, 16, 32, 64, 128]
    },
    'network_settings': {
        'number_of_layers': [n for n in range(1, 6)],
        'number_of_hidden_units': [16, 32, 64, 128, 256],
        'activation_function': ['tanh', 'sigmoid', 'ReLU'],
        'apply_dropout': ['No', 'Yes'],
    },
    'temporal_connection': {
        'timestep_within_layer': ['Default', 'Uniform Skipping', 'Dense Random Skipping', 'Sparse Random Skipping'],
        'timestep_between_layer': ['Default', 'Full Connection', 'Feedback Transition', 'Skip Transition']
    }
}


class SearchSpace:
    def __init__(self, is_onehot:bool = True):
        self.is_onehot = is_onehot
        self.samples = []
        self.N = 0

    def build_search_space(self, N:int = 10):
        global_setting = get_random_settings(N, self.is_onehot)
        network = get_random_networks(N, self.is_onehot)
        temporal_connection = get_random_temporal_connections(network, self.is_onehot)

        self.samples = { 
            'global_settings': global_setting, 
            'network_settings': temporal_connection 
        }
        
        self.N = N


    def get_random_architectures(self, n_arch:int = 1, with_adj:bool = False):
        selected_archs = []
        
        for idx in np.random.choice(self.N, n_arch, replace = False):
            arch = {
                'global_settings': {},
                'network_settings': self.samples['network_settings'][idx],
                'temporal_connection': self.samples['network_settings'][idx]['temporal_connection']
            }
            
            for gkey in self.samples['global_settings'].keys():
                arch['global_settings'][gkey] = self.samples['global_settings'][gkey][idx]
            
            selected_archs.append(arch)
            
        return {
            "onehot": self._get_onehot_vectors(selected_archs, with_adj), 
            "categorical": self._get_categorical_vectors(selected_archs)
        }
        

    def get_architecture_configs(self, arch_matrices):
        # Input: One-hot Architecture Matrix
        configs = []
        for arch_matrix in arch_matrices:
            if len(arch_matrix) == 4:
                global_settings, network_graph, temporal_connectivity, _ = arch_matrix
            else:
                global_settings, network_graph, temporal_connectivity = arch_matrix

            """
            Global Settings
            #0: 0 - 4: anomaly_scoring_function
            #1: 5 - 6: output_direction
            #2: 7 - 9: loss_function
            #3: 10 - 11: noise_injection
            #4: 12 - 16: autoencoder
            #5: 17 - 21: z_dim
            #6: 22 - 24: layer type 
            """
            g_indices = np.where(global_settings == 1)[0]

            scoring = ['absolute', 'square', 'gauss_dist', 'mahalanobis', 'max_norm'][g_indices[0] - 0] # ['absolute', 'square', 'gauss_dist', 'mahalanobis', 'max_norm']
            # is_separated = 1 == g_indices[1] - 5
            is_reverse = 1 == g_indices[1] - 5
            loss_fn = ['mse', 'mae', 'logcosh'][g_indices[2] - 7]
            noise = 1 == g_indices[3] - 10
            n_ae = [n for n in range(1, 6)][g_indices[4] - 12]
            z_dim = [8, 16, 32, 64, 128][g_indices[5] - 17]
            cell = ['RNN', 'LSTM', 'GRU'][g_indices[6] - 22] # ['RNN', 'LSTM', 'GRU']

            main_configs = {
                "scoring": scoring,
                "reverse_output": is_reverse,
                "loss_fn": loss_fn,
                "noise": noise,
                "n_ae": n_ae,
                "z_dim": z_dim,
                "cell": cell
            }

            """
            Neural Networks (for each layer)
            #0: 0 - 4: number_of_hidden_units
            #1: 5 - 7: activation_function
            #2: 8 - 9: apply_dropout

            Temporal Connectivity (for each layer)
            # 0 - 3: timestep_between_layer
            # 0 - 3: timestep_within_layer
            """    

            hidden_units = [16, 32, 64, 128, 256]

            activations = ['tanh', 'sigmoid', 'relu']
            dropouts = [0.0, 0.2]

            between_connections = ['default', 'full', 'feedback', 'skip']
            within_connections = ['default', 'uniform_skip', 'dense_random_skip', 'sparse_random_skip']

            encoder_net, decoder_net = network_graph
            encoder_tmp, decoder_tmp = temporal_connectivity
            encoder_btw, decoder_btw = between_connections[np.where(encoder_tmp[0] == 1)[0][0]], between_connections[np.where(decoder_tmp[0] == 1)[0][0]] # ['Default', 'Full Connection', 'Feedback Transition', 'Skip Transition']
            encoder_configs, decoder_configs = [], [] # layer-wise network + temporal connectivity configs

            for l in range(5):
                if np.sum(encoder_net[l]) > 0 and np.sum(encoder_tmp[l + 1]) > 0: 
                    en_indices = np.where(encoder_net[l] == 1)[0]
                    en_tmp_idx = np.where(encoder_tmp[l + 1] == 1)[0][0]
                    encoder_configs.append({
                        "n_units": hidden_units[en_indices[0] - 0],
                        "activation": activations[en_indices[1] - 5],
                        "dropout": dropouts[en_indices[2] - 8],
                        "connection": within_connections[en_tmp_idx]
                    })

                if np.sum(decoder_net[l]) > 0 and np.sum(decoder_tmp[l + 1]) > 0:
                    de_indices = np.where(decoder_net[l] == 1)[0]
                    de_tmp_idx = np.where(decoder_tmp[l + 1] == 1)[0][0]
                    decoder_configs.append({
                        "n_units": hidden_units[de_indices[0] - 0],
                        "activation": activations[de_indices[1] - 5],
                        "dropout": dropouts[de_indices[2] - 8],
                        "connection": within_connections[de_tmp_idx]
                    })

            layer_configs = {
                "encoder": encoder_configs,
                "encoder_btw": encoder_btw,
                "decoder": decoder_configs,
                "decoder_btw": decoder_btw
            }
            
            configs.append([main_configs, layer_configs])
            
        return configs
        
    
    def _get_categorical_vectors(self, selected_archs:list): 
        arch_matrices = []
        for arch in selected_archs:
            settings = []
            for key in arch['global_settings'].keys():
                settings += arch['global_settings'][key]
            settings = np.where(np.array(settings) == 1.)[0]

            encoder_network = np.zeros([5, 3]) - 1.
            decoder_network = np.zeros([5, 3]) - 1.
            for i, o in enumerate(arch['network_settings']['encoder']):
                net_onehot = np.array(o['unit'] + o['activation'] + o['dropout'])
                encoder_network[i] = np.where(net_onehot == 1.)[0]

            for i, o in enumerate(arch['network_settings']['decoder']):
                net_onehot = np.array(o['unit'] + o['activation'] + o['dropout'])
                decoder_network[i] = np.where(net_onehot == 1.)[0]
            network = np.stack([encoder_network, decoder_network])

            encoder_temp = np.zeros([6]) - 1.
            decoder_temp = np.zeros([6]) - 1.
            encoder_temp[0] = np.where(np.array(arch['temporal_connection']['encoder']['between']) == 1.)[0]
            for i, o in enumerate(arch['temporal_connection']['encoder']['within']):
                encoder_temp[i + 1] = np.where(np.array(o) == 1.)[0]
            decoder_temp[0] = np.where(np.array(arch['temporal_connection']['decoder']['between']) == 1.)[0]
            for i, o in enumerate(arch['temporal_connection']['decoder']['within']):
                decoder_temp[i + 1] = np.where(np.array(o) == 1.)[0]
            temporal_connection = np.stack([encoder_temp, decoder_temp])

            arch_matrices.append([settings, network, temporal_connection])

        return arch_matrices


    def _get_onehot_vectors(self, selected_archs:list, with_adj:bool = False): 
        arch_matrices = []
        for arch in selected_archs:
            settings = []
            for key in arch['global_settings'].keys():
                settings += arch['global_settings'][key]
            settings = np.array(settings)

            encoder_network = np.zeros([5, 10])
            decoder_network = np.zeros([5, 10])
            for i, o in enumerate(arch['network_settings']['encoder']):
                encoder_network[i] = o['unit'] + o['activation'] + o['dropout']

            for i, o in enumerate(arch['network_settings']['decoder']):
                decoder_network[i] = o['unit'] + o['activation'] + o['dropout']
            network = np.stack([encoder_network, decoder_network])

            encoder_temp = np.zeros([6, 4])
            decoder_temp = np.zeros([6, 4])
            encoder_temp[0] = arch['temporal_connection']['encoder']['between']
            for i, o in enumerate(arch['temporal_connection']['encoder']['within']):
                encoder_temp[i + 1] = o
            decoder_temp[0] = arch['temporal_connection']['decoder']['between']
            for i, o in enumerate(arch['temporal_connection']['decoder']['within']):
                decoder_temp[i + 1] = o
            temporal_connection = np.stack([encoder_temp, decoder_temp])

            if with_adj: # layer-level connections
                en_adjM = np.zeros([5, 5], dtype=np.int)
                de_adjM = np.zeros([5, 5], dtype=np.int)
                
                en_adjM[0, 0] = 1
                for l in range(len(arch['network_settings']['encoder'])):
                    if l - 1 >= 0:
                        en_adjM[l, l] = 1
                        en_adjM[l, l - 1] = 1

                de_adjM[0, 0] = 1
                for l in range(len(arch['network_settings']['decoder'])):
                    if l - 1 >= 0:
                        de_adjM[l, l] = 1
                        de_adjM[l, l - 1] = 1
                
                adjM = np.stack([en_adjM, de_adjM])
                arch_matrices.append([settings, network, temporal_connection, adjM])
            else:
                arch_matrices.append([settings, network, temporal_connection])

        return arch_matrices
    
    
    def _get_connection_matrix(self, n_ae_ensembles, n, t, l, btw_connection, connection, n_layers):    
        # between-layer connectivity patterns (in addition to the "inputs")
        if btw_connection == "feedback":
            if l < n_layers - 1: # not the last layer
                n_ae_ensembles[n][t][l][0] = 1 # l + 1
        elif btw_connection == "skip":
            if l > 0: # not the first layer
                n_ae_ensembles[n][t][l][0] = -1 # l - 1
        elif btw_connection == "full":
            if l == 0: # first layer
                n_ae_ensembles[n][t][l][0] = 1 # l + 1
            elif l == n_layers - 1: # last layer
                n_ae_ensembles[n][t][l][0] = -1 # l - 1
            else:
                n_ae_ensembles[n][t][l][0] = 1 # l + 1
                n_ae_ensembles[n][t][l][1] = -1 # l - 1

        # within-layer temporal connectivity
        if connection == "uniform_skip":
            n_ae_ensembles[n][t][l][2] = -(2 ** l) # s_idx = t - skip_length if t - skip_length >= 0 else 0
        elif connection == "dense_random_skip":
            skip2 = np.random.randint(low=2, high=10, size=1)[0]
            # s1_idx, s2_idx = t - 1 if t >= 0 else 0, t - skip2 if t - skip2 >= 0 else 0
            n_ae_ensembles[n][t][l][2] = -1
            n_ae_ensembles[n][t][l][3] = -skip2
        elif connection == "sparse_random_skip":
            skip1, skip2 = np.random.randint(low=1, high=10, size=2)
            w1, w2 = [(0., 1.), (1., 0.), (1., 1.)][np.random.choice(3, size=1)[0]] # sparseness weights
            # s1_idx, s2_idx = t - skip1 if t - skip1 >= 0 else 0, t - skip2 if t - skip2 >= 0 else 0
            n_ae_ensembles[n][t][l][2] = -(skip1 * w1)
            n_ae_ensembles[n][t][l][3] = -(skip2 * w2)
        else:
            # s_idx = t - 1 # get previous state or initial state (default setting)
            n_ae_ensembles[n][t][l][2] = -1

        return n_ae_ensembles
    

    def get_architecture_connections(self, graph_configs, n_timesteps):
        main_configs, layer_configs = graph_configs

        n_layers = len(layer_configs["encoder"])
        n_ae = main_configs["n_ae"]

        # enc/dec: n_ae, n_timesteps, n_layers, [0, 1] n_between_connection (get hidden state from the previous timestep), [2, 3] n_within_connection (get hidden state from the designated timestep)
        ae_ensembles = np.zeros([2, 5, n_timesteps, 5, 2 * 2], dtype=np.int) # at most two "additional" connections

        encoder_btw = layer_configs["encoder_btw"]
        decoder_btw = layer_configs["decoder_btw"]
        for n in range(n_ae):
            for t in range(1, n_timesteps):
                for l in range(n_layers):
                    en_connection, de_connection = layer_configs["encoder"][l]["connection"], layer_configs["decoder"][l]["connection"]

                    ae_ensembles[0] = self._get_connection_matrix(ae_ensembles[0], n, t, l, encoder_btw, en_connection, n_layers)
                    ae_ensembles[1] = self._get_connection_matrix(ae_ensembles[1], n, t, l, decoder_btw, de_connection, n_layers)

        return ae_ensembles
    
    
    def __str__(self):
        texts = ''
        second_mapping = {
            'unit': 'number_of_hidden_units',
            'type': 'layer_type',
            'activation': 'activation_function',
            'dropout': 'apply_dropout',
            'within': 'timestep_within_layer',
            'between': 'timestep_between_layer'
        }
        
        if self.N > 0:

            global_keys = self.samples['global_settings'].keys()

            for i in range(self.N):
                texts += f'Architecture #{i}:\n'

                texts += f'|_Global Settings:\n'
                for key in global_keys: # each global setting loops through each sample
                    key_idx = np.argmax(self.samples['global_settings'][key][i])
                    texts += f" | {key}: {names['global_settings'][key][key_idx]}\n"

                texts += f'|_Network Settings:\n'
                network = self.samples['network_settings'][i] # each sample loops through each setting

                encoders = network['encoder']
                et_idx = np.argmax(network['temporal_connection']['encoder']['between'])
                texts += f" |_Encoder: {len(encoders)} layer(s) \n" 
                texts += f" |_Between Layer Temporal Connection: {names['temporal_connection']['timestep_between_layer'][et_idx]} \n"
                for l, enc in enumerate(encoders):
                    etw_idx = np.argmax(network['temporal_connection']['encoder']['within'][l])
                    texts += f"    |_Layer {l + 1}: \n"
                    texts += f"    |_Within Layer Temporal Connection: {names['temporal_connection']['timestep_within_layer'][etw_idx]} \n"
                    for ekey in enc.keys():
                        ekey_idx = np.argmax(enc[ekey])
                        texts += f"     | {second_mapping[ekey]}: {names['network_settings'][second_mapping[ekey]][ekey_idx]} \n"


                decoders = network['decoder']
                dt_idx = np.argmax(network['temporal_connection']['decoder']['between'])
                texts += f" |_Decoder: {len(decoders)} layer(s) \n"
                texts += f" |_Temporal Connection Between Layer: {names['temporal_connection']['timestep_between_layer'][dt_idx]} \n"
                for l, dec in enumerate(decoders):
                    dtw_idx = np.argmax(network['temporal_connection']['decoder']['within'][l])
                    texts += f"    |_Layer {l + 1}: \n"
                    texts += f"    |_Within Layer Temporal Connection: {names['temporal_connection']['timestep_within_layer'][dtw_idx]} \n"
                    for dkey in dec.keys():
                        dkey_idx = np.argmax(dec[dkey])
                        texts += f"     | {second_mapping[dkey]}: {names['network_settings'][second_mapping[dkey]][dkey_idx]} \n"
        else:
            texts += 'Cannot Print Empty Search Space!'
            
        return texts