import uuid, random, argparse, os, itertools, sys, gc, warnings, logging, json, multiprocessing, time, glob


# Argument Parser for selecting Dataset and GPU
parser = argparse.ArgumentParser()
# parser.add_argument('--budget', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--train_ratio', type=float, required=False, default=1.0)
parser.add_argument('--n_repeat', type=int, required=False, default=3)
parser.add_argument('--gpu', type=str, required=False, default="0")
parser.add_argument('--layer', type=str, required=True)


args = parser.parse_args()

data_name = args.data.lower()
gpu_id = args.gpu
cell_type = args.layer.upper()
train_ratio = float(args.train_ratio)
n_repeat = int(args.n_repeat)

cell_id = {
    'RNN': 22,
    'LSTM': 23,
    'GRU': 24
} # [22: 'RNN', 23:'LSTM', 24:'GRU']

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id if gpu_id else "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, Input, Model
from tensorflow.keras.losses import MSE, MAE, logcosh 

from utils.evaluator import compute_metrics, compute_anomaly_scores, compute_new_metrics
from utils.experiment import core_seed, running_seeds, save_model_configs, save_results
from utils.experiment import data_loaders, seq_lengths, strides, n_epochs, batch_size, clear_memory

from PASTA.search_space import SearchSpace
from PASTA.graph_builder import build_graph

from tqdm import tqdm


# Set Model Name
model_name = os.path.basename(__file__).split('.')[0]


# THESE LINES ARE FOR REPRODUCIBILITY
random.seed(core_seed)
np.random.seed(core_seed)
tf.random.set_seed(core_seed)


# Data Loaders 
seq_length, stride = seq_lengths[data_name], strides[data_name]
data = data_loaders[data_name](seq_length=seq_length, stride=stride) # load dataset / 0 for non-subsequence time series
    
    
def _find_best_scores(scores):
    eTaF1_i = scores['eTaF1'].index(max(scores['eTaF1']))
    paF1_i = scores['paF1'].index(max(scores['paF1']))
    F1_i = scores['F1'].index(max(scores['F1']))
    
    return {
        'eTaP': scores['eTaP'][eTaF1_i], 'eTaR': scores['eTaR'][eTaF1_i], 'eTaF1': scores['eTaF1'][eTaF1_i], 'eFA': scores['false_alarms'][eTaF1_i], 'eTaF1_th': scores['thresholds'][eTaF1_i],
        'paP': scores['paP'][paF1_i], 'paR': scores['paR'][paF1_i], 'paF1': scores['paF1'][paF1_i], 'paF1_th': scores['thresholds'][paF1_i],
        'precision': scores['precision'][F1_i], 'recall': scores['recall'][F1_i], 'F1': scores['F1'][F1_i], 'F1_th': scores['thresholds'][F1_i]
    }


def run_full_train(config): # arch_matrices, graph_configs, arch_connections):
    ebtw, ewithin, dbtw, dwithin = config
    if not f'results/archs/{data_name}/Full/{cell_type}_EB{ebtw}_EW{ewithin}_DB{dbtw}_DW{dwithin}.npy' in glob.glob(f'results/archs/{data_name}/Full/*.npy'):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=5000)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        print(f'Arch Config: {config}')

        random.seed(core_seed)
        np.random.seed(core_seed)
        tf.random.set_seed(core_seed)

        search_space = SearchSpace()
        # ebtw, ewithin, dbtw, dwithin = config

        X1 = np.zeros(25)
        X2 = np.zeros([2, 5, 10])
        X3 = np.zeros([2, 6, 4])
        X4 = np.zeros([2, 5, 5])

        X1[1] = 1.
        X1[5] = 1.
        X1[7] = 1.
        X1[10] = 1.
        X1[12] = 1.
        X1[18] = 1.
        X1[cell_id[cell_type]] = 1. # [22: 'RNN', 23:'LSTM', 24:'GRU']

        X2[0][0] = np.array([0., 0., 1., 0., 0., 1., 0., 0., 1., 0.])
        X2[0][1] = np.array([0., 0., 1., 0., 0., 1., 0., 0., 1., 0.])
        X2[1][0] = np.array([0., 0., 1., 0., 0., 1., 0., 0., 1., 0.])
        X2[1][1] = np.array([0., 0., 1., 0., 0., 1., 0., 0., 1., 0.])

        X3[0][0][ebtw] = 1. # encoder btw
        X3[1][0][dbtw] = 1. # decoder btw
        X3[0][1][ewithin] = 1. # 1st-layer encoder within
        X3[0][2][ewithin] = 1. # 2nd-layer encoder within    
        X3[1][1][dwithin] = 1. # 1st-layer decoder within
        X3[1][2][dwithin] = 1. # 2nd-layer decoder within    

        X4[0][0] = np.array([1, 0, 0, 0, 0]) # adj encoder
        X4[0][1] = np.array([1, 1, 0, 0, 0]) # adj encoder
        X4[1][0] = np.array([1, 0, 0, 0, 0]) # adj decoder
        X4[1][1] = np.array([1, 1, 0, 0, 0]) # adj decoder

        onehot = [X1, X2, X3, X4]
        graph_config = search_space.get_architecture_configs([[X1, X2, X3, X4]])[0]
        arch_connection = search_space.get_architecture_connections(graph_config, seq_length)

        main_configs, layer_configs = graph_config
        reverse_output = main_configs["reverse_output"]

        arch_stats = {
            'datasets': [],
            'build_time': [],
            'train_time': [],
            'valid_time': [],
            'test_time': [],
            'n_epochs': [],
            'train_loss': [],
            'valid_loss': [],
            'test_loss': []
        }

        scores = { 'datasets': [], 
                  'valid': {
                      'eTaP': [], 'eTaR': [], 'eTaF1': [], 'eFA': [], 'paP': [], 'paR': [], 'paF1': [], 'precision': [], 'recall': [], 'F1': [],
                      'eTaF1_th': [], 'paF1_th': [], 'F1_th': [], 'anomaly_scores': [], 'labels': []
                  },
                  'test': {
                      'eTaP': [], 'eTaR': [], 'eTaF1': [], 'eFA': [], 'paP': [], 'paR': [], 'paF1': [], 'precision': [], 'recall': [], 'F1': [],
                      'eTaF1_th': [], 'paF1_th': [], 'F1_th': [], 'anomaly_scores': [], 'labels': []
                  }
                 }

        for i in tqdm(range(len(data['x_train']))):
            x_train, x_valid, x_test = data['x_train'][i], data['x_valid'][i], data['x_test'][i]
            y_valid, y_test = data['y_valid'][i], data['y_test'][i]
            y_segment_valid, y_segment_test = data['y_segment_valid'][i], data['y_segment_test'][i]

            start_time = time.time()
            model = build_graph(x_train.shape, graph_config, arch_connection)
            build_time = time.time() - start_time
            print(f'graph built: {build_time:.2f} seconds.')

            if reverse_output:
                x_train_reverse = np.flip(x_train, axis=1)
                x_valid_reverse = np.flip(x_valid, axis=1)
                x_test_reverse = np.flip(x_test, axis=1)

                # Prepare the training dataset.
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train_reverse))
                train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                # Prepare the validation dataset.
                valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, x_valid_reverse))
                valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test_reverse))
                test_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            else:
                # Prepare the training dataset.
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
                train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                # Prepare the validation dataset.
                valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, x_valid))
                valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
                test_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            patience = 5
            losses = {
                'mse': tf.keras.losses.MSE,
                'mae': tf.keras.losses.MAE,
                'logcosh': tf.keras.losses.logcosh
            }
            optimizer = optimizers.Adam()
            loss_fn = main_configs["loss_fn"]
            es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min", restore_best_weights=True)

            start_time = time.time()
            model.compile(loss=loss_fn, optimizer=optimizer)
            logs = model.fit(train_dataset, validation_data=valid_dataset, epochs=n_epochs, callbacks=[es], verbose=2)
            train_time = time.time() - start_time
            print(f'Train Time: {train_time:.2f}s')

            if reverse_output:
                train_pred = [np.flip(rec, axis=1) for rec in model.predict(x_train)]
                valid_pred = [np.flip(rec, axis=1) for rec in model.predict(x_valid)]
                test_pred = [np.flip(rec, axis=1) for rec in model.predict(x_test)]
                train_errors = logs.history['loss'] # model.evaluate(x_train, x_train_reverse)
                valid_errors = logs.history['val_loss'] # model.evaluate(x_valid, x_valid_reverse)
                test_errors = model.evaluate(x_test, x_test_reverse, verbose=0)
            else:
                train_pred = model.predict(x_train)
                valid_pred = model.predict(x_valid)
                test_pred = model.predict(x_test)
                train_errors = logs.history['loss'] # model.evaluate(x_train, x_train)
                valid_errors = logs.history['val_loss'] # model.evaluate(x_valid, x_valid)
                test_errors = model.evaluate(x_test, x_test, verbose=0)

            train_pred = np.array(train_pred)
            valid_pred = np.array(valid_pred)  
            test_pred = np.array(test_pred)            

            start_time = time.time()
            valid_rec = compute_anomaly_scores(x_valid, valid_pred, scoring=main_configs["scoring"], x_val = x_train, rec_val = train_pred)
            valid_scores = compute_new_metrics(valid_rec, y_valid, stride=stride)
            best_valid_scores = _find_best_scores(valid_scores)
            valid_time = time.time() - start_time

            start_time = time.time()
            test_rec = compute_anomaly_scores(x_test, test_pred, scoring=main_configs["scoring"], x_val = x_valid, rec_val = valid_pred)
            test_scores = compute_new_metrics(test_rec, y_test, stride=stride)
            best_test_scores = _find_best_scores(test_scores)
            test_time = time.time() - start_time

            clear_memory(model, keras.backend) # Clear GPU memory

            # Record Scores for current data set i
            # 'eTaP': [], 'eTaR': [], 'eTaF1': [], 'eFA': [], 'paP': [], 'paR': [], 'paF1': [], 'precision': [], 'recall': [], 'F1': [],
            # 'eTaF1_th': [], 'paF1_th': [], 'F1_th': [], 'anomaly_scores': [], 'labels': []

            scores['datasets'].append(f'{data_name}_{i}')
            scores['valid']['anomaly_scores'].append(valid_scores['anomaly_scores'])
            scores['valid']['labels'].append(y_valid.tolist())        
            scores['test']['anomaly_scores'].append(test_scores['anomaly_scores'])
            scores['test']['labels'].append(y_test.tolist())

            # Find best scores based on the best eTaF
            scores['valid']['eTaP'].append(best_valid_scores['eTaP'])
            scores['valid']['eTaR'].append(best_valid_scores['eTaR'])
            scores['valid']['eTaF1'].append(best_valid_scores['eTaF1'])
            scores['valid']['eFA'].append(best_valid_scores['eFA'])        
            scores['valid']['paP'].append(best_valid_scores['paP'])
            scores['valid']['paR'].append(best_valid_scores['paR'])
            scores['valid']['paF1'].append(best_valid_scores['paF1'])        
            scores['valid']['precision'].append(best_valid_scores['precision'])
            scores['valid']['recall'].append(best_valid_scores['recall'])
            scores['valid']['F1'].append(best_valid_scores['F1'])        
            scores['valid']['eTaF1_th'].append(best_valid_scores['eTaF1_th'])
            scores['valid']['paF1_th'].append(best_valid_scores['paF1_th'])
            scores['valid']['F1_th'].append(best_valid_scores['F1_th'])
            print(f"best valid eTaF1: {best_valid_scores['eTaF1']}")

            scores['test']['eTaP'].append(best_test_scores['eTaP'])
            scores['test']['eTaR'].append(best_test_scores['eTaR'])
            scores['test']['eTaF1'].append(best_test_scores['eTaF1'])
            scores['test']['eFA'].append(best_test_scores['eFA'])        
            scores['test']['paP'].append(best_test_scores['paP'])
            scores['test']['paR'].append(best_test_scores['paR'])
            scores['test']['paF1'].append(best_test_scores['paF1'])        
            scores['test']['precision'].append(best_test_scores['precision'])
            scores['test']['recall'].append(best_test_scores['recall'])
            scores['test']['F1'].append(best_test_scores['F1'])        
            scores['test']['eTaF1_th'].append(best_test_scores['eTaF1_th'])
            scores['test']['paF1_th'].append(best_test_scores['paF1_th'])
            scores['test']['F1_th'].append(best_test_scores['F1_th'])
            print(f"best test eTaF1: {best_test_scores['eTaF1']}")

            # Save architecture's stats
            arch_stats['datasets'].append(f'{data_name}_{i}')
            arch_stats['build_time'].append(build_time)
            arch_stats['train_time'].append(train_time)
            arch_stats['valid_time'].append(valid_time)
            arch_stats['test_time'].append(test_time)
            arch_stats['n_epochs'].append(len(logs.epoch))
            arch_stats['train_loss'].append(train_errors)
            arch_stats['valid_loss'].append(valid_errors)
            arch_stats['test_loss'].append(test_errors)

        with open(f'results/archs/{data_name}/Full/{cell_type}_EB{ebtw}_EW{ewithin}_DB{dbtw}_DW{dwithin}.npy', 'wb') as f:
            arch_configs = {
                'onehot': onehot,
                'connection': arch_connection,
                'stats': arch_stats,
                'scores': scores,
                'seed': core_seed
            }
            np.save(f, arch_configs)
            print(f'saved new trained architecture with config: {config}')
            
if __name__ == '__main__':
    ebtw_connections = [0, 1, 2, 3] # ['default', 'full', 'feedback', 'skip']
    ewithin_connections = [0, 1, 2, 3] # ['default', 'uniform_skip', 'dense_random_skip', 'sparse_random_skip']
    dbtw_connections = [0, 1, 2, 3] # ['default', 'full', 'feedback', 'skip']
    dwithin_connections = [0, 1, 2, 3] # ['default', 'uniform_skip', 'dense_random_skip', 'sparse_random_skip']
    configs = itertools.product(ebtw_connections, ewithin_connections, dbtw_connections, dwithin_connections)
    # configs = list(configs)[128+64:]
    
    n_pool = {
        'tods': 8,
        'asd': 8,
        'swat': 2,
        'psm': 2
    }
    
    with multiprocessing.Pool(n_pool[data_name]) as pool:
        results = pool.map(run_full_train, configs)
    
    print(f'completed building clean architecture subset of {data_name}!')
