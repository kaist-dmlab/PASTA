import uuid, random, argparse, os, itertools, sys, gc, warnings, logging, json, multiprocessing, time


# Argument Parser for selecting Dataset and GPU
parser = argparse.ArgumentParser()
parser.add_argument('--budget', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--train_ratio', type=float, required=False, default=1.0)
parser.add_argument('--n_repeat', type=int, required=False, default=3)
parser.add_argument('--gpu', type=str, required=False, default="0")

args = parser.parse_args()

data_name = args.data.lower()
gpu_id = args.gpu
budget = int(args.budget)
train_ratio = float(args.train_ratio)
n_repeat = int(args.n_repeat)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id if gpu_id else "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, Input, Model
from tensorflow.keras.losses import MSE, MAE, logcosh 

from utils.evaluator import compute_metrics, compute_anomaly_scores
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

n_epochs = 5
    
def _find_best_scores(scores):
    idx = scores['f1'].index(max(scores['f1']))
    return {
        'TaP': scores['precision'][idx], 'TaR': scores['recall'][idx], 'TaF': scores['f1'][idx],
        'count': scores['count'][idx], 'ratio': scores['ratio'][idx], 'th': scores['thresholds'][idx]
    }

def run_train_model(rid:int, seed:int, bid:int, arch_matrices, graph_configs, arch_connections):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=9*1024+512)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    print(f'ArchID:{rid}-{bid}')
    main_configs, layer_configs = graph_configs
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
    
    # score for the BEST TaF ONLY! (Not for all 1K thresholds)
    scores = { 
        'datasets': [], 
        'valid': { 'TaF': [], 'TaP': [], 'TaR': [], 'TaF': [], 
                  'count': [], 'ratio': [], 
                  'threshold': [], 'anomaly_scores': [] },
        'test': { 'TaF': [], 'TaP': [], 'TaR': [], 'TaF': [],
                 'count': [], 'ratio': [], 
                 'threshold': [], 'anomaly_scores': [] },
        'labels': []
    }    

    if data_name == 'asd':
        selected_idx = [0, 2, 4, 6, 8, 10] # reducted training
    else:
        selected_idx = list(range(len(data['x_train'])))
        
    for i in tqdm(selected_idx):
        x_train, x_valid, x_test = data['x_train'][i], data['x_valid'][i], data['x_test'][i]
        y_valid, y_test = data['y_valid'][i], data['y_test'][i]
        y_segment_valid, y_segment_test = data['y_segment_valid'][i], data['y_segment_test'][i]
        
        start_time = time.time()
        model = build_graph(x_train.shape, graph_configs, arch_connections)
        build_time = time.time() - start_time
        print(f'graph built: {build_time:.2f} seconds.')

        # adjust number of samples for speed up the training process
        x_train = x_train[:int(x_train.shape[0] * train_ratio)]
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
        
        losses = {
            'mse': tf.keras.losses.MSE,
            'mae': tf.keras.losses.MAE,
            'logcosh': tf.keras.losses.logcosh
        }
        optimizer = optimizers.Adam()
        loss_fn = main_configs["loss_fn"]
        # model.compile(loss=loss_fn, optimizer=optimizer)
        
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                train_rec = model(x, training=True)
                loss_value = losses[loss_fn](y, train_rec)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss_value

        @tf.function
        def test_step(x, y):
            valid_rec = model(x, training=False)
            return losses[loss_fn](y, valid_rec)
        
        start_time = time.time()
        for epoch in range(n_epochs):
            start_epoch = time.time()
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                loss_value = train_step(x_batch, y_batch)
            print(f'time used at epoch {epoch}: {(time.time() - start_epoch):.2f} seconds')
        # model.fit(train_dataset, epochs=n_epochs, verbose=2)
        train_time = time.time() - start_time
        
        if reverse_output:
            train_pred = [np.flip(rec, axis=1) for rec in model.predict(x_train)]
            valid_pred = [np.flip(rec, axis=1) for rec in model.predict(x_valid)]
            test_pred = [np.flip(rec, axis=1) for rec in model.predict(x_test)]
            train_errors = np.mean(test_step(x_train, x_train_reverse))
            valid_errors = np.mean(test_step(x_valid, x_valid_reverse))
            test_errors = np.mean(test_step(x_test, x_test_reverse))
        else:
            train_pred = model.predict(x_train)
            valid_pred = model.predict(x_valid)
            test_pred = model.predict(x_test)
            train_errors = np.mean(test_step(x_train, x_train))
            valid_errors = np.mean(test_step(x_valid, x_valid))
            test_errors = np.mean(test_step(x_test, x_test))
        
        train_pred = np.array(train_pred)
        valid_pred = np.array(valid_pred)  
        test_pred = np.array(test_pred)
        
        start_time = time.time()
        valid_rec = compute_anomaly_scores(x_valid, valid_pred, scoring=main_configs["scoring"], x_val = x_train, rec_val = train_pred)
        valid_scores = compute_metrics(valid_rec, y_valid, y_segment_valid, stride=stride)
        best_valid_scores = _find_best_scores(valid_scores)
        valid_time = time.time() - start_time
        
        start_time = time.time()
        test_rec = compute_anomaly_scores(x_test, test_pred, scoring=main_configs["scoring"], x_val = x_valid, rec_val = valid_pred)
        test_scores = compute_metrics(test_rec, y_test, y_segment_test, stride=stride)
        best_test_scores = _find_best_scores(test_scores)
        test_time = time.time() - start_time
        
        clear_memory(model, keras.backend) # Clear GPU memory
        
        # Record Scores for current data set i
        scores['datasets'].append(f'{data_name}_{i}')
        scores['labels'].append(y_test.tolist())
        scores['valid']['anomaly_scores'].append(valid_scores['anomaly_scores'])
        scores['test']['anomaly_scores'].append(test_scores['anomaly_scores'])
        
        # Find best scores based on the best TaF
        scores['valid']['TaP'].append(best_valid_scores['TaP'])
        scores['valid']['TaR'].append(best_valid_scores['TaR'])
        scores['valid']['TaF'].append(best_valid_scores['TaF'])
        scores['valid']['count'].append(best_valid_scores['count'])
        scores['valid']['ratio'].append(best_valid_scores['ratio'])
        scores['valid']['threshold'].append(best_valid_scores['th'])
        
        scores['test']['TaP'].append(best_test_scores['TaP'])
        scores['test']['TaR'].append(best_test_scores['TaR'])
        scores['test']['TaF'].append(best_test_scores['TaF'])
        scores['test']['count'].append(best_test_scores['count'])
        scores['test']['ratio'].append(best_test_scores['ratio'])
        scores['test']['threshold'].append(best_test_scores['th'])
        
        # Save architecture's stats
        arch_stats['datasets'].append(f'{data_name}_{i}')
        arch_stats['build_time'].append(build_time)
        arch_stats['train_time'].append(train_time)
        arch_stats['valid_time'].append(valid_time)
        arch_stats['test_time'].append(test_time)
        arch_stats['n_epochs'].append(n_epochs)
        arch_stats['train_loss'].append(train_errors)
        arch_stats['valid_loss'].append(valid_errors)
        arch_stats['test_loss'].append(test_errors)
        
    with open(f'results/archs/{data_name}/{rid}/{str(uuid.uuid1(bid))}.npy', 'wb') as f:
        arch_configs = {
            'onehot': arch_matrices,
            'connection': arch_connections,
            'stats': arch_stats,
            'scores': scores,
            'seed': seed
        }
        np.save(f, arch_configs)
        print(f'saved new architecture at {rid}-{bid}')

if __name__ == '__main__':
    for rid, seed in enumerate(running_seeds(n_repeat)):
        print(f'running id {rid}/{n_repeat-1} with seed = {seed} and budget = {budget} samples.')
        # setting for each run
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        search_space = SearchSpace()
        search_space.build_search_space(budget)
        arch_matrices = search_space.get_random_architectures(budget, with_adj = True)
        graph_configs = search_space.get_architecture_configs(arch_matrices["onehot"])
        arch_connections =  []
        for config in graph_configs:
            arch_connections.append(search_space.get_architecture_connections(config, seq_length))
            
        for i in range(budget):
            p = multiprocessing.Process(target=run_train_model, args=(rid, seed, i, arch_matrices['onehot'][i], graph_configs[i], arch_connections[i],))
            p.start()
            p.join()
