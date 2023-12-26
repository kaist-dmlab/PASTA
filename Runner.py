import uuid, random, argparse, os, itertools, sys, gc, warnings, logging, json, multiprocessing, time, glob

# Argument Parser for selecting Dataset and GPU
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--z_dim', type=int, required=True)
parser.add_argument('--budget', type=int, required=True)
parser.add_argument('--gpu', type=str, required=False, default="0")

args = parser.parse_args()

data_name = args.data.lower()
gpu_id = args.gpu
budget = int(args.budget)
z_dim = int(args.z_dim)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id if gpu_id else "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

sys.path.append(os.getcwd())

logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import ngboost as ngb

from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, Input, Model
from tensorflow.keras.losses import MSE, MAE, logcosh 

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from scipy.stats import kendalltau as tau
from scipy.stats import pearsonr as corr
from scipy.stats import spearmanr

from tqdm import tqdm

from utils.evaluator import compute_metrics, compute_anomaly_scores, compute_new_metrics
from utils.experiment import core_seed, running_seeds, save_model_configs, save_results
from utils.experiment import data_loaders, seq_lengths, strides, n_epochs, batch_size, clear_memory

from PASTA.search_space import SearchSpace, names
from PASTA.graph_builder import build_graph

networks = {
    'tods': ['PASTA_CTAE', 'NGBRegressor'],
    'asd': ['PASTA_CTAE', 'NGBRegressor'],
    'swat': ['PASTA_CTAE', 'NGBRegressor'],
    'psm': ['PASTA_CTAE', 'NGBRegressor']
}

network, predictor_name = networks[data_name]

# Set Model Name
model_name = 'PASTA'

# THESE LINES ARE FOR REPRODUCIBILITY
random.seed(core_seed)
np.random.seed(core_seed)
tf.random.set_seed(core_seed)

# Data Loaders 
seq_length, stride = seq_lengths[data_name], strides[data_name]
data = data_loaders[data_name](seq_length=seq_length, stride=stride) # load dataset / 0 for non-subsequence time series

def valid_topK(idx): # arch_matrices, graph_configs, arch_connections
    
    scores = []     
    main_configs, layer_configs = graph_configs[idx]
    reverse_output = main_configs["reverse_output"]
    
    selected_idx = list(range(len(data['x_train']))) # dataset index in a benchmark
        
    for i in tqdm(selected_idx):
        x_train, x_valid, x_test = data['x_train'][i], data['x_valid'][i], data['x_test'][i]
        y_valid, y_test = data['y_valid'][i], data['y_test'][i]
        y_segment_valid, y_segment_test = data['y_segment_valid'][i], data['y_segment_test'][i]

        start_time = time.time()
        model = build_graph(x_train.shape, graph_configs[idx], arch_connections[idx])
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
        else:
            # Prepare the training dataset.
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
            train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            # Prepare the validation dataset.
            valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, x_valid))
            valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        patience = 5
        wait = 0
        best = np.inf
        is_early_stop = False
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
        else:
            train_pred = model.predict(x_train)
            valid_pred = model.predict(x_valid)

        train_pred = np.array(train_pred)
        valid_pred = np.array(valid_pred)  

        valid_rec = compute_anomaly_scores(x_valid, valid_pred, scoring=main_configs["scoring"], x_val = x_train, rec_val = train_pred)
        valid_scores = compute_new_metrics(valid_rec, y_valid, stride=stride)

        scores.append(max(valid_scores['eTaF1']))

    print(idx, np.average(scores))
    return {'perf': np.average(scores), 'idx': idx}

def run_full_train(rid:int, seed:int, arch_matrices, graph_configs, arch_connections):
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)    
    
    tf.keras.backend.clear_session()
    
    # setting for each run
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)    
    
    main_configs, layer_configs = graph_configs
    reverse_output = main_configs["reverse_output"]
    
    arch_stats = {
        'datasets': [],
        'build_time': [],
        'train_time': [],
        'test_time': [],
        'n_epochs': [],
        'train_loss': [],
        'valid_loss': [],
        'test_loss': []
    }  
    
    scores = { 
        'datasets': [], 
        'best_eTaPR': [], 'eTaP': [], 'eTaR': [], 'eTaF1': [], 'false_alarms': [],
        'best_PA': [], 'paP': [], 'paR': [], 'paF1': [],
        'best_PR': [], 'precision': [], 'recall': [], 'F1': [],
        'thresholds': [] 
    }    
    anomaly_scores = {'datasets': [], 'anomaly_scores': [], 'labels': []}
    
    selected_idx = list(range(len(data['x_train']))) # dataset index in a benchmark
        
    for i in tqdm(selected_idx):
        x_train, x_valid, x_test = data['x_train'][i], data['x_valid'][i], data['x_test'][i]
        y_valid, y_test = data['y_valid'][i], data['y_test'][i]
        y_segment_valid, y_segment_test = data['y_segment_valid'][i], data['y_segment_test'][i]
        
        start_time = time.time()
        model = build_graph(x_train.shape, graph_configs, arch_connections)
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
        else:
            # Prepare the training dataset.
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
            train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            # Prepare the validation dataset.
            valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, x_valid))
            valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        patience = 5
        wait = 0
        best = np.inf
        is_early_stop = False
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
        test_rec = compute_anomaly_scores(x_test, test_pred, scoring=main_configs["scoring"], x_val = x_valid, rec_val = valid_pred)
        test_scores = compute_new_metrics(test_rec, y_test, stride=stride)
        test_time = time.time() - start_time
        print(f'Test Time: {test_time:.2f}s, {max(test_scores["eTaF1"])}')
        
        eTaF1_i = test_scores['eTaF1'].index(max(test_scores['eTaF1']))
        paF1_i = test_scores['paF1'].index(max(test_scores['paF1']))
        F1_i = test_scores['F1'].index(max(test_scores['F1']))

        scores['best_eTaPR'].append({'eTaP': test_scores['eTaP'][eTaF1_i], 'eTaR': test_scores['eTaR'][eTaF1_i], 'eTaF1': test_scores['eTaF1'][eTaF1_i], 'idx': eTaF1_i})
        scores['best_PA'].append({'paP': test_scores['paP'][paF1_i], 'paR': test_scores['paR'][paF1_i], 'paF1': test_scores['paF1'][paF1_i], 'idx': paF1_i})
        scores['best_PR'].append({'precision': test_scores['precision'][F1_i], 'recall': test_scores['recall'][F1_i], 'F1': test_scores['F1'][F1_i], 'idx': F1_i})

        scores['datasets'].append(f'{data_name}_{i}')
        scores['precision'].append(test_scores['precision'])
        scores['recall'].append(test_scores['recall'])
        scores['F1'].append(test_scores['F1'])
        scores['paP'].append(test_scores['paP'])
        scores['paR'].append(test_scores['paR'])
        scores['paF1'].append(test_scores['paF1'])
        scores['eTaP'].append(test_scores['eTaP'])
        scores['eTaR'].append(test_scores['eTaR'])
        scores['eTaF1'].append(test_scores['eTaF1'])
        scores['false_alarms'].append(test_scores['false_alarms'])
        scores['thresholds'].append(test_scores['thresholds'])
        
        # Save architecture's stats
        arch_stats['datasets'].append(f'{data_name}_{i}')
        arch_stats['build_time'].append(build_time)
        arch_stats['train_time'].append(train_time)
        arch_stats['test_time'].append(test_time)
        arch_stats['n_epochs'].append(len(logs.epoch))
        arch_stats['train_loss'].append(train_errors)
        arch_stats['valid_loss'].append(valid_errors)
        arch_stats['test_loss'].append(test_errors)        
        
        clear_memory(model, keras.backend) # Clear GPU memory

        anomaly_scores['datasets'].append(f'{data_name}_{i}')
        anomaly_scores['anomaly_scores'].append(test_scores['anomaly_scores'])
        anomaly_scores['labels'].append(y_test.tolist())
        
    save_results(model_name, data_name, rid, scores, anomaly_scores)
    
    # save found architecture, can be disabled if not needed
    if rid == 0:
        with open(f'results/architectures/{data_name}/PASTA/{str(uuid.uuid1())}.npy', 'wb') as f:
            arch_configs = {
                'onehot': arch_matrices,
                'connection': arch_connections,
                'stats': arch_stats,
                'scores': scores,
                'seed': seed
            }
            np.save(f, arch_configs)
            print(f'saved found architecture of {data_name}!')
    

def predict_z(X, return_dict):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)    
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.load_model(f'results/pretrained_models/{network}/model_{seq_length}_{z_dim}_all/')
    encoder = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('encoder_output').output)
    
    return_dict[0] = np.array(encoder.predict(X))
    

N, Ni, TopK = int(2e4), 20, 100
if __name__ == '__main__':                          
    # initial N_low + N_full
    used_budget = 0
    start_time = time.time()
    f_names = glob.glob(f'results/architectures/{data_name}/Full/*.npy') + glob.glob(f'results/architectures/{data_name}/Reduced/*.npy')
    y_true_f1 = []
    x_setting_test = []
    x_network_test = []
    x_temp_test = []
    x_encoder_test = []
    x_decoder_test = []
    
    for fname in tqdm(f_names):
        test_arch = np.load(fname, allow_pickle=True).item()
        xs_test = test_arch['onehot']
        xc_test = test_arch['connection']

        x_setting_test.append(xs_test[0])
        x_network_test.append(xs_test[1])
        x_temp_test.append(xs_test[2])
        x_encoder_test.append(xc_test[0])
        x_decoder_test.append(xc_test[1])

        y_true_f1.append(np.average(test_arch['scores']['valid']['eTaF1']))

    x_test = [np.array(x_setting_test), np.array(x_network_test), np.array(x_temp_test), np.array(x_encoder_test), np.array(x_decoder_test)]
    y = np.array(y_true_f1)
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=predict_z, args=(x_test, return_dict,))
    p.start()
    p.join()

    Z = return_dict.values()[0]
    
    # budget
    N0 = int(budget * 0.50)
    predictors = {
        'NGBRegressor': ngb.NGBRegressor(random_state=core_seed, verbose=0),
        'LGBMRegressor': lgb.LGBMRegressor(random_state=core_seed),
        'XGBRegressor': xgb.XGBRegressor(random_state=core_seed),
        'BaggingRegressor': BaggingRegressor(base_estimator=MLPRegressor(activation='logistic', solver='lbfgs', random_state=core_seed), random_state=core_seed),
        'AdaBoostRegressor': AdaBoostRegressor(base_estimator=MLPRegressor(activation='logistic', solver='lbfgs', random_state=core_seed), random_state=core_seed)
    }
    predictor = predictors[predictor_name]
    train_idx = np.random.choice(len(Z), size=N0, replace=False)
    Z0, y0 = Z[train_idx], y[train_idx]
    predictor.fit(Z0, y0)
    used_budget += N0
    print(f'{used_budget}/{budget}')
        
    # evaluation pool N
    search_space = SearchSpace()
    search_space.build_search_space(N)
    arch_matrices = search_space.get_random_architectures(N, with_adj = True)
    graph_configs = search_space.get_architecture_configs(arch_matrices["onehot"])
    arch_connections =  []
    for config in tqdm(graph_configs):
        arch_connections.append(search_space.get_architecture_connections(config, seq_length))    
    xs_test = arch_matrices['onehot']
    xc_test = arch_connections

    x_setting_test = []
    x_network_test = []
    x_temp_test = []
    x_encoder_test = []
    x_decoder_test = []

    for i in range(N):
        x_setting_test.append(xs_test[i][0])
        x_network_test.append(xs_test[i][1])
        x_temp_test.append(xs_test[i][2])
        x_encoder_test.append(xc_test[i][0])
        x_decoder_test.append(xc_test[i][1])

    x_test = [np.array(x_setting_test), np.array(x_network_test), np.array(x_temp_test), np.array(x_encoder_test), np.array(x_decoder_test)]
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=predict_z, args=(x_test, return_dict,))
    p.start()
    p.join()
    
    Z = return_dict.values()[0]
    Z_Ni = Z0
    y_Ni = y0
    selected_idx = []
    while used_budget < budget:
        y_pred = predictor.predict(Z).ravel()
        topK_idx = np.argpartition(y_pred, -TopK)[-TopK:]
        topK_Ni_idx = np.random.choice(topK_idx, size=Ni, replace=False)
        
        Z_Ni = np.concatenate([Z[topK_Ni_idx], Z_Ni], axis=0)
        y_Ni = np.concatenate([y_pred[topK_Ni_idx], y_Ni], axis=0)
        predictor.fit(Z_Ni, y_Ni) # P_i+1
        Z = np.delete(Z, topK_Ni_idx, axis=0) # exclude those obsereved!
        
        used_budget += Ni
        print(f'{used_budget}/{budget}')
    
    K = 5
    y_pred = predictor.predict(Z).ravel()
    topK_idx = np.argpartition(y_pred, -K)[-K:] # get predicted best arch index
    print(y_pred[topK_idx])
    search_time = time.time() - start_time
    print(f'Search Time: {search_time}')
    
    with multiprocessing.Pool(1) as pool:
        results = pool.map(valid_topK, topK_idx)

    best_idx = None
    best_valid = -np.inf
    for result in results:
        if result['perf'] > best_valid:
            best_idx = result['idx']
            best_valid = result['perf']
    
    print(f'Best Valid: {best_valid}')
    
    # full train the best predicted architecture
    for rid, seed in enumerate(running_seeds(3)):
        print(f'full train running id {rid}/{3-1} with seed = {seed} and budget = {budget} samples.')

        p = multiprocessing.Process(target=run_full_train, args=(rid, seed, arch_matrices["onehot"][best_idx], graph_configs[best_idx], arch_connections[best_idx],))
        p.start()
        p.join()
