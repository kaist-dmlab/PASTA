import json, gc
import numpy as np

from utils.data_loader import load_ASD, load_TODS, load_SWaT, load_PSM # multivariate datasets

N = 5
core_seed = 7
batch_size = 32
n_epochs = 100
save_freq = 20
max_queries = 100

seq_lengths = {
    'asd': 100, # 100 Ref. InterFusion
    'tods': 100, # 100 Ref. TODS Benchmark (from 10 --> 100)
    'swat': 30, # 30 Ref. InterFusion
    'psm': 30 # 30 Due to large size
}

strides = {
    'asd': 1, 
    'tods': 1, 
    'swat': 1,
    'psm': 1
}

data_loaders = {
    'asd': load_ASD, 
    'tods': load_TODS, 
    'swat': load_SWaT,
    'psm': load_PSM
}

def clear_memory(model, K):
    del model
    K.clear_session()
    gc.collect()

def log_running_results(model_name, data_name, running_id, seed, time_used, metric_scores, anomaly_scores):
    print('todo')

def save_results(model_name, data_name, running_id, metric_scores, anomaly_scores):
    anomaly_score_path = f'results/anomaly_scores/{model_name}'
    evaluation_result_path = f'results/evaluation/{model_name}'
    
    with open(f'{anomaly_score_path}/{data_name}_{running_id}.json', 'w+') as outfile:
        outfile.write(json.dumps(anomaly_scores, indent = 4))    
    
    with open(f'{evaluation_result_path}/{data_name}_{running_id}.json', 'w+') as outfile:
        outfile.write(json.dumps(metric_scores, indent = 4))    
        
    print(f'Saved Results of {model_name}_{data_name}_{running_id}!')

    
def save_model_configs(model_name, data_name, params):
    model_configs_path = f'results/configs/{model_name}'
    
    with open(f'{model_configs_path}/{data_name}.json', 'w+') as outfile:
        outfile.write(json.dumps(params, indent = 4))
        
    print(f'Saved Model Configs of {model_name}_{data_name}!')


def running_seeds(n=N):
    return [2**i for i in range(n)]
