import numpy as np

from tqdm import tqdm

temporal_choices = {
    "n_within": 4,
    "n_between": 4
}

def _random_setting(n_choices:int, n_samples:int, is_onehot:bool):
    if is_onehot:
        return np.eye(n_choices)[np.random.choice(n_choices, n_samples)]
    else:
        return np.random.choice(n_choices, n_samples) + 1

    
def get_random_temporal_connections(networks:list, is_onehot:bool):    
    network_settings = []

    for network in tqdm(networks):
        setting = { 
            'encoder': { 'between': None, 'within': [] }, 
            'decoder': { 'between': None, 'within': [] } 
        }
        
        setting['encoder']['between'] = _random_setting(temporal_choices['n_between'], 1, is_onehot)[0].tolist()
        setting['decoder']['between'] = _random_setting(temporal_choices['n_between'], 1, is_onehot)[0].tolist()
        
        for _ in network['encoder']:
            setting['encoder']['within'].append(_random_setting(temporal_choices['n_within'], 1, is_onehot)[0].tolist())
            setting['decoder']['within'].append(_random_setting(temporal_choices['n_within'], 1, is_onehot)[0].tolist())

        network['temporal_connection'] = setting
        network_settings.append(network)
    
    return network_settings