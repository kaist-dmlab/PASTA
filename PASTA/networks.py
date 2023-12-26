import numpy as np

from tqdm import tqdm

n_layer_choices = 5
layer_choices = {
    "n_hidden_unit": 5,
    "n_activation": 3,
    "n_dropout": 2
}

def _random_setting(n_choices:int, n_samples:int, is_onehot:bool):
    if is_onehot:
        return np.eye(n_choices)[np.random.choice(n_choices, n_samples)]
    else:
        return np.random.choice(n_choices, n_samples) + 1

    
def get_random_networks(N:int, is_onehot:bool):
    n_layers = _random_setting(n_layer_choices, N, is_onehot=False)
    
    sample_settings = []
    for n in tqdm(n_layers):
        network_settings = { 'encoder': [], 'decoder': [] }
        
        for l in range(n):
            encoder_settings = dict()
            decoder_settings = dict()
            
            for key in layer_choices.keys():
                encoder_settings[key.split('n_')[-1]] = _random_setting(layer_choices[key], 1, is_onehot)[0].tolist()
                decoder_settings[key.split('n_')[-1]] = _random_setting(layer_choices[key], 1, is_onehot)[0].tolist()
                
            network_settings['encoder'].append(encoder_settings)
            network_settings['decoder'].append(decoder_settings)
        
        sample_settings.append(network_settings)
            
    return sample_settings