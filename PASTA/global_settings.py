import numpy as np

from tqdm import tqdm

setting_choices = {
    "n_anomaly_scoring_function": 5,
    # "n_multivariate_learning": 2,
    "n_output_direction": 2,
    "n_loss_function": 3, # ['MSE', 'MAE', 'logcosh']
#     "n_regularization": 4,
    "n_noise_injection": 2,
    "n_autoencoder": 5,
    "n_z_dim": 5,
    "n_type": 3 # ['RNN', 'LSTM', 'GRU']
}

def _random_setting(n_choices:int, n_samples:int, is_onehot:bool):
    if is_onehot:
        return np.eye(n_choices)[np.random.choice(n_choices, n_samples)]
    else:
        return np.random.choice(n_choices, n_samples) + 1

    
def get_random_settings(N:int, is_onehot:bool):
    sample_settings = dict()
    
    for key in tqdm(setting_choices.keys()):
        sample_settings[key.split('n_')[-1]] = _random_setting(setting_choices[key], N, is_onehot).tolist()
    
    return sample_settings