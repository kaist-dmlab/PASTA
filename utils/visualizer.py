import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot  as plt

def visualize_anomaly(model_name, preds, labels, destination='screen', to_file=False):
    plt.style.use('seaborn-white')
    
    pred_idx = np.where(np.array(preds) == 1)[0]
    label_idx = np.where(np.array(labels) == 1)[0]
    
    fig, ax = plt.subplots(figsize=(25, 5))

    ax.vlines(pred_idx, ymin=5, ymax=9.5, color='b', linewidth=0.5, label='Prediction')
    ax.vlines(label_idx, ymin=0.5, ymax=4.5, color='r', linewidth=0.5, label='Ground Truth')
    ax.set(xlim=(0, len(labels)), ylim=(0, 10), yticks=[])

    if destination == 'screen':
        plt.show()
        
    if to_file:
        plt.savefig(f'{model_name}.pdf', format='pdf', dpi=300)
        

def visualize_architecture(graph_configs, arch_connections):
    # TODO: visualize a given neural architecture
    return