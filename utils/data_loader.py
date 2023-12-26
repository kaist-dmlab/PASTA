# Dataset Loader
import os
import numpy as np
import pandas as pd

from itertools import groupby
from operator import itemgetter

from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
for each dataset
    x_train shape: (number of samples, sequence length, number of features)
    x_test shape: (number of samples, sequence length, number of features)
    y_test shape: (number of individual samples)
"""

# Generated training sequences for use in the model.
def _create_sequences(values, seq_length, stride, historical):
    seq = []
    if historical:
        for i in range(seq_length, len(values) + 1, stride):
            seq.append(values[i-seq_length:i])
    else:
        for i in range(0, len(values) - seq_length + 1, stride):
            seq.append(values[i : i + seq_length])
   
    return np.stack(seq)

def _count_anomaly_segments(values):
    values = np.where(values == 1)[0]
    anomaly_segments = []
    
    for k, g in groupby(enumerate(values), lambda ix : ix[0] - ix[1]):
        anomaly_segments.append(list(map(itemgetter(1), g)))
    return len(anomaly_segments), anomaly_segments

# 4 Multivariate Datasets: ASD, TODS, SWaT, PSM
def load_ASD(seq_length=100, stride=1, historical=False):
    path = f'./datasets/ASD'
    f_names = sorted([f for f in os.listdir(f'{path}/train') if os.path.isfile(os.path.join(f'{path}/train', f))])

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    
    for f_name in f_names:
        train_df = pd.read_pickle(f'{path}/train/{f_name}')
        test_df = pd.read_pickle(f'{path}/test/{f_name}')
        labels = pd.read_pickle(f'{path}/test_label/{f_name}').astype(int)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)
        
        valid_idx = int(test_df.shape[0] * 0.3)
        valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]

        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        else:
            x_train.append(train_df)
            x_valid.append(valid_df)
            x_test.append(test_df)
            
        valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

        y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
        y_segment_test.append(_count_anomaly_segments(test_labels)[1])

        y_valid.append(valid_labels)
        y_test.append(test_labels) 
        
    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test, 
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_TODS(seq_length=100, stride=1, historical=False):
    path = f'./datasets/TODS'
    f_names = sorted([f for f in os.listdir(f'{path}/train') if os.path.isfile(os.path.join(f'{path}/train', f))])
    
    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    
    for f_name in f_names:
        train_df = pd.read_csv(f'{path}/train/{f_name}').values
        test_df = pd.read_csv(f'{path}/test/{f_name}').drop(columns=['label']).values
        labels = pd.read_csv(f'{path}/test/{f_name}')['label'].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)

        valid_idx = int(test_df.shape[0] * 0.3)
        valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]

        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        else:
            x_train.append(train_df)
            x_valid.append(valid_df)
            x_test.append(test_df)

        valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

        y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
        y_segment_test.append(_count_anomaly_segments(test_labels)[1])

        y_valid.append(valid_labels)
        y_test.append(test_labels) 
    
    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test, 
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_SWaT(seq_length=60, stride=1, historical=False):
    path = f'./datasets/SWaT/downsampled'
    
    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    
    train_df = np.load(f'{path}/train.npy')
    test_df = np.load(f'{path}/test.npy')
    labels = np.load(f'{path}/test_label.npy')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)
    
    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]
    
    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)
        
    valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(test_labels)[1])

    y_valid.append(valid_labels)
    y_test.append(test_labels) 

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test, 
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_PSM(seq_length=60, stride=1, historical=False):
    path = f'./datasets/PSM'

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    
    train_df = pd.read_csv(f'{path}/train.csv').iloc[:, 1:].fillna(method="ffill").values
    test_df = pd.read_csv(f'{path}/test.csv').iloc[:, 1:].fillna(method="ffill").values
    labels = pd.read_csv(f'{path}/test_label.csv')['label'].values.astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]

    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)

    valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(test_labels)[1])

    y_valid.append(valid_labels)
    y_test.append(test_labels) 
        
    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test, 
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}
