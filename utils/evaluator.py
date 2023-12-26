import multiprocessing

import numpy as np
import pandas as pd


from scipy.stats import norm, iqr
from scipy.spatial.distance import mahalanobis

from sklearn.metrics import auc, roc_curve
from utils.data_loader import _count_anomaly_segments

from utils.eTaPR.eTaPR_pkg import etapr


n_thresholds = 1000

def _simulate_thresholds(rec_errors, n, verbose):
    # maximum value of the anomaly score for all time steps in the test data
    thresholds, step_size = [], abs(np.max(rec_errors) - np.min(rec_errors)) / n
    th = np.min(rec_errors)
    
    if verbose:
        print(f'Threshold Range: ({np.max(rec_errors)}, {np.min(rec_errors)}) with Step Size: {step_size}')
    for i in range(n):
        thresholds.append(float(th))
        th = th + step_size

    return thresholds


def _flatten_anomaly_scores(values, stride, flatten=False):
    flat_seq = []
    if flatten:
        for i, x in enumerate(values):
            if i == len(values) - 1:
                flat_seq = flat_seq + list(np.ravel(x).astype(float))
            else:
                flat_seq = flat_seq + list(np.ravel(x[:stride]).astype(float))
    else:
        flat_seq = list(np.ravel(values).astype(float))

    return flat_seq


def compute_anomaly_scores(x, rec_x, scoring='square', x_val=None, rec_val=None, x_train=None, rec_train=None):
    """
    average anomaly scores from different sensors/channels/metrics/variables (in case of multivariate time series)
    scoring choices: ['absolute', 'square', 'gauss_dist', 'mahalanobis', 'max_norm']
    """
    scoring = scoring.lower()
    if scoring == 'absolute': # Ref. DAEMON, TadGAN
        scores = np.mean(np.mean(np.abs(x - rec_x), axis=-1), axis = 0) if len(rec_x.shape) > 3 else np.mean(np.abs(x - rec_x), axis=-1)
        return scores
    
    elif scoring == 'square': # Ref. USAD, MTAD-GAT, MSCRED, BeatGAN, OED (S-RNN)
        scores = np.mean(np.mean(np.square(x - rec_x), axis=-1), axis = 0) if len(rec_x.shape) > 3 else np.mean(np.square(x - rec_x), axis=-1)
        return scores
    
    elif scoring == 'gauss_dist': # Ref. EncDec-AD, RAE-RRN, RAMED
        errors = []
        rec_val = np.mean(rec_val, axis = 0) if len(rec_val.shape) > 3 else rec_val
        for i in range(x_val.shape[0]):
            errors.append(np.abs(x_val[i] - rec_val[i]))
        errors = np.array(errors)

        mu = np.mean(errors, axis = 0)
        cov = np.zeros([mu.shape[1], mu.shape[1]])
        for e in errors:
            sig = np.dot((e.T - mu.T), (e.T - mu.T).T)
            cov += sig
        sigma = cov / errors.shape[0]

        scores = []
        sigma_inv = np.linalg.inv(sigma)
        rec_x = np.mean(rec_x, axis = 0) if len(rec_x.shape) > 3 else rec_x
        for i in range(x.shape[0]):
            scores_t = []
            for t in range(x[i].shape[0]):
                e_t = np.abs(x[i][t] - rec_x[i][t])
                e_minus_mu = e_t - mu[t]
                score = np.dot(np.dot(e_minus_mu, sigma_inv), e_minus_mu)
                scores_t.append(score)
            scores.append(scores_t)

        return np.array(scores)
    
    elif scoring == 'mahalanobis': # Ref. NSIBF, TCN-AE
        errors = []
        rec_val = np.mean(rec_val, axis = 0) if len(rec_val.shape) > 3 else rec_val
        for i in range(x_val.shape[0]):
            errors.append(np.abs(x_val[i] - rec_val[i]))
        errors = np.array(errors)

        mu = np.mean(errors, axis = 0)
        cov = np.zeros([mu.shape[1], mu.shape[1]])
        for e in errors:
            sig = np.dot((e.T - mu.T), (e.T - mu.T).T)
            cov += sig
        sigma = cov / errors.shape[0]

        scores = []
        sigma_inv = np.linalg.inv(sigma)
        rec_x = np.mean(rec_x, axis = 0) if len(rec_x.shape) > 3 else rec_x
        for i in range(x.shape[0]):
            scores_t = []
            for t in range(x[i].shape[0]):
                e_t = np.abs(x[i][t] - rec_x[i][t])
                score = mahalanobis(e_t, mu[t], sigma_inv)
                scores_t.append(score)
            scores.append(scores_t)
            
        return np.array(scores)
    
    elif scoring == 'max_norm': # Ref. GDN
        rec_x = np.mean(rec_x, axis = 0) if len(rec_x.shape) > 3 else rec_x
        e = np.abs(x - rec_x)
        e_med = np.median(e)
        e_iqr = iqr(e)
        epsilon = 1e-7

        err_scores = (e - e_med) / (np.abs(e_iqr) + epsilon)
        err_scores = np.max(err_scores, axis = -1)

        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3

        for i in range(before_num, len(err_scores)):
            smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])
        
        return smoothed_err_scores


def compute_new_metrics(anomaly_scores, labels, n=n_thresholds, delta=0.01, theta_r=0.1, theta_p=0.5, stride=1, verbose=False):
    thresholds = _simulate_thresholds(anomaly_scores, n, verbose) 
    correct_predictions, false_alarms = [], []
    P, R, F1 = [], [], []
    eTaP, eTaR, eTaF1 = [], [], []
    paP, paR, paF1 = [], [], []
    
    flat_seq = _flatten_anomaly_scores(anomaly_scores, stride, flatten=len(anomaly_scores.shape) == 2)
    
    for th in thresholds:
        pred_anomalies = np.zeros(len(flat_seq)).astype(int) # default with no anomaly
        pred_anomalies[np.where(np.array(flat_seq) > th)[0]] = 1 # assign 1 if scores > threshold
    
        if len(labels) != len(pred_anomalies):
            print(f'evaluating with unmatch shape: Labels: {len(labels)} vs. Preds: {len(pred_anomalies)}')
            labels = labels[-len(pred_anomalies):] # ref. OmniAnomaly
         
        perf = etapr.evaluate_w_streams(np.array(labels), pred_anomalies, theta_p=theta_p, theta_r=theta_r, delta=delta)

        P.append(float(perf['precision']))
        R.append(float(perf['recall']))
        F1.append(float(2 * (perf['precision'] * perf['recall']) / (perf['precision'] + perf['recall'] + 1e-7)))
        eTaP.append(float(perf['eTaR']))
        eTaR.append(float(perf['eTaP']))
        eTaF1.append(float(perf['f1']))
        paP.append(float(perf['point_adjust_precision']))
        paR.append(float(perf['point_adjust_recall']))
        paF1.append(float(2 * (perf['point_adjust_precision'] * perf['point_adjust_recall']) / (perf['point_adjust_precision'] + perf['point_adjust_recall'] + 1e-7)))
        
        correct_predictions.append(perf['Correct_Predictions'])
        false_alarms.append(float(perf['N False Alarm']))
        
    
    return {
        'eTaP': np.nan_to_num(eTaP).tolist(),
        'eTaR': np.nan_to_num(eTaR).tolist(),
        'eTaF1': np.nan_to_num(eTaF1).tolist(),
        'paP': np.nan_to_num(paP).tolist(),
        'paR': np.nan_to_num(paR).tolist(),
        'paF1': np.nan_to_num(paF1).tolist(),
        'F1': np.nan_to_num(F1).tolist(),
        'precision': np.nan_to_num(P).tolist(),
        'recall': np.nan_to_num(R).tolist(),
        'correct_predictions': correct_predictions,
        'false_alarms': false_alarms,
        'thresholds': thresholds,
        'anomaly_scores': flat_seq
    }