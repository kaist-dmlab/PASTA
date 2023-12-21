# PASTA
This is the official implementation of the ***PASTA: Neural Architecture Search for Anomaly Detection in Multivariate Time Series*** paper _under review_ IEEE Transactions on Emerging Topics in Computational Intelligence. <!-- [[Paper]()] -->

## Abstract
Time-series anomaly detection uncovers rare errors or intriguing events of interest that significantly deviate from normal patterns. In order to precisely detect anomalies, a detector needs to capture intricate underlying temporal dynamics of a time series, often in multiple scales. Thus, a fixed-designed neural network may not be optimal for capturing such complex dynamics as different time-series data require different learning processes to reflect their unique characteristics. This paper proposes a Prediction-based neural Architecture Search for Time series Anomaly detection framework, dubbed PASTA. Unlike previous work, besides searching for a connection between operations, we design a novel search space to search for optimal connections in the temporal dimension among recurrent cells within/between each layer, i.e., temporal connectivity, and encode them via multi-level configuration encoding networks. Experimental results from both real-world and synthetic benchmarks show that the discovered architectures by PASTA outperform the second-best state-of-the-art baseline by about 23% in F1 and 21% in VUS scores on average, confirming that the design of temporal connectivity is critical for time-series anomaly detection.


## Benchmark Datasets
| Benchmark  | Application Domain | Source | Publication | License |
| :--- | :--- | :--- | :--- | :---
| TODS  | Synthetic  | [Generator](https://github.com/datamllab/tods/tree/benchmark/benchmark/synthetic) | [NeurIPS](https://openreview.net/forum?id=r8IvOsnHchr) | `Apache 2.0`  |
| ASD  | Web Server  | [Download](https://github.com/zhhlee/InterFusion/tree/main/data)  | [KDD](https://dl.acm.org/doi/abs/10.1145/3447548.3467075)  | `MIT License`  |
| PSM  | Web Server  | [Download](https://github.com/eBay/RANSynCoders/tree/main/data)  | [KDD](https://dl.acm.org/doi/10.1145/3447548.3467174)  | `CC BY 4.0`  |
| SWaT  | Water Treatment Plant  | [Request](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)  | [CRITIS](https://link.springer.com/chapter/10.1007/978-3-319-71368-7_8) | `N/A`  |


## Pretrained (Model, Performance) Pairs for Performance Predictor
For sequence length $K = 100$, [TODS (267.6 MB)](https://www.mediafire.com/file/jpi83pgfj5evtva/TODS.zip/file), [ASD (351 MB)](https://www.mediafire.com/file/cjl2y2o05hpkyyy/ASD.zip/file).

For sequence length $K = 30$, [PSM (469.4 MB)](https://www.mediafire.com/file/e51goagbpje0ydw/PSM.zip/file), [SWaT (414 MB)](https://www.mediafire.com/file/we15fglcmu1ol69/SWaT.zip/file).

### An example of (model, performance) pairs
```python
{
  "onehot": [array([0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
        0., 1., 0., 0., 0., 1., 0., 0.]),
 array([[[0., 0., 1., 0., 0., 1., 0., 0., 1., 0.],
         [0., 0., 1., 0., 0., 1., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
 ...,
 array([[[1., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]],
 ...],  #
  "connection": [[[[[ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]

   [[ 0  0 -1  0]
    [ 0  0 -2  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]

   [[ 0  0 -1  0]
    [ 0  0 -2  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]

   ...]]],
  "stats": {
    "datasets": ['asd_0', ..., 'asd_11'],
    "build_time"; [15.236296892166138, ..., 9.724196672439575],
    "train_time": [1453.6974523067474, ..., 996.9426283836365],
    ...
  }, 
  "scores": {
    "datasets": ['asd_0', ..., 'asd_11'],
    "valid": {
      "eTaP": [0.7288074712643677, ..., 0.7185672514619883],
      "eTaR": [0.40148048357274757, ..., 0.06912423343346295],
      "eTaF1" 0.5177476672956607, ..., 0.12611640821353556],
      ...
    },
    ...
  },
  "seed": 7
}
```

## Uniformly Sampled Models for Unsupervised Pretraining of Multi-level Configuration Encoding
$K = 100$: [300K Models (763 MB)](https://www.mediafire.com/file/weyo5jz80cm0tb6/arch_100.zip/file) and $K = 30$: [600K Models (607 MB)](https://www.mediafire.com/file/nrkfpz52hk2bjq2/arch_30.zip/file)
```python
# loading saved models
arch_matrices = np.load(f'datasets/ARCH_{seq_length}/arch_matrices.npy', allow_pickle=True).item()
graph_configs = np.load(f'datasets/ARCH_{seq_length}/graph_configs.npy', allow_pickle=True)
arch_connections = np.load(f'datasets/ARCH_{seq_length}/arch_connections.npy', allow_pickle=True)
```
or directly run the following snippet (it will take about minutes to hours depending on the subspace size)
```python
# directly build search space with the given budget
subspace_size = 100
search_space = SearchSpace()
search_space.build_search_space(subspace_size)
arch_matrices = search_space.get_random_architectures(subspace_size, with_adj = True)
graph_configs = search_space.get_architecture_configs(arch_matrices["onehot"])

arch_connections =  []
  for config in tqdm(graph_configs):
    arch_connections.append(search_space.get_architecture_connections(config, seq_length))
```

### An example of untrained models (architectures)
```python
arch_matrices = {
  "onehot": [
  [array([0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 1.]),
 array([[[0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
         [0., 0., 0., 0., 1., 0., 1., 0., 0., 1.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
 
        [[0., 0., 0., 0., 1., 0., 0., 1., 0., 1.],
         [0., 0., 0., 1., 0., 0., 1., 0., 0., 1.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]),
 array([[[0., 0., 1., 0.],
         [0., 0., 1., 0.],
         [1., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],
 
        [[0., 0., 1., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]]),
 array([[[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
 
        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]])]
  , ...],  
  "categorical": [
  [array([ 1,  5,  9, 11, 13, 19, 24]),
 array([[[ 1.,  5.,  9.],
         [ 4.,  6.,  9.],
         [-1., -1., -1.],
         [-1., -1., -1.],
         [-1., -1., -1.]],
 
        [[ 4.,  7.,  9.],
         [ 3.,  6.,  9.],
         [-1., -1., -1.],
         [-1., -1., -1.],
         [-1., -1., -1.]]]),
 array([[ 2.,  2.,  0., -1., -1., -1.],
        [ 2.,  1.,  2., -1., -1., -1.]])]
  , ...],
}

graph_configs = [
  [
    array([
    {'scoring': 'square', 'reverse_output': False, 'loss_fn': 'logcosh', 'noise': True, 'n_ae': 2, 'z_dim': 32, 'cell': 'GRU'},
    {'encoder': [
        {'n_units': 32, 'activation': 'tanh', 'dropout': 0.2, 'connection': 'dense_random_skip'}, 
        {'n_units': 256, 'activation': 'sigmoid', 'dropout': 0.2, 'connection': 'default'} ], 
        'encoder_btw': 'feedback', 
     'decoder': [
        {'n_units': 256, 'activation': 'relu', 'dropout': 0.2, 'connection': 'uniform_skip'}, 
        {'n_units': 128, 'activation': 'sigmoid', 'dropout': 0.2, 'connection': 'dense_random_skip'}], 
        'decoder_btw': 'feedback'}],
     dtype=object), 
     ...
]

arch_connections = [
  [[[[[ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]

   [[ 1  0 -1 -5]
    [ 0  0 -1  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]

   [[ 1  0 -1 -6]
    [ 0  0 -1  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]

   ...,
   [[ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]

   [[ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]

   [[ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]
    [ 0  0  0  0]]]]],
    ...
]
```

## Usage

### Requirements
Python 3.9 with
```bash
pip install -r requirements.txt
```

### (Reduced-scale) Examples for Demonstration
Simple search on **PSM** benchmark: `PASTA_Example_Demo.ipynb`

### Running Script
```bash
python Runner.py --data DATA_NAME --gpu GPU_ID --budget BUDGET --z_dim Z_DIM
```
**DATA_NAME**: can be one of `["TODS", "ASD", "PSM", "SWaT"]` or your own data sets (need additional setup, please see `utils/data_loader.py` and `utils/experiment.py`)

**GPU_ID**: a specific gpu id _(default: 0)_

**BUDGET**: a total number of queries

**Z_DIM**: latent space size of the multi-level configuration encoder


## Citation
```
TBD
```
