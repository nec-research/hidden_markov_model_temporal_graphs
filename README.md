# HMM4G: a Hidden Markov Model for Temporal Graph Representation Learning

The repository contains the scripts to reproduce the results of the paper

Errica F., Gravina A., Bacciu D., Micheli A., _Hidden Markov Models for Temporal Graph Representation Learning_, ESANN 2023.

If you found the code and paper useful, please consider citing us:
```bibtex
@inproceedings{errica_hidden_2023,
title={Hidden Markov Models for Temporal Graph Representation Learning},
author={Errica, Federico and Gravina, Alessio and Bacciu, Davide and Micheli, Alessio},
booktitle={Proceedings of the 31st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
year={2023},
}
```

## How to reproduce

The steps below are used to reproduce both model selection and risk assessment phases of HMM4G.

### Step 1) Create venv environment

    source create_environment.sh [cpu/cu112/cu113/cu116/cu117]

This will create a venv environment called `hmm4g` with cpu or cuda support depending on your choice. 

We will be using `pydgn 1.3.1` to run the experiments.

### Step 2) Create dataset

You can prepare the dataset using the following command

    pydgn-dataset --config-file DATA_CONFIGS/config_pedalme.yml

(and similarly for the other datasets using the configuration files in the `DATA_CONFIGS` folder.)

### Step 3) Launch node embedding experiment (remove debug mode to parallelize)

    pydgn-train --config-file MODEL_CONFIGS/pedalme/config_HMM4G_gaussian_embedding.yml --debug
    
This will create the HMM4G embeddings in the `HMM4G_EMBEDDINGS` folder.

(and similarly for the other datasets using the configuration files in the `MODEL_CONFIGS` folder.)

### Step 4) Launch classification task (remove debug mode to parallelize)

    pydgn-train --config-file MODEL_CONFIGS/pedalme/config_HMM4G_classifier_mlp.yml --debug

This will launch model selection and risk assessment for the readout MLP and compute the final scores.