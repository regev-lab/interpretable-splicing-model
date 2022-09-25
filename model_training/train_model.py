import argparse
import math
import os
import re
import sys
import time
import unicodedata
from functools import partial
from itertools import product

from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump, load
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    GRU,
    Activation,
    ActivityRegularization,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    Layer,
)
from tensorflow.keras.models import Model, load_model
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback

from model import *

print("TF version:", tf.__version__, file=sys.stderr)

SEED = 981
print("Using seed:", SEED, file=sys.stderr)
np.random.seed(SEED)  # for reproducibility


def dict_product(d):
    keys = list(d.keys())
    values = list(d.values())
    p = list(product(*values))
    return [{k: e for k, e in zip(keys, t)} for t in p]


def make_model_filename(d, date=time.strftime("%Y%m%d"), extension=".h5"):
    return (
        f"model_{date}________"
        + "_____".join([f"{k}___{v}" for k, v in d.items()])
        + extension
    )


def prod(l):
    out = 1
    for elem in l:
        out *= elem
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="grid search runner")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--results_folder", type=str, required=True)
    parser.add_argument("--epochs_per_batch_step", type=int, default=10)
    args = parser.parse_args()

    # create folders if they're missing
    for d in [args.model_folder, args.results_folder]:
        if not os.path.isdir(d):
            os.makedirs(d)

    grid_parameters: Dict[str, List] = {
        "energy_activation": ["softplus"],
        "activity_regularization": [0.0001],
        "position_regularization": [5e-06],
        "adjacency_regularization": [0.01],
        "adjacency_regularization_so": [0.001],
        "position_regularization_structure": [0.0],
        "adjacency_regularization_structure": [0.0],
        "adjacency_regularization_so_structure": [0.0],
        "filter_width": [6],
        "num_filters": [20],
        "structure_filter_width": [30],
        "num_structure_filters": [8],
        "dropout_rate": [0.01],
        "model_type": ["custom_adjacency_regularizer"],
    }

    xTr = load(os.path.join(args.data_folder, f"xTr_ES7_HeLa_ABC.pkl.gz"))
    yTr = load(os.path.join(args.data_folder, f"yTr_ES7_HeLa_ABC.pkl.gz"))
    xTe = load(os.path.join(args.data_folder, f"xTe_ES7_HeLa_ABC.pkl.gz"))
    yTe = load(os.path.join(args.data_folder, f"yTe_ES7_HeLa_ABC.pkl.gz"))

    model_hparams = dict_product(grid_parameters)[args.index]
    print(
        f"Number of total models: {prod([len(grid_parameters[k]) for k in grid_parameters])}. Running index {args.index}"
    )
    print(model_hparams)
    model_fname = (
        model_hparams["model_type"]
        + "_"
        + time.strftime("%Y%m%d")
        + "_"
        + str(args.index)
    )
    dump(model_hparams, os.path.join(args.model_folder, f"{model_fname}_lookup.pkl"))
    print(model_fname)

    model = get_model(
        input_length=90,
        randomized_region=(10, 80),
        num_filters=model_hparams["num_filters"],
        num_structure_filters=model_hparams["num_structure_filters"],
        filter_width=model_hparams["filter_width"],
        structure_filter_width=model_hparams["structure_filter_width"],
        dropout_rate=model_hparams["dropout_rate"],
        activity_regularization=model_hparams["activity_regularization"],
        tune_energy=True,
        position_regularization=model_hparams["position_regularization"],
        adjacency_regularization=model_hparams["adjacency_regularization"],
        adjacency_regularization_so=model_hparams["adjacency_regularization_so"],
        position_regularization_structure=model_hparams[
            "position_regularization_structure"
        ],
        adjacency_regularization_structure=model_hparams[
            "adjacency_regularization_structure"
        ],
        adjacency_regularization_so_structure=model_hparams[
            "adjacency_regularization_so_structure"
        ],
        energy_activation=model_hparams["energy_activation"],
    )

    print("gpus:", tf.config.list_physical_devices("GPU"), file=sys.stderr)

    batch_schedule = [16, 64, 128, 256, 512, 1024, 2048]
    epoch_schedule = [args.epochs_per_batch_step] * 7

    # train only sequence layers
    for b, e in zip(tqdm(batch_schedule), epoch_schedule):
        train_model(
            model,
            xTr,
            yTr,
            filename=os.path.join(args.model_folder, f"{model_fname}_step1.h5"),
            custom_callbacks=[
                TqdmCallback(verbose=1, tqdm_class=partial(tqdm, leave=False))
            ],
            verbose=0,
            epochs=e,
            batch_size=b,
        )
    eval_scores = model.evaluate(xTe, yTe)
    dump(eval_scores, os.path.join(args.results_folder, f"{model_fname}_step1.results"))

    # set selector for structure
    model.get_layer("output_selector").set_weights(
        [np.array([0, 1.0, 0]).astype(np.float32)]
    )
    # train structure
    for b, e in zip(tqdm(batch_schedule), epoch_schedule):
        train_model(
            model,
            xTr,
            yTr,
            filename=os.path.join(args.model_folder, f"{model_fname}_step2.h5"),
            custom_callbacks=[
                TqdmCallback(verbose=1, tqdm_class=partial(tqdm, leave=False))
            ],
            verbose=0,
            epochs=e,
            batch_size=b,
        )
    eval_scores = model.evaluate(xTe, yTe)
    dump(eval_scores, os.path.join(args.results_folder, f"{model_fname}_step2.results"))

    # set selector for tuner
    model.get_layer("output_selector").set_weights(
        [np.array([0, 0.0, 1.0]).astype(np.float32)]
    )
    for b, e in zip(tqdm(batch_schedule), epoch_schedule):
        train_model(
            model,
            xTr,
            yTr,
            filename=os.path.join(args.model_folder, f"{model_fname}_step3.h5"),
            custom_callbacks=[
                TqdmCallback(verbose=1, tqdm_class=partial(tqdm, leave=False))
            ],
            verbose=0,
            epochs=e,
            batch_size=b,
        )
    eval_scores = model.evaluate(xTe, yTe)
    dump(eval_scores, os.path.join(args.results_folder, f"{model_fname}_step3.results"))
    print(eval_scores)
