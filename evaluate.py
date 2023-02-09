# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json

from pathlib import Path

from logging import getLogger
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np
from copy import deepcopy
from symbolicregression.utils import to_cuda
import glob
import scipy
import pickle

from parsers import get_parser
import symbolicregression
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.envs import build_env
from symbolicregression.metrics import *

from symbolicregression.trainer import Trainer
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor
from symbolicregression.model.model_wrapper import ModelWrapper
from symbolicregression.envs.new_environment import create_test_iterator
from symbolicregression_env.envs import Node, NodeParseError
from sklearn.model_selection import train_test_split
import pandas as pd

from tqdm import tqdm
import time

np.seterr(all="raise")


def read_file(filename, label="target", sep=None):

    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert X.shape[1] == feature_names.shape[0]

    return X, y, feature_names


class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def evaluate(self, iterator, params, logger=None):

        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        logger.info(f"====== STARTING EVALUATION (multi-gpu: {params.multi_gpu}) =======")
        scores = OrderedDict({"epoch": self.trainer.epoch})
        params = self.params
   
        embedder, encoder, decoder = (
                self.modules["embedder_module"],
                self.modules["encoder_module"],
                self.modules["decoder_module"],
            )
        output_tokenizer = self.modules["output_tokenizer"]
        if params.multi_gpu:
            embedder, encoder, decoder = embedder.module, encoder.module, decoder.module

        embedder.eval()
        encoder.eval()
        decoder.eval()

        env = self.env
        output_symbols = ["<EOS>", "<PAD>"]
        output_symbols += env.get_symbols() + output_tokenizer.get_symbols()
        output_id2word = {i: s for i, s in enumerate(output_symbols)}

        iterator = create_test_iterator(env, self.trainer.data_path, self.params)
        results = []

        for samples in iterator:
            datasets = [np.concatenate([yi[:, None], xi], 1) for xi, yi in zip(samples["x"], samples["y"])] 
            n_datasets = len(datasets)
            names = samples["name"] ##will need to duplicate if use more than 1 sample
            x, x_len = embedder(datasets)
            encoded = encoder("fwd", x=x, lengths=x_len, causal=False).transpose(0, 1)

            generations, _ = decoder.generate(encoded, x_len, sample_temperature=None, max_len=params.max_generated_output_len)
            generations = generations.transpose(0, 1)
            for dataset_id, name, generation in zip(np.arange(n_datasets), names, generations):
                words = [output_id2word[tok.item()] for tok in generation]
                assert words[0]=="<EOS>" and words[-1]=="<EOS>" 
                try:
                    decoded_expression: Node = output_tokenizer.decode(words[1:-1])
                    prefix = decoded_expression.prefix()
                    ytilde = decoded_expression.evaluate(samples["x"][dataset_id])
                    r2_train = stable_r2_score(samples["y"][dataset_id], ytilde)
                    r2_test = np.nan
                    failed = False
                except NodeParseError:
                    prefix = ""
                    r2_train = np.nan
                    r2_test = np.nan
                    failed = True
                results.append({"dataset": name, "expression": prefix , "r2_train": r2_train, "r2_test": r2_test, "failed": failed})

        results_df = pd.DataFrame(results)
        scores["r2_train_mean"]=results_df["r2_train"].mean()
        scores["r2_train_median"]=results_df["r2_train"].median()
        scores["r2_test_mean"]=results_df["r2_test"].mean()
        scores["r2_test_median"]=results_df["r2_test"].median()
        scores["decoded_failed"]=results_df["failed"].mean()

        return scores

def main(params):

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)
    scores = {}
    save = params.save_results

    if params.eval_in_domain:
        evaluator.set_env_copies(["valid1"])
        scores = evaluator.evaluate_in_domain(
            "valid1",
            "functions",
            save=save,
            logger=logger,
            ablation_to_keep=params.ablation_to_keep,
        )
        logger.info("__log__:%s" % json.dumps(scores))

    if params.eval_on_pmlb:
        target_noise = params.target_noise
        random_state = params.random_state
        data_type = params.pmlb_data_type

        if data_type == "feynman":
            filter_fn = lambda x: x["dataset"].str.contains("feynman")
        elif data_type == "strogatz":
            print("Strogatz data")
            filter_fn = lambda x: x["dataset"].str.contains("strogatz")
        elif data_type == "603_fri_c0_250_50":
            filter_fn = lambda x: x["dataset"].str.contains("603_fri_c0_250_50")
        else:
            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman")
            )

        pmlb_scores = evaluator.evaluate_pmlb(
            target_noise=target_noise,
            verbose=params.eval_verbose_print,
            random_state=random_state,
            save=save,
            filter_fn=filter_fn,
            logger=logger,
            save_file=None,
            save_suffix="eval_pmlb.csv",
        )
        logger.info("__pmlb__:%s" % json.dumps(pmlb_scores))


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()
    # params.reload_checkpoint = "/checkpoint/sdascoli/symbolicregression/shift_all/use_skeleton_True_use_sympy_False_tokens_per_batch_10000_n_enc_layers_4_n_dec_layers_16"
    params.reload_checkpoint = "/checkpoint/sdascoli/symbolicregression/shift_all/use_skeleton_False_use_sympy_False_tokens_per_batch_10000_n_enc_layers_4_n_dec_layers_16/"
    # params.reload_checkpoint = "/checkpoint/sdascoli/symbolicregression/newgen/use_skeleton_False_use_sympy_False_tokens_per_batch_10000_n_enc_layers_4_n_dec_layers_16/"
    pk = pickle.load(open(params.reload_checkpoint + "/params.pkl", "rb"))
    pickled_args = pk.__dict__
    for p in params.__dict__:
        if p in pickled_args and p not in ["dump_path", "reload_checkpoint"]:
            params.__dict__[p] = pickled_args[p]

    params.multi_gpu = False
    params.is_slurm_job = False
    params.eval_on_pmlb = True  # True
    params.eval_in_domain = False
    params.local_rank = -1
    params.master_port = -1
    params.num_workers = 1
    params.target_noise = 0.0
    params.max_input_points = 200
    params.random_state = 14423
    params.max_number_bags = 10
    params.save_results = False
    params.eval_verbose_print = True
    params.beam_size = 1
    params.rescale = True
    params.max_input_points = 200
    params.pmlb_data_type = "black_box"
    params.n_trees_to_refine = 10
    main(params)
