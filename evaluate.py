# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from decimal import Underflow
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
from symbolicregression.envs.environment import create_test_iterator
from symbolicregression.envs import Node, NodeParseError
from sklearn.model_selection import train_test_split
import pandas as pd

from tqdm import tqdm
import time

np.seterr(all="raise")

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

    def evaluate(self, iterator, params, save_name: str, logger=None):

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
        output_id2word = self.modules["output_id2word"]
        output_tokenizer = self.modules["output_tokenizer"]

        if params.multi_gpu:
            embedder, encoder, decoder = embedder.module, encoder.module, decoder.module
       
        embedder.eval()
        encoder.eval()
        decoder.eval()

        results = []

        for samples in iterator:
            if samples is None:
                break
            start_time = time.time()
            xys = [(xi[mask], yi[mask]) for xi, yi, mask in zip(samples["x"], samples["y"], samples["is_train"])] 
            n_datasets = len(xys)
            names = samples["name"] ##will need to duplicate if use more than 1 sample

            with torch.no_grad():
                
                x, x_len = embedder(xys)
                encoded = encoder("fwd", x=x, lengths=x_len, causal=False).transpose(0, 1)
                generations, _ = decoder.generate(encoded, x_len, sample_temperature=None, max_len=params.max_generated_output_len) ##TODO: support beam search / sampling
                generations = generations.transpose(0, 1)

            for dataset_id, name, generation in zip(np.arange(n_datasets), names, generations):
                words = [output_id2word[tok.item()] for tok in generation]
                while words[-1] == "<PAD>": words.pop(-1)
                assert words[0]=="<EOS>" and words[-1]=="<EOS>"
                try:
                    X, Y = samples["x"][dataset_id], samples["y"][dataset_id]
                    is_train = samples["is_train"][dataset_id] 
                    xtrain, xtest, ytrain, ytest = X[is_train], X[~is_train], Y[is_train], Y[~is_train] 
                    decoded_expression: Node = output_tokenizer.decode(words[1:-1])

                    prefix = decoded_expression.prefix()
                    ytilde_train = decoded_expression.evaluate(xtrain)
                    ytilde_test = decoded_expression.evaluate(xtest)
                    r2_train = stable_r2_score(ytrain, ytilde_train)
                    r2_test = stable_r2_score(ytest, ytilde_test)
                    failed = np.nan

                except (NodeParseError, ZeroDivisionError, OverflowError, FloatingPointError) as e:
                    prefix = ""
                    r2_train = -np.inf
                    r2_test = -np.inf
                    failed = str(e)

                result = {"dataset": name, "expression": prefix , "r2_train": r2_train, "r2_test": r2_test, "failed": failed, "time": time.time()-start_time}
                if "expression" in samples:
                    result["ground_truth"] = samples["expression"][dataset_id]
                results.append(result)

        results_df = pd.DataFrame(results)
        scores["r2_train_mean"]=results_df["r2_train"].mean()
        scores["r2_train_median"]=results_df["r2_train"].median()
        scores["r2_test_mean"]=results_df["r2_test"].mean()
        scores["r2_test_median"]=results_df["r2_test"].median()
        scores["decoded_failed"]=(results_df["failed"].isna()).mean()
        if self.trainer.epoch % params.save_results_every == 0:
            eval_dir = Path(self.params.job_dir) / f"eval_{save_name}"
            os.makedirs(eval_dir, exist_ok=True)
            file_to_save = eval_dir / f"epoch_{self.trainer.epoch}.csv"
            results_df.to_csv(file_to_save, sep=";")
            print(f"Saved results under {file_to_save}")
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

    if params.eval_in_domain:
        test_iterator = create_test_iterator(env=env, data_path="", folder="", params=params)
        scores = evaluator.evaluate(
                test_iterator,
                params,
                save_name="in-domain",
                logger=logger,
            )
        logger.info("__log__:%s" % json.dumps(scores))

    if params.eval_on_pmlb:
        srbench_iterator = create_test_iterator(env=env, data_path="", folder=params.srbench_path, params=params)

        srbench_scores = evaluator.evaluate(
            srbench_iterator,
            params,
            save_name="pmlb",
            logger=logger,
        )
        logger.info("__pmlb__:%s" % json.dumps(srbench_scores))


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()
    params.reload_checkpoint = "/checkpoint/pakamienny/new_symbolicregression/paper/tokens_per_batch_10000_lr_0.0002_accumulate_gradients_1/2023-02-13_02-12-10/periodic-0.pth"
    # params.reload_checkpoint = "/checkpoint/sdascoli/symbolicregression/shift_all/use_skeleton_True_use_sympy_False_tokens_per_batch_10000_n_enc_layers_4_n_dec_layers_16"
    # params.reload_checkpoint = "/checkpoint/sdascoli/symbolicregression/newgen/use_skeleton_False_use_sympy_False_tokens_per_batch_10000_n_enc_layers_4_n_dec_layers_16/"
    pk = pickle.load(open(Path(params.reload_checkpoint).parent / "params.pkl", "rb"))
    pickled_args = pk.__dict__
    for p in params.__dict__:
        if p in pickled_args and p not in ["dump_path", "reload_checkpoint"]:
            params.__dict__[p] = pickled_args[p]

    params.multi_gpu = False
    params.is_slurm_job = False
    params.eval_on_pmlb = True  # True
    params.eval_in_domain = False
    params.batch_size_eval = 2
    params.eval_size = 2

    params.local_rank = -1
    params.master_port = -1
    params.num_workers = 1

    params.job_dir = params.dump_path
    print(f"Launching eval in {params.job_dir}")
    main(params)
