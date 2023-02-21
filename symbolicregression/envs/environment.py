# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from distutils.log import INFO
from logging import getLogger
import os
import io
import sys
import copy
import json
from turtle import xcor
import pandas as pd
import operator
from typing import Optional, List, Dict, Tuple
from collections import deque, defaultdict
import time
import traceback
from pathlib import Path
# import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from symbolicregression.envs.function_utils import zip_dic, ZMQNotReady, ZMQNotReadySample, TrainReader
from symbolicregression.envs import ExpressionGenerator, ExpressionGeneratorArgs, Node
from symbolicregression.envs.features import sample_features_from_mixture

from symbolicregression.model.embedders import conv_out_len

import numpy as np
from typing import Optional, Dict
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import collections
import math
import scipy
import random, string

def rstr(length: int = 6) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))

logger = getLogger()

SKIP_ITEM = "SKIP_ITEM"

def batch_expressions(expression_encoder, word2id: Dict[str, int],  expressions: Node):
    """
    Take as input a list of n sequences (torch.LongTensor vectors) and return
    a tensor of size (slen, n) where slen is the length of the longest
    sentence, and a vector lengths containing the length of each sentence.
    """
    encoded_expressions = [expression_encoder.encode(expr) for expr in expressions]
    lengths = torch.LongTensor([2 + len(encoded_expr) for encoded_expr in encoded_expressions])
    sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(word2id["<PAD>"])
    sent[0] = word2id["<EOS>"]
    for i, encoded_expr in enumerate(encoded_expressions):
        sent[1 : lengths[i] - 1, i].copy_(torch.LongTensor([word2id[tok] for tok in encoded_expr]))
        sent[lengths[i] - 1, i] = word2id["<EOS>"]
    return sent, lengths

def create_train_iterator(env, data_path, params, **args):
    """
    Create a dataset for this environment.
    """
    logger.info(f"Creating train iterator")

    def size_fn(sample):

        flat_size = (sample["is_train"].sum()) * ((sample["x"].shape[1]+1)*3 + 1 - int(params.use_emb_positional_embeddings))
        if params.embedder_type == "flat":
            return flat_size
        elif params.embedder_type == "conv":
            kernel_size = params.emb_conv_kernel
            stride = params.emb_conv_stride
            dilation = 1
            padding = params.emb_conv_kernel-1
            conv_size = conv_out_len([flat_size], kernel_size, stride, dilation, padding).item()
            return conv_size


   # size_fn = lambda batch: batch
    dataset = EnvDataset(
        env=env,
        train=True,
        size_fn=size_fn,
        params=params,
        path=data_path,
        skip=params.queue_strategy is not None,
        **args,
    )
    if params.queue_strategy is None:
        collate_fn = dataset.collate_fn
    else:
        collate_fn = dataset.collate_reduce_padding(
                dataset.collate_fn,
                key_fn=size_fn,
                max_size=None
            )

    return DataLoader(
        dataset,
        timeout=(0 if params.num_workers == 0 else 3600),
        batch_size=params.batch_size,
        num_workers=(
            params.num_workers
            if data_path is None or params.num_workers == 0
            else 1
        ),
        shuffle=False,
        collate_fn=collate_fn,
    )

def create_test_iterator(env, data_path, folder, params, **args):
    """
    Create a dataset for this environment.
    """
    logger.info(f"Creating test iterator")

   # size_fn = lambda batch: batch
    dataset = EnvDataset(
        env=env,
        train=False,
        size_fn=None,
        params=params,
        path=data_path,
        folder=folder,
        size=params.eval_size,
        skip=False,
        **args,
    )
    collate_fn = dataset.collate_fn   
    return DataLoader(
        dataset,
        timeout=0,
        batch_size=params.batch_size_eval,
        num_workers=1,
        shuffle=False,
        collate_fn=collate_fn,
    )


class EnvDataset(Dataset):
    def __init__(
        self,
        env,
        params,
        size_fn,
        train: bool = True,
        path: str = "",
        folder: str = "",
        skip=False,
        size=None,
        type=None,
        **args,
    ):
        super(EnvDataset).__init__()
        self.env = env
        self.train=train
        self.size_fn = size_fn
        self.skip = skip
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.folder = folder
        self.files = None
        self.count = 0
        self.remaining_data = 0
        self.type = type
        self.params = params
        self.errors = defaultdict(int)

        assert not params.batch_load or params.reload_size > 0
        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        self.collate_queue: Optional[List] = [] if self.train else None
        self.collate_queue_size = params.collate_queue_size
        self.tokens_per_batch = params.tokens_per_batch

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10_000
        else:
            assert size > 0
            self.size = size

    def collate_size_fn(self, batch) -> int:
        if len(batch) == 0: 
            return 0
        return len(batch) * max([self.size_fn(sample) for sample in batch])
        
    def load_chunk(self):
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, "
            f"basepos {self.basepos}"
        )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.params.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, "
            f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def collate_reduce_padding(self, collate_fn, key_fn, max_size=None):
        if self.params.queue_strategy == None:
            return collate_fn

        f = self.collate_reduce_padding_uniform

        def wrapper(b):
            try:
                return f(collate_fn=collate_fn, key_fn=key_fn, max_size=max_size,)(b)
            except ZMQNotReady:
                return ZMQNotReadySample()

        return wrapper

    def _fill_queue(self, n: int):
        """
        Add elements to the queue (fill it entirely if `n == -1`)
        Optionally sort it (if `key_fn` is not `None`)
        Compute statistics
        """
        assert self.train, "Not Implemented"
        assert (
            len(self.collate_queue) <= self.collate_queue_size
        ), "Problem with queue size"

        # number of elements to add
        n = self.collate_queue_size - len(self.collate_queue) if n == -1 else n
        assert n > 0, "n<=0"

        for _ in range(n):
            if self.path == "":
                sample = self.generate_sample()
            else:
                sample = self.read_sample()
            self.collate_queue.append(sample)

        # sort sequences
        self.collate_queue.sort(key=self.size_fn)

    def collate_reduce_padding_uniform(self, collate_fn, key_fn, max_size=None):
        """
        Stores a queue of COLLATE_QUEUE_SIZE candidates (created with warm-up).
        When collating, insert into the queue then sort by key_fn.
        Return a random range in collate_queue.
        @param collate_fn: the final collate function to be used
        @param key_fn: how elements should be sorted (input is an item)
        @param size_fn: if a target batch size is wanted, function to compute the size (input is a batch)
        @param max_size: if not None, overwrite params.batch.tokens
        @return: a wrapped collate_fn
        """

        def wrapped_collate(sequences: List):

            if not self.train:
                return collate_fn(sequences)

            # fill queue

            assert all(seq == SKIP_ITEM for seq in sequences)
            assert (
                len(self.collate_queue) < self.collate_queue_size
            ), "Queue size too big, current queue size ({}/{})".format(
                len(self.collate_queue), self.collate_queue_size
            )
            self._fill_queue(n=-1)
            assert (
                len(self.collate_queue) == self.collate_queue_size
            ), "Fill has not been successful"

            # select random index
            before = self.env.rng.randint(-self.batch_size, len(self.collate_queue))
            before = max(min(before, len(self.collate_queue) - self.batch_size), 0)
            after = self.get_last_seq_id(before, max_size)

            # create batch / remove sampled sequences from the queue
            to_ret = collate_fn(self.collate_queue[before:after])
            self.collate_queue = (
                self.collate_queue[:before] + self.collate_queue[after:]
            )
            return to_ret

        return wrapped_collate

    def get_last_seq_id(self, before: int, max_size: Optional[int]) -> int:
        """
        Return the last sequence ID that would allow to fit according to `size_fn`.
        """
        max_size = self.tokens_per_batch if max_size is None else max_size

        if max_size < 0:
            after = before + self.batch_size
        else:
            after = before
            while (
                after < len(self.collate_queue)
                and self.collate_size_fn(self.collate_queue[before:after]) < max_size
            ):
                after += 1
            # if we exceed `tokens_per_batch`, remove the last element
            size = self.collate_size_fn(self.collate_queue[before:after])
            if size > max_size:
                if after > before + 1:
                    after -= 1
                else:
                    logger.warning(
                        f"Exceeding tokens_per_batch: {size} "
                        f"({after - before} sequences)"
                    )
        return after

    def collate_fn(self, data):
        """
        Collate samples into a batch.
        """
        if data is None:
            return None

        samples = zip_dic(data)
        return samples

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.env.rng is not None:
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [worker_id, self.params.global_rank, self.env_base_seed]
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{seed} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [
                worker_id,
                self.params.global_rank,
                self.env_base_seed,
            ]
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                f"Initialized test generator, with seed {seed} (random state: {self.env.rng})"
            )

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0), "issue in worker id"
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def init_folder(self, folder):
        if self.files is not None:
            return
        self.files = list(Path(folder).glob("*/*.tsv.gz"))
        print(f"Read {folder}. Found {len(self.files)} datasets")


    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path != "":

            if self.train and self.skip:
                return SKIP_ITEM
            else:
                return self.read_sample()

        if self.folder != "":
            self.init_folder(self.folder)
            try:
                sample = self.read_file(index)
            except IndexError:
                return None
            return sample

        else:
            if self.train and self.skip:
                return SKIP_ITEM
            else: 
                sample = self.generate_sample()
                return sample


    def generate_sample(self):
        n_observations = np.random.randint(self.params.n_min_observations, self.params.n_max_observations)
        n_ops = np.random.randint(self.params.min_ops, self.params.max_ops)
        max_n_vars = np.random.randint(self.params.min_vars, self.params.max_vars+1)
        expr, (x, y) = self.env.get_sample(n_observations=n_observations+100, n_ops=n_ops, max_n_vars=max_n_vars)
        is_train = np.full(n_observations+100, True)
        is_train[n_observations:]=False
        sample = {
            "name": f"generated_{rstr(8)}", 
            "expression": expr, 
            "x": x, 
            "y": y, 
            "is_train": is_train,
            "is_train_or_valid": is_train
            }
        return sample

    def read_sample(self):

        if not hasattr(self, "train_reader"):
            self.train_reader = TrainReader(
                paths=list(Path(self.path).glob("shard.*.jsonl")),
                rng=self.env.rng,
                buffer_size=10_000,
                shuffle=True,
                start=max(0, self.params.global_rank),
                step=max(1, self.params.world_size),
                debug=True
            )

        line = next(self.train_reader)
        sample = json.loads(line)
        n_observations = np.random.randint(self.params.n_min_observations, self.params.n_max_observations)
        is_train = np.full(n_observations, True)

        expr = Node.from_prefix(sample["prefix"])
        ne_expr = expr.to_numexpr()
        x = sample_features_from_mixture(self.env.rng, feature_dim=len(sample["x"]), n=n_observations)
        y = ne_expr({f"x_{i}" : x[:,i] for i in range(len(sample["x"]))})
            
        sample.update({
            "name": f"generated_{rstr(8)}", 
            "expression": expr, 
            "x": x, 
            "y": y, 
            "is_train": is_train,
            "is_train_or_valid": is_train
            })

        return sample

    def read_file(self, index):
        file = self.files[index]
        name = file.name.split(".")[0]
        x, y, _ = read_csv_file(str(file), nrows=1_000)

        feature_dim = x.shape[1]
        assert feature_dim<=10, f"Found more than 10 dim in {name}"

        if feature_dim > self.params.max_vars:
            ##TODO: add feature selection
            selected_features = np.random.choice(np.arange(feature_dim), size=self.params.max_vars, replace=False)
            x = x[:, selected_features]

        is_nan = np.isnan(x).any(axis=1)
        x, y = x[~is_nan], y[~is_nan]

        train_or_valid_idxs, test_idxs = train_test_split(np.arange(len(x)), train_size=0.75, test_size=0.25, random_state=self.env.rng)
        is_train_or_valid = np.full(len(x), True)
        is_train_or_valid[test_idxs] = False
        train_idxs = np.random.choice(train_or_valid_idxs, size=min(self.params.n_max_observations, len(train_or_valid_idxs)), replace=False)
        is_train = np.full(len(x), False)
        is_train[train_idxs]=True

        scaler = StandardScaler()
        scaler.fit(x[train_or_valid_idxs])
        x = scaler.transform(x)

        sample = {
            "name": name, 
            "x": x, 
            "y": y, 
            "is_train": is_train, 
            "is_train_or_valid": is_train_or_valid
            }

        return sample



def read_csv_file(
    filename: str, label: str = "target", nrows: int = 9999, sep: str = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None
    input_data = pd.read_csv(
        filename, sep=sep, compression=compression, nrows=nrows, engine="python"
    )
    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]
    return X, y, feature_names