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
import operator
from typing import Optional, List, Dict
from collections import deque, defaultdict
import time
import traceback

# import math
from symbolicregression.envs.utils import zip_dic, ZMQNotReady, ZMQNotReadySample
import symbolicregression_env
from symbolicregression_env.envs import ExpressionGenerator, ExpressionGeneratorArgs, Node
import numpy as np
from typing import Optional, Dict
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import collections
import math
import scipy

SPECIAL_WORDS = [
    "<EOS>",
    "<X>",
    "</X>",
    "<Y>",
    "</Y>",
    "</POINTS>",
    "<INPUT_PAD>",
    "<OUTPUT_PAD>",
    "<PAD>",
    "(",
    ")",
    "SPECIAL",
    "OOD_unary_op",
    "OOD_binary_op",
    "OOD_constant",
]
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

    dataset = EnvDataset(
        env=env,
        train=True,
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
                key_fn=lambda data: data["x"].shape[0]*data["x"].shape[1],
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


class EnvDataset(Dataset):
    def __init__(
        self,
        env,
        params,
        train: bool = True,
        path: str = "",
        skip=False,
        size=None,
        type=None,
        input_length_modulo=-1,
        **args,
    ):
        super(EnvDataset).__init__()
        self.env = env
        self.train=train
        self.skip = skip
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.count = 0
        self.remaining_data = 0
        self.type = type
        self.input_length_modulo = input_length_modulo
        self.params = params
        self.errors = defaultdict(int)

        if "test_env_seed" in args:
            self.test_env_seed = args["test_env_seed"]
        else:
            self.test_env_seed = None
        if "env_info" in args:
            self.env_info = args["env_info"]
        else:
            self.env_info = None

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

        # generation, or reloading from file
        print(path)
        if path != "":
            assert os.path.isfile(path), "{} not found".format(path)
            if params.batch_load:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = []
                        for i, line in enumerate(f):
                            lines.append(json.loads(line.rstrip()))
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i == params.reload_size:
                                break
                            if i % params.n_gpu_per_node == params.local_rank:
                                # lines.append(line.rstrip())
                                lines.append(json.loads(line.rstrip()))
                # self.data = [xy.split("=") for xy in lines]
                # self.data = [xy for xy in self.data if len(xy) == 3]
                self.data = lines
                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def collate_size_fn(self, batch: Dict) -> int:
        if len(batch) == 0:
            return 0
        return len(batch) * max(
            [seq["x"].shape[0] for seq in batch]
        )

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

    def _fill_queue(self, n: int, key_fn):
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
                ##TODO
                assert (
                    False
                ), "need to finish implementing load dataset, but do not know how to handle read index"
                sample = self.read_sample(index)
            self.collate_queue.append(sample)

        # sort sequences
        if key_fn is not None:
            self.collate_queue.sort(key=key_fn)

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
            self._fill_queue(n=-1, key_fn=key_fn)
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

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """

        samples = zip_dic(elements)
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
            if self.env_info is not None:
                seed += [self.env_info]
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
                self.test_env_seed if "valid" in self.type else 0,
            ]
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                "Initialized {} generator, with seed {} (random state: {})".format(
                    self.type, seed, self.env.rng
                )
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

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path == "":
            if self.train and self.skip:
                return SKIP_ITEM
            else:
                sample = self.generate_sample()
                return sample
        else:
            if self.train and self.skip:
                return SKIP_ITEM
            else:
                return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index

        def str_list_to_float_array(lst):
            for i in range(len(lst)):
                for j in range(len(lst[i])):
                    lst[i][j] = float(lst[i][j])
            return np.array(lst)

        x = copy.deepcopy(self.data[idx])
        x["x_to_fit"] = str_list_to_float_array(x["x_to_fit"])
        x["y_to_fit"] = str_list_to_float_array(x["y_to_fit"])
        x["x_to_predict"] = str_list_to_float_array(x["x_to_predict"])
        x["y_to_predict"] = str_list_to_float_array(x["y_to_predict"])
        x["tree"] = self.env.equation_encoder.decode(x["tree"].split(","))
        x["tree_encoded"] = self.env.equation_encoder.encode(x["tree"])
        infos = {}

        for col in x.keys():
            if col not in [
                "x_to_fit",
                "y_to_fit",
                "x_to_predict",
                "y_to_predict",
                "tree",
                "tree_encoded",
            ]:
                infos[col] = int(x[col])
        x["infos"] = infos
        for k in infos.keys():
            del x[k]
        return x

    def generate_sample(self):
        expr, (x, y) = self.env.get_sample()
        sample = {"expression": expr, "x": x, "y": y}
        return sample



def select_dico_index(dico, idx):
    new_dico = {}
    for k in dico.keys():
        new_dico[k] = dico[k][idx]
    return new_dico
