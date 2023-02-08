# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from symbolicregression.utils import to_cuda
import torch.nn.functional as F

Dataset = Tuple[np.ndarray, np.ndarray]

    
class Embedder(ABC, nn.Module):
    """
    Base class for embedders, transforms a sequence of pairs into a sequence of embeddings.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, datasets: List[Dataset]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, datasets: List[Dataset]) -> List[torch.Tensor]:
        pass

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_length_after_batching(self, datasets: List[Dataset]) -> List[int]:
        pass

class FlatEmbedder(Embedder):
    def __init__(self, float_tokenizer, word2id: Dict[str, int], dim: int, use_cpu: bool = False):
        from .transformer import Embedding

        super().__init__()
        assert word2id["<EOS>"] == 0 and word2id["<PAD>"] == 1
        self.use_cpu = use_cpu
        self.float_tokenizer = float_tokenizer
        self.word2id = word2id
        self.embeddings = Embedding(
            len(word2id),
            dim,
            padding_idx=1,
        )
        self.positional_embeddings = Embedding(13, dim, padding_idx=1) #10 for dimensions, 1 for output, 1 for <PAD> and 1 for <EOS>
        self.float_positional_embeddings = Embedding(5, dim, padding_idx=1) #s,m,e + eos+pad
        if not use_cpu:
            self.embeddings = self.embeddings.cuda()
            self.positional_embeddings = self.positional_embeddings.cuda()
            self.float_positional_embeddings = self.float_positional_embeddings.cuda()

    def forward(self, datasets: List[Dataset]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_datasets, positional_tokens, float_positional_tokens = self.encode(datasets)
        batch_datasets, batch_len = self.batch(encoded_datasets)
        batch_positionals, _ = self.batch(positional_tokens)
        batch_float_positionals, _ = self.batch(float_positional_tokens)
        batch_datasets, batch_positionals, batch_float_positionals, batch_len = to_cuda(
            batch_datasets, batch_positionals, batch_float_positionals, batch_len, use_cpu=self.use_cpu
        )
        dataset_embeddings = self.embed(batch_datasets, batch_positionals, batch_float_positionals)
        return dataset_embeddings, batch_len

    def encode(self, datasets: List[Dataset]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        datasets_toks = []
        datasets_positional_toks = []
        datasets_float_positional_toks = []

        for dataset in datasets:
            dataset_toks = []
            dataset_positional_toks = []
            dataset_float_positional_toks = []

            for observation in dataset:
                obs_toks = []
                obs_positional_toks = []
                obs_float_positional_toks = []
                for i, d in enumerate(observation):
                    encoded_d = self.float_tokenizer.encode(d)
                    obs_toks.extend([self.word2id[e] for e in encoded_d])
                    obs_positional_toks.extend([i for j in range(len(encoded_d))])
                    obs_float_positional_toks.extend([j for j in range(len(encoded_d))])
                dataset_toks.extend(obs_toks)
                dataset_positional_toks.extend(obs_positional_toks)
                dataset_float_positional_toks.extend(obs_float_positional_toks)
            datasets_toks.append(torch.LongTensor(dataset_toks))
            datasets_positional_toks.append(2+torch.LongTensor(dataset_positional_toks))
            datasets_float_positional_toks.append(2+torch.LongTensor(dataset_float_positional_toks))

        return datasets_toks, datasets_positional_toks, datasets_float_positional_toks

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.LongTensor([2+len(x) for x in seqs])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.word2id["<PAD>"])
        sent[0] = self.word2id["<EOS>"]
        for i, seq in enumerate(seqs):
            sent[1 : lengths[i] - 1, i] = seq
            sent[lengths[i] - 1, i] = self.word2id["<EOS>"]
        return sent, lengths

    def embed(self, batch_datasets: torch.LongTensor, batch_positionals: torch.LongTensor, batch_float_positionals: torch.LongTensor) -> torch.Tensor:
        return self.embeddings(batch_datasets) + self.positional_embeddings(batch_positionals) + self.float_positional_embeddings(batch_float_positionals)

    def get_length_after_batching(self, seqs: List[Dataset]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            if self.pad_to_max_dim:
                sep, d_in, d_out = (
                    0,
                    self.params.max_input_dimension,
                    self.params.max_output_dimension,
                )
            else:
                x, y = seq[0]
                sep, d_in, d_out = 2, len(x), len(y)
            lengths[i] = len(seq) * (
                (2 + self.params.mantissa_len) * (d_in + d_out) + sep
            )
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths

