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
    def encode(self, datasets: List[np.ndarray]) -> List[torch.Tensor]:
        pass

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FlatEmbedder(Embedder):
    def __init__(self, float_tokenizer, word2id: Dict[str, int], params):
        # dim: int, use_cpu: bool = False):
        from .transformer import Embedding

        super().__init__()
        assert word2id["<EOS>"] == 0 and word2id["<PAD>"] == 1
       # self.dtype = torch.half if params.fp16 else torch.float
        self.dim = params.enc_emb_dim
        self.use_cpu = params.cpu
        self.float_tokenizer = float_tokenizer
        self.word2id = word2id
        self.embeddings = Embedding(
            len(word2id),
            self.dim,
            padding_idx=1,
        )
        self.positional_embeddings = Embedding(13, self.dim, padding_idx=1) #10 for dimensions, 1 for output, 1 for <PAD> and 1 for <EOS>
        self.float_positional_embeddings = Embedding(5, self.dim, padding_idx=1) #s,m,e + eos+pad
        if not self.use_cpu:
            self.embeddings = self.embeddings.cuda()
            self.positional_embeddings = self.positional_embeddings.cuda()
            self.float_positional_embeddings = self.float_positional_embeddings.cuda()


    def forward(self, xys: List[Dataset]) -> Tuple[torch.Tensor, torch.Tensor]:
        datasets = [np.concatenate([yi[:, None], xi], 1) for xi, yi in xys] 
        encoded_datasets, positional_tokens, float_positional_tokens = self.encode(datasets)
        batch_datasets, batch_len = self.batch(encoded_datasets)
        batch_positionals, _ = self.batch(positional_tokens)
        batch_float_positionals, _ = self.batch(float_positional_tokens)
        batch_datasets, batch_positionals, batch_float_positionals, batch_len = to_cuda(
            batch_datasets, batch_positionals, batch_float_positionals, batch_len, use_cpu=self.use_cpu
        )
        dataset_embeddings = self.embed(batch_datasets, batch_positionals, batch_float_positionals)
        return dataset_embeddings, batch_len

    def encode(self, datasets: List[np.ndarray]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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

def conv_out_len(len1, kernel_size, stride, dilation, padding):
    if not isinstance(len1, torch.TensorType):
        len1 = torch.Tensor(len1)
    conv_size = torch.floor(
        (len1 + 2 * padding - dilation * (kernel_size - 1) - 1).float() / stride + 1
    )
    return conv_size.long()

class ConvEmbedder(Embedder):
    def __init__(self, float_tokenizer, word2id: Dict[str, int], params):
        from .transformer import Embedding

        super().__init__()
        assert word2id["<EOS>"] == 0 and word2id["<PAD>"] == 1
       # self.dtype = torch.half if params.fp16 else torch.float
        self.dim = params.enc_emb_dim
        self.use_cpu = params.cpu
        self.float_tokenizer = float_tokenizer
        self.word2id = word2id
        self.embeddings = Embedding(
            len(word2id),
            self.dim,
            padding_idx=1,
        )
        self.use_positional_embeddings = params.use_emb_positional_embeddings
        self.pad_to_max_dim = params.use_emb_positional_embeddings

        if self.use_positional_embeddings:
            self.positional_embeddings = Embedding(13, self.dim, padding_idx=1) #10 for dimensions, 1 for output, 1 for <PAD> and 1 for <EOS>
            self.float_positional_embeddings = Embedding(5, self.dim, padding_idx=1) #s,m,e + eos+pad
            if not self.use_cpu:
                self.positional_embeddings = self.positional_embeddings.cuda()
                self.float_positional_embeddings = self.float_positional_embeddings.cuda()

        self.conv = nn.Conv1d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=(params.emb_conv_kernel,),
                stride=(params.emb_conv_stride,),
                padding=(params.emb_conv_kernel-1,) ,
            )
        
        if not self.use_cpu:
            self.embeddings = self.embeddings.cuda()
            self.conv = self.conv.cuda()

    def forward(self, xys: List[Dataset]) -> Tuple[torch.Tensor, torch.Tensor]:

        datasets = [np.concatenate([yi[:, None], xi], 1) for xi, yi in xys] 
        encoded_datasets, positional_tokens, float_positional_tokens = self.encode(datasets)
        batch_datasets, batch_len = self.batch(encoded_datasets)
        batch_positionals, _ = self.batch(positional_tokens)
        batch_float_positionals, _ = self.batch(float_positional_tokens)
        batch_datasets, batch_positionals, batch_float_positionals, batch_len = to_cuda(
            batch_datasets, batch_positionals, batch_float_positionals, batch_len, use_cpu=self.use_cpu
        )
        dataset_embeddings = self.embed(batch_datasets, batch_positionals, batch_float_positionals)
        (kernel_size,), (stride,), (dilation,), (padding,) = self.conv.kernel_size, self.conv.stride, self.conv.dilation, self.conv.padding
        batch_len = conv_out_len(batch_len, kernel_size, stride, dilation, padding)
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
                obs_toks = [] if self.use_positional_embeddings else [self.word2id["SEP1"]]                        
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
        _x = self.embeddings(batch_datasets) 
        if self.use_positional_embeddings:
            _x += self.positional_embeddings(batch_positionals) + self.float_positional_embeddings(batch_float_positionals)
        _, _bs, _dim = _x.shape
        x = _x.transpose(0, 2).transpose(0, 1)
        x = self.conv(x)
        x = x.transpose(2, 0).transpose(1, 2)
        assert x.shape[1:] == (_bs, _dim)
        return x