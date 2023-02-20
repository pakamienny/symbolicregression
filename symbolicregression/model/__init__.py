# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from symbolicregression.model.embedders import FlatEmbedder, ConvEmbedder
from symbolicregression.model.transformer import TransformerModel
from symbolicregression.tokenizers import FloatTokenizer
from symbolicregression.tokenizers import ExpressionTokenizer

logger = getLogger()

def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}

    float_tokenizer = FloatTokenizer(replicate_special_tokens=params.pad_to_max_dim)
    expression_tokenizer = ExpressionTokenizer(precision=1)

    modules["input_tokenizer"]=float_tokenizer
    modules["output_tokenizer"]=expression_tokenizer

    input_symbols = ["<EOS>", "<PAD>", "SEP1", "SEP2", "SEP3"]
    output_symbols = ["<EOS>", "<PAD>"]

    input_symbols += float_tokenizer.get_symbols()
    output_symbols += env.get_symbols() + expression_tokenizer.get_symbols()
   
    input_id2word = {i: s for i, s in enumerate(input_symbols)}
    input_word2id = {s: i for i, s in input_id2word.items()}
    modules["input_id2word"]=input_id2word
    modules["input_word2id"]=input_word2id

    output_id2word = {i: s for i, s in enumerate(output_symbols)}
    output_word2id = {s: i for i, s in output_id2word.items()}

    modules["output_id2word"]=output_id2word
    modules["output_word2id"]=output_word2id

    if params.embedder_type == "flat":
        modules["embedder_module"] = FlatEmbedder(float_tokenizer, word2id=input_word2id, params=params)
    elif params.embedder_type == "conv":
        modules["embedder_module"] = ConvEmbedder(float_tokenizer, word2id=input_word2id, params=params)

    modules["encoder_module"] = TransformerModel(
        params,
        input_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    if not params.cpu: 
        modules["encoder_module"] = modules["encoder_module"].cuda(
    )
    modules["decoder_module"] = TransformerModel(
        params,
        output_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )
    if not params.cpu: 
        modules["decoder_module"] = modules["decoder_module"].cuda(
    )
    # reload pretrained modules
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            if not k.endswith("_module"): continue
            assert k in reloaded
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        if k.endswith("module"):
            logger.info(
                f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
            )


    return modules
