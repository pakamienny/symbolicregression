# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch
from .embedders import FlatEmbedder
from .transformer import TransformerModel
from .sklearn_wrapper import SymbolicTransformerRegressor
from .model_wrapper import ModelWrapper

logger = getLogger()
import symbolicregression_env

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

    float_tokenizer = symbolicregression_env.FloatTokenizer()
    expression_tokenizer = symbolicregression_env.ExpressionTokenizer(precision=1)

    modules["input_tokenizer"]=float_tokenizer
    modules["output_tokenizer"]=expression_tokenizer

    input_symbols = ["<EOS>", "<PAD>"]
    output_symbols = ["<EOS>", "<PAD>"]

    input_symbols += float_tokenizer.get_symbols()
    output_symbols += env.get_symbols() + expression_tokenizer.get_symbols()
   
    input_id2word = {i: s for i, s in enumerate(input_symbols)}
    input_word2id = {s: i for i, s in input_id2word.items()}

    output_id2word = {i: s for i, s in enumerate(output_symbols)}

    modules["embedder"] = FlatEmbedder(float_tokenizer, input_word2id, 512) #LinearPointEmbedder(params, env)
    #env.get_length_after_batching = modules["embedder"].get_length_after_batching

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
