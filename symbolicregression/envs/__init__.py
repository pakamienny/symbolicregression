# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from symbolicregression.envs.generator import ExpressionGenerator, ExpressionGeneratorArgs
from symbolicregression.envs.graph import Node, NodeParseError
from symbolicregression.envs.environment import create_test_iterator, create_train_iterator

logger = getLogger()


def build_env(params):
    """
    Build environment.
    """
    gen_args = ExpressionGeneratorArgs( 
        n_vars=params.max_vars, 
        unary_ops_str=params.unary_ops_str,
        binary_ops_str=params.binary_ops_str,
        leaf_probs_str=params.leaf_probs_str,
        operators_upsample_str=params.operators_upsample_str    
        )
    env = ExpressionGenerator.build(args=gen_args)
    return env
