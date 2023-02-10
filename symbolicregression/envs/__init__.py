# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from symbolicregression.envs.generator import ExpressionGenerator, ExpressionGeneratorArgs
from symbolicregression.envs.graph import Node, NodeParseError
from symbolicregression.envs.new_environment import create_test_iterator, create_train_iterator

logger = getLogger()


def build_env(params):
    """
    Build environment.
    """
    env_args = ExpressionGeneratorArgs()
    env = ExpressionGenerator.build(args=env_args)
    return env
