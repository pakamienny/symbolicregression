from typing import Optional, Tuple, List, Set, Dict
from dataclasses import dataclass, field
from functools import cached_property
from params import Params
import numpy as np

from symbolicregression.envs.graph import *
from symbolicregression.envs.features import sample_features_from_uniform
from symbolicregression.envs.rejector import check_constraints



@dataclass
class ExpressionGeneratorArgs(Params):
    n_vars: int = field(default=5, metadata={"help": "Number of variables"})
    unary_ops_str: str = field(
        default="sin,square", metadata={"help": "Unary operators. Empty string for all."}
    )
    binary_ops_str: str = field(
        default="+,-,/,*",
        metadata={"help": "Binary operators. Empty string for all."},
    )
    leaf_probs_str: str = field(
        default="0.5,0.5",
        metadata={
            "help": "Leaf probabilities (float, variable). Must sum to 1"
        },
    )

    @cached_property
    def unary_ops(self) -> List[str]:
        if self.unary_ops_str == "":
            ops = SUPPORTED_UNARY_OPS
        else:
            ops = [x for x in self.unary_ops_str.split(",") if len(x) > 0]
            assert all(op in SUPPORTED_UNARY_OPS for op in ops)
        return ops

    @cached_property
    def binary_ops(self) -> List[str]:
        if self.binary_ops_str == "":
            ops = SUPPORTED_BINARY_OPS
        else:
            ops = [x for x in self.binary_ops_str.split(",") if len(x) > 0]
            assert len(ops) == len(set(ops))
            assert all(op in SUPPORTED_BINARY_OPS for op in ops), ops

        return ops

    @property
    def leaf_probs(self) -> Tuple[float, float, float]:
        p = [float(x) for x in self.leaf_probs_str.split(",")]
        assert len(p) == 2, p
        assert all(x >= 0 for x in p) and abs(sum(p) - 1) < 1e-7, p
        assert (self.n_vars > 0) == (p[1] > 0), f"got {self.n_vars} vars"
        return p[0], p[1]

    def __post_init__(self):
        _ = self.unary_ops
        _ = self.binary_ops
        _ = self.leaf_probs

class ExpressionGenerator:

    MAX_OPS = 100

    def __init__(
        self,
        n_vars: int,
        unary_ops: List[str],
        binary_ops: List[str],
        leaf_probs: Tuple[float, float] = (
            0.0,
            1.0,
        ),
        seed=None,
    ):
        self.n_vars = n_vars
        self.vars = [f"x_{i}" for i in range(n_vars)]

        # leaf probabilities
        assert (
            len(leaf_probs) == 2
            and all(p >= 0 for p in leaf_probs)
            and abs(sum(leaf_probs) - 1) < 1e-7
        ), f"{leaf_probs} {type(leaf_probs)}"
        self.leaf_probs = np.array(leaf_probs, dtype=np.float32)

        assert self.n_vars  > 0 if self.leaf_probs[1] > 0 else True, self.leaf_probs

        # operators / tree distributions
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops

        self.unary = len(self.unary_ops) > 0
        self.distrib = self.generate_dist(ExpressionGenerator.MAX_OPS)

        # environment random generator
        assert seed is None or type(seed) is int and seed >= 0
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def build(args: ExpressionGeneratorArgs, seed: Optional[int] = None):
        assert type(args) is ExpressionGeneratorArgs
        return ExpressionGenerator(
            n_vars=args.n_vars,
            unary_ops=args.unary_ops,
            binary_ops=args.binary_ops,
            leaf_probs=args.leaf_probs,
            seed=seed,
        )

    def set_rng(self, rng):
        old_rng, self.rng = self.rng, rng
        return old_rng

    def get_float(self):
        """
        Generate a random float.
        """
        v = self.rng.randn()
        return v

    def generate_leaf(
        self,
        imposed_type: Optional[NodeType] = None,
        max_n_vars: Optional[int] = None,
    ):
        """
        Generate a random leaf.
        """
        leaf_types: List[NodeType] = [FLOAT, VARIABLE]

        if imposed_type is None:
            n_type = self.rng.choice(leaf_types, p=self.leaf_probs)  # type: ignore
        else:
            assert imposed_type in leaf_types
            n_type = imposed_type

        if n_type is FLOAT:
            value = "C_0" #self.get_float()
        elif n_type is VARIABLE:
            if max_n_vars is not None:
                rand_idx = self.rng.randint(min(max_n_vars, len(self.vars)))
            else:
                rand_idx = self.rng.randint(len(self.vars))
            value = self.vars[rand_idx]
        else:
            raise NotImplementedError
        return n_type, value

    def generate_operator(self, arity: int) -> Tuple[NodeType, str]:
        """
        Generate a random operator.
        """
        assert arity in [1, 2]
        value = self.rng.choice(
            self.unary_ops if arity == 1 else self.binary_ops,
        )
        ntype = UNARY if arity == 1 else BINARY
        return ntype, value

    def generate_dist(self, max_ops: int) -> List[List[int]]:
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n - 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D: List[List[int]] = [[0] + [1 for _ in range(1, 2 * max_ops + 1)]]
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        return D

    def sample_next_pos(self, n_empty: int, n_ops: int) -> Tuple[int, int]:
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `n_empty` - 1}.
        """
        assert n_empty > 0
        assert n_ops > 0
        scores = []
        if self.unary:
            for i in range(n_empty):
                scores.append(self.distrib[n_ops - 1][n_empty - i])
        for i in range(n_empty):
            scores.append(self.distrib[n_ops - 1][n_empty - i + 1])
        probs = [p / self.distrib[n_ops][n_empty] for p in scores]
        p = np.array(probs, dtype=np.float64)
        e: int = self.rng.choice(len(p), p=p)
        arity = 1 if self.unary and e < n_empty else 2
        e %= n_empty
        return e, arity

    def generate_expr(
        self,
        n_ops: int,
        max_n_vars=int,
    ) -> Node:
        """
        Generate a random expression.
        """
        assert n_ops <= ExpressionGenerator.MAX_OPS
        tree = Node()
        empty_nodes = [tree]
        next_en = 0
        n_empty = 1
        while n_ops > 0:
            next_pos, arity = self.sample_next_pos(n_empty, n_ops)
            for n in empty_nodes[next_en : next_en + next_pos]:
                n.type, n.value = self.generate_leaf(
                    max_n_vars=max_n_vars
                )
            next_en += next_pos
            (
                empty_nodes[next_en].type,
                empty_nodes[next_en].value,
            ) = self.generate_operator(arity)
            for _ in range(arity):
                e = Node()
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            n_empty += arity - 1 - next_pos
            n_ops -= 1
            next_en += 1
        for n in empty_nodes[next_en:]:
            n.type, n.value = self.generate_leaf(
                imposed_type=None,
                max_n_vars=max_n_vars,
            )
        return tree


    def sample_expression(
        self,
        n_ops: int,
        max_n_vars=int,
    ) -> Node:
        expr = self.generate_expr(n_ops, max_n_vars)

        def relabel_constant(node: Node):
            for _, child in enumerate(node.children):
                relabel_constant(child)
            if node.is_constant_expression():
                    node.value = "C_0"
                    node.children=[]

        def fill_float(node: Node):
            if node.is_constant_placeholder():
                node.value = self.get_float()
            else:
                for _, child in enumerate(node.children):
                    fill_float(child)

        relabel_constant(expr)
        fill_float(expr)
        ## relabel constant subtrees by a constant.
        return expr

    def get_sample(self, n_observations: int, n_ops: int = 15, max_n_vars: int = 5):
        
        while True:
            expr = self.sample_expression(n_ops, max_n_vars)
            if not check_constraints(expr):
                continue

            ne_expr = expr.to_numexpr()
            x = sample_features_from_uniform(self.rng, limits=(-10,10), feature_dim=max_n_vars, n=n_observations)
            y = ne_expr({f"x_{i}" : x[:,i] for i in range(max_n_vars)})
            if np.any(np.isnan(y)) or np.max(np.abs(y))>1e5:
                continue
            break
        return expr, (x,y)
    
    def get_symbols(self):
        unary_ops = list(SUPPORTED_UNARY_OPS.keys())
        binary_ops = list(SUPPORTED_BINARY_OPS.keys())
        return VARIABLES + SPECIAL_CONSTANTS + ["C_0"] + unary_ops + binary_ops


if __name__ == "__main__":
    env_args = ExpressionGeneratorArgs()
    env = ExpressionGenerator.build(env_args)
    expr = env.generate_expr(10, 10)
    print(expr)