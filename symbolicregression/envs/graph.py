from typing import Optional, Tuple, List, Set, Dict, Union
from enum import Enum
from functools import partial
import numexpr, copy
import re
import numpy as np
import sympy as sp
from symbolicregression.envs.utils import *

class NodeType(Enum):
    UNDEFINED = 0
    VARIABLE = 1
    SPECIAL_CONSTANT = 2
    FLOAT = 3
    UNARY = 4
    BINARY = 5    
    CONSTANT_PLACEHOLDER = 6


UNDEFINED = NodeType.UNDEFINED
FLOAT = NodeType.FLOAT
VARIABLE = NodeType.VARIABLE
SPECIAL_CONSTANT = NodeType.SPECIAL_CONSTANT
UNARY = NodeType.UNARY
BINARY = NodeType.BINARY
CONSTANT_PLACEHOLDER = NodeType.CONSTANT_PLACEHOLDER

VARIABLES = [f"x_{i}" for i in range(10)]
CONSTANT_PLACEHOLDERS = [f"C_{i}" for i in range(100)]
SPECIAL_CONSTANTS = []

TOK_TYPE = {
    x: y
    for x, y in [(variable, VARIABLE) for variable in VARIABLES]
    + [(u_ops, UNARY) for u_ops in SUPPORTED_UNARY_OPS]
    + [(b_ops, BINARY) for b_ops in SUPPORTED_BINARY_OPS]
    + [(c, CONSTANT_PLACEHOLDER) for c in CONSTANT_PLACEHOLDERS]

}

class NodeParseError(Exception):
    pass

class SympyException(Exception):
    pass


class Node:
    def __init__(self, 
            value = None, 
            children: Optional[List] = None,
            ):

        self.value = value
        self.children: List[Node] = [] if children is None else children

    def push_child(self, child):
        self.children.append(child)

    def infix(self) -> str:
        if len(self.children) == 0:
            return str(self.value)
        elif len(self.children) == 1:
            c = self.children[0].infix()
            return SUPPORTED_UNARY_OPS[self.value]["infix"](c)
        elif len(self.children) == 2:
            c0 = self.children[0].infix()
            c1 = self.children[1].infix()
            return SUPPORTED_BINARY_OPS[self.value]["infix"](c0, c1)
        else:
            raise Exception(f"Too many children: {len(self.children)}")

    def prefix(self):
        s = str(self.value)
        for c in self.children:
            s += "," + c.prefix()
        return s

    @classmethod
    def from_prefix(cls, prefix: str) -> "Node":
        splitted_prefix = prefix.split(",")

        def aux(offset: int) -> Tuple["Node", int]:
            if offset >= len(splitted_prefix):
                raise NodeParseError(
                    f"Missing token, parsing {' '.join(splitted_prefix)}"
                )
            tok = splitted_prefix[offset]

            tok_type = TOK_TYPE.get(tok, None)  
            if tok_type is BINARY:
                lhs, rhs_offset = aux(offset + 1)
                rhs, next_offset = aux(rhs_offset)
                return Node(tok, [lhs, rhs]), next_offset
            elif tok_type is UNARY:
                term, next_offset = aux(offset + 1)
                return Node(tok, [term]), next_offset
            else:
                return Node(tok), offset + 1

        node, last_offset = aux(0)
        if last_offset != len(splitted_prefix):
            raise NodeParseError(
                f"Didn't parse everything: {prefix}. "
                f"Stopped at length {last_offset}"
            )
        return node

    def to_torch(self):
        raise NotImplementedError

    def to_sympy(self):
        infix = self.infix()
        eq_vars = set(re.findall(r"\bx[0-9]", infix))
        constant_vars = set(re.findall(r"\bC[0-100]", infix))
        _locals = {eq_var: sp.Symbol(eq_var, real=True) for eq_var in eq_vars}
        _locals.update(
            {
                constant_var: sp.Symbol(constant_var, real=True)
                for constant_var in constant_vars
            }
        )
        sp_eq = sp.sympify(infix, locals=_locals, evaluate=True,).evalf()
        if sp_eq.has(
            sp.oo,
            -sp.oo,
            -sp.zoo,
            sp.zoo,
            sp.S.Infinity,
            sp.S.NegativeInfinity,
            sp.I,
            sp.conjugate,
            sp.AccumBounds,
            sp.StrictLessThan,
            sp.nan,
        ):
            raise SympyException(f"Unexpected symbols when parsing {infix}. Got {sp_eq}")
        return sp_eq
        
    def to_numexpr(self):

        def evaluate_numexpr(infix, local_dict):
            return numexpr.evaluate(infix, local_dict)

        return partial(evaluate_numexpr, self.infix())



    def get_torch_str(self):
        if self.is_unary():
            torch_fn = SUPPORTED_UNARY_OPS[self.value]["torch"]
            return f"{torch_fn}({self.children[0].get_torch_str()})"
        elif self.is_binary():
            torch_fn = SUPPORTED_BINARY_OPS[self.value]["torch"]
            return f"{torch_fn}({self.children[0].get_torch_str()},{self.children[1].get_torch_str()})"
        elif self.is_var():
            idx = int(self.value.split("_")[-1])
            return f"x[:,{idx}]"
        elif self.is_constant_placeholder():
            idx = int(self.value.split("_")[-1])
            return f"coeffs[{idx}]"
        else:
            return str(self.value)

    def to_torch(self):    
        def evaluate_torch(eval_str, x):
            return eval(eval_str)
        return partial(evaluate_torch, self.get_torch_str())

    def evaluate(self, x, simulator="numexpr") -> np.ndarray:
        assert simulator in ["numexpr", "torch", "fast_eval"]
        ##TODO: check dimension is correct compared to x.shape[1]
        if simulator == "numexpr":
            ne_expr = self.to_numexpr()
            x_dico = {f"x_{i}" : x[:,i] for i in range(x.shape[1])}
            return ne_expr(x_dico)
        else:
            raise NotImplementedError

    def eq(self, node) -> bool:
        """
        Check if two trees are exactly equal, i.e. two expressions are exactly equal (strong equality)
        """
        return (
            self.type == node.type
            and self.value == node.value
            and len(self.children) == len(node.children)
            and all(c1.eq(c2) for c1, c2 in zip(self.children, node.children))
        )

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def is_var(self):
        tok_type = TOK_TYPE.get(self.value, None)
        return tok_type is VARIABLE

    def is_constant_placeholder(self):
        tok_type = TOK_TYPE.get(self.value, None)
        return tok_type is CONSTANT_PLACEHOLDER

    def is_constant_expression(self):
        if self.is_leaf():
            if self.is_var():
                return False
            else: 
                return True
        else:
            return all([child.is_constant_expression() for child in self.children])

    def is_leaf(self):
        return len(self.children)==0

    def is_unary(self):
        return len(self.children)==1

    def is_binary(self):
        return len(self.children)==2

    def _get_ops(self, ops: Set[str]) -> None:
        if self.is_unary() or self.is_binary():
            ops.add(self.value)
        for c in self.children:
            c._get_ops(ops)

    def get_ops(self):
        res: Set[str] = set()
        self._get_ops(res)
        return res

    def _get_vars(self, vars: Set[str]) -> None:
        if self.is_var():
            vars.add(self.value)
        for c in self.children:
            c._get_vars(vars)
    
    def get_vars(self):
        res: Set[str] = set()
        self._get_vars(res)
        return res

    def replace_all(self, to_replace: Dict[str, Union[float, str]], inplace=False):
        if inplace:
            obj = self
        else:
            obj = self.copy()
        if obj.value in to_replace:
            obj.value = to_replace[self.value]
        for c in obj.children:
            c.replace_all(to_replace, inplace=True)
        return obj

    def copy(self):
        return copy.deepcopy(self) 
        
    def skeletonize(self):
        idx = 0
        skeleton_expr_prefix = []
        constants = {}
        for elem in self.prefix().split(","):
            tok_type = TOK_TYPE.get(elem, None)
            if  tok_type is None:
                constants[f"C_{idx}"]=float(elem)
                skeleton_expr_prefix.append(f"C_{idx}")
                idx += 1
            else:
                skeleton_expr_prefix.append(elem)
        return Node.from_prefix(",".join(skeleton_expr_prefix)), constants
        
if __name__ == "__main__":
    print("Entering tests")

    def test_graph():
        x = Node("pow2")
        x.push_child(Node(1.3))
        assert  x.prefix() == "pow2,1.3"

        x = Node("-")
        x.push_child(Node("x"))
        x.push_child(Node(1.3))
        assert  x.prefix() == "-,x,1.3", x.prefix()
        print("passed graph construct test")
    test_graph()
    