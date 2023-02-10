from typing import Optional, Tuple, List, Set, Dict
from dataclasses import dataclass, field
from functools import cached_property
from params import Params
import numpy as np

from symbolicregression.envs.graph import *


NESTED_CONSTRAINTS: Dict[str, Dict[str, int]] = {
    "square": {"square": 3},
    "cos": {"sin": 0, "cos": 0, "tan": 0, "ln": 0},
    "sin": {"sin": 0, "cos": 0, "tan": 0, "ln": 0},
    "tan": {"sin": 0, "cos": 0, "tan": 0, "ln": 0},
    "exp": {"exp": 0},
    "sqrt": {"sqrt": 0, "ln": 0},
    "abs": {"abs": 0},
    "inv": {"inv": 1},
    "ln": {"sin": 0, "cos": 0, "tan": 0, "sqrt": 0, "ln": 0},
}

MAX_SIZE = 500

class ConstraintException(Exception):
    pass

def breadth_first_traversal(root: Node) -> List[Node]:
    to_return = []
    stack = [root]
    while len(stack) > 0:
        current = stack.pop(0)
        to_return.append(current)
        if not current.is_leaf():
            stack.append(current.children[0])
            if current.is_binary():
                stack.append(current.children[1])
    return to_return
    
def compute_occurences(node: Node, op: str) -> Dict[str, int]:
    node_ops =  node.get_ops()
    if node.is_leaf():
        return {op: 0 for op in node_ops}
    elif node.value == op:
        res = {op: 0 for op in node_ops}
        for x in breadth_first_traversal(node)[1:]:
            if x.value in node_ops:
                res[x.value] += 1
        return res
    elif node.is_unary():
        return compute_occurences(node.children[0], op)
    else:
        left_occurences = compute_occurences(node.children[0], op)
        right_occurences = compute_occurences(node.children[1], op)
        res = {other_op: max(left_occurences.get(other_op, float('-inf')), right_occurences.get(other_op, float('-inf'))) for other_op in node_ops} 
        return res

def check_nested_constraints(expression: Node):
    nested_constraints = NESTED_CONSTRAINTS
    found_ops =  expression.get_ops()
    ops = found_ops.intersection(nested_constraints.keys())
    for op in ops:
        constraint = nested_constraints[op]
        op_occurence = compute_occurences(expression, op)
        other_ops = found_ops.intersection(constraint.keys())
        for other_op in other_ops:
            if op_occurence[other_op] > constraint[other_op]:
                raise ConstraintException(
                    f"ConstraintException {(op, other_op, constraint[other_op])}. Got {op_occurence[other_op]} Eq: {expression}"
                )

def check_constraints(expression: Node):
    if len(expression) > 50:
        return False
    try:
        check_nested_constraints(expression)        
        return True
    except ConstraintException:
        return False
    