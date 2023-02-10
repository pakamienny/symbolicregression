from typing import List

from symbolicregression.tokenizers.base import Tokenizer
from symbolicregression.tokenizers.float_tokenizer import FloatTokenError, FloatTokenizer
from symbolicregression.envs import Node, NodeParseError

class ExpressionTokenizer(Tokenizer):
    def __init__(
        self, precision
    ):
        super().__init__()
        self.float_tokenizer = FloatTokenizer(precision=precision, replicate_special_tokens=True)

    def get_symbols(self) -> List[str]:
        return self.float_tokenizer.get_symbols()

    def encode(self, expression: Node) -> List[str]:
        res = []
        for x in expression.prefix().split(","):
            try:
                float(x)
                res.extend(self.float_tokenizer.encode(float(x)))
            except ValueError as e:
                res.extend([x])
        return res
        
    def decode(self, tokens: List[str]) -> Node:
        prefix = []
        k = 0
        while k < len(tokens):
            if tokens[k] in self.get_symbols():
                try:
                    prefix.append(str(self.float_tokenizer.decode(tokens[k:k+3])))
                except FloatTokenError:
                    raise NodeParseError("float could not be decoded")
                k+=3
            else:
                prefix.append(tokens[k])
                k+=1
        return Node.from_prefix(",".join(prefix))