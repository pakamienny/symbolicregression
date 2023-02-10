from typing import List
import re
import numpy as np

from symbolicregression.tokenizers.base import Tokenizer

NAN_TOKEN = "<NAN>"
INF_TOKEN = "<INF>"
NEG_INF_TOKEN = "<-INF>"

class FloatTokenError(Exception):
    pass

class FloatTokenizer(Tokenizer):
    def __init__(
        self, precision: int = 3, replicate_special_tokens: bool = False, mantissa_len: int = 1, max_exponent: int = 100,
    ):
        super().__init__()
        self.precision = precision
        self.mantissa_len = mantissa_len
        self.max_exponent = max_exponent
        self.base = (self.precision + 1) // self.mantissa_len
        self.max_token = 10 ** self.base
        self.replicate_special_tokens = replicate_special_tokens
        assert (self.precision + 1) % self.mantissa_len == 0

        self.nan_token = NAN_TOKEN
        self.inf_token = INF_TOKEN
        self.neg_inf_token = NEG_INF_TOKEN


        self.symbols: List[str] = ["S+", "S-"]
        self.symbols.extend([f"N{i:0{self.base}d}" for i in range(self.max_token)])
        self.symbols.extend([f"E{i}" for i in range(-max_exponent, max_exponent + 1)])

        self.zero_plus = ["S+", *["N" + "0" * self.base] * mantissa_len, "E0"]
        self.zero_minus = ["S-", *["N" + "0" * self.base] * mantissa_len, "E0"]


    def get_symbols(self) -> List[str]:
        return self.symbols
        
    def encode(self, val: float) -> List[str]:
        """
        Tokenize a float number.
        """
        if val == 0.0:
            return self.zero_plus
        elif np.isnan(val):
            return [self.nan_token]*3 if self.replicate_special_tokens else  [self.nan_token]
        elif np.isinf(val):
            return [self.inf_token if val >= 0 else self.neg_inf_token]*3 if self.replicate_special_tokens else  [self.inf_token if val >= 0 else self.neg_inf_token]
   
        precision = self.precision
        str_m, str_exp = f"{val:.{precision}e}".split("e")
        m1, m2 = str_m.lstrip("-").split(".")
        m: str = m1 + m2
        assert re.fullmatch(r"\d+", m) and len(m) == precision + 1, m
        expon = int(str_exp) - precision
        if expon > self.max_exponent:
            return [self.inf_token if val >= 0 else self.neg_inf_token]*3 if self.replicate_special_tokens else  [self.inf_token if val >= 0 else self.neg_inf_token]
        if expon < -self.max_exponent:
            return self.zero_plus if val >= 0 else self.zero_minus
        assert len(m) % self.base == 0
        m_digits = [m[i : i + self.base] for i in range(0, len(m), self.base)]
        assert len(m_digits) == self.mantissa_len
        sign = "S+" if val >= 0 else "S-"
        return [sign, *[f"N{d}" for d in m_digits], f"E{expon}"]

    def decode(self, tokens: List[str]) -> float:
        """
        Detokenize a float number.
        """
    
        if tokens[0] == self.inf_token:
            return np.inf
        elif tokens[0] == self.neg_inf_token:
            return -np.inf
        elif tokens[0] == self.nan_token:
            return np.nan

        if tokens[0] not in ["S-", "S+"]:
            raise FloatTokenError(f"Unexpected first token: {tokens[0]}")
        if tokens[-1][0] != "E":
            raise FloatTokenError(f"Unexpected last token: {tokens[-1]}")

        sign = 1 if tokens[0] == "S+" else -1
        mant_str = ""
        for x in tokens[1:-1]:
            mant_str += x[1:]
        mant = int(mant_str)
        exp = int(tokens[-1][1:])
        value = sign * mant * (10 ** exp)
        return float(value)
