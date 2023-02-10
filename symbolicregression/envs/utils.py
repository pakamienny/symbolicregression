
import torch


torch_cos = lambda x: torch.cos(x)
torch_sin = lambda x: torch.sin(x)
torch_tan = lambda x: torch.tan(x)
torch_inv = lambda x: torch.inv(x)
torch_square = lambda x: x**2

torch_sub = lambda x, y: x-y
torch_add = lambda x, y: x+y
torch_mul = lambda x, y: x*y
torch_div = lambda x, y: x/y



SUPPORTED_UNARY_OPS = { 
    "cos": {"infix": lambda x: f"cos({x})", "torch": "torch_cos"},
    "sin": {"infix": lambda x: f"sin({x})", "torch": "torch_sin"},
    "tan": {"infix": lambda x: f"tan({x})", "torch": "torch_tan"},
    "inv": {"infix": lambda x: f"({x} ** -1)", "torch": "torch_inv"},
    "square": {"infix": lambda x: f"({x} ** 2)",  "torch": "torch_square"},
}

SUPPORTED_BINARY_OPS = {
    "-": {"infix": lambda x,y: f"({x}-{y})", "torch": "torch_sub"}, 
    "+": {"infix": lambda x,y: f"({x}+{y})", "torch": "torch_add"},
    "*": {"infix": lambda x,y: f"({x}*{y})", "torch": "torch_mul"},
    "/": {"infix": lambda x,y: f"({x}/{y})", "torch": "torch_div"},
}

SUPPORTED_OPS = {**SUPPORTED_BINARY_OPS, **SUPPORTED_UNARY_OPS}