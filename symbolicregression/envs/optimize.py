
from typing import Optional, Tuple, List, Set, Dict
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.optimize import minimize
from functorch import grad, hessian
from functools import partial
from scipy import optimize
from torch.autograd import Variable

from symbolicregression.envs.graph import Node
from symbolicregression.envs.utils import *


def run_scipy_optimization(expr: Node, x: np.ndarray, y: np.ndarray, verbose: bool = False):
    skeleton_expr, init_constants = expr.skeletonize()
    init_values = np.array(list(init_constants.values()))
    
    def objective_fn(constants):
        expr_with_constants = skeleton_expr.replace_all({f"C_{i}": constant for i, constant in enumerate(constants)}, inplace=False)
        ne_expr = expr_with_constants.to_numexpr()
        ytilde = ne_expr({f"x_{i}" : x[:, i] for i in range(10)})
        mse = ((y - ytilde)**2).mean()/2
        return mse
    
    try:
        result = minimize(objective_fn, init_values, method='BFGS', options={"maxiter": 10, "disp": False})
    except ValueError:
        return expr

    best_constants = result['x']
    if verbose:
        print(f"MSE: {objective_fn(init_values)} -> {objective_fn(best_constants)}")
    return skeleton_expr.replace_all({f"C_{i}": constant for i, constant in enumerate(best_constants)}, inplace=False)


class TorchOptim():
    def __init__(
        self,
        skeleton_expr: Node, 
        data: Tuple[np.ndarray, np.ndarray], 
    ):
        self.skeleton_expr = skeleton_expr
        self.X = torch.Tensor(data[0])
        self.y = torch.Tensor(data[1])
        
    def solve(self, coeffs0: torch.TensorType, verbose=True):
        skeleton_expr = self.skeleton_expr
        skeleton_eval_string = skeleton_expr.get_torch_str()

        def objective_torch(coeffs):
            x = self.X
            ytilde =  eval(skeleton_eval_string)
            res = (self.y - ytilde).pow(2).mean().div(2)
            return res
    
        def objective_numpy(coeffs):
            res = objective_torch(coeffs).item()
            return res

        def grad_numpy(coeffs):
            res = grad(partial(objective_torch))(torch.tensor(coeffs)).detach().numpy()
            return res

        result = optimize.minimize(objective_numpy, x0=coeffs0, method='BFGS',  jac=grad_numpy, options={"maxiter": 10, "disp": False})
                    
        best_coeffs = result['x']
        if verbose:
            print(f"MSE: {objective_numpy(coeffs0)} -> {objective_numpy(best_coeffs)}")

        return skeleton_expr.replace_all({f"C_{i}": constant for i, constant in enumerate(best_coeffs)}, inplace=False)
            
def run_torch_optim(expr: Node, x: np.ndarray, y: np.ndarray, verbose: bool = False):    
    skeleton_expr, init_constants = expr.skeletonize()
    torch_optim = TorchOptim(skeleton_expr, (x,y))
    init_constants = list(init_constants.values())
    init_constants = torch.Tensor(init_constants)
    return torch_optim.solve(init_constants, verbose)
    