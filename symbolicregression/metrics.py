import numpy as np
import copy
import warnings
from sklearn.metrics import r2_score, mean_squared_error

def stable_r2_score(y, y_tilde):
    y_true = copy.deepcopy(y)
    y_pred = copy.deepcopy(y_tilde)

    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    assert y_true.shape == y_pred.shape, f"Got y {y_true.shape} and {y_pred.shape}"
    if (np.isnan(y_true) != np.isnan(y_pred)).any():
        return -np.inf
    elif (np.isinf(y_true) != np.isinf(y_pred)).any():
        return -np.inf
    elif (np.isinf(y_true) & (np.sign(y_true) != np.sign(y_pred))).any():
        return -np.inf
    to_remove = np.isnan(y_true) | np.isinf(y_true)
    y_true_finite = y_true[~to_remove]
    y_pred_finite = y_pred[~to_remove]
    if y_true_finite.shape[0] == 0 and y_pred_finite.shape[0] == 0:
        return 1.0
    if (y_true_finite == y_pred_finite).all():
        return 1.0
    elif y_true_finite.shape[0] > 0 and y_pred_finite.shape[0] > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                return r2_score(y_true_finite, y_pred_finite, force_finite=True)
            except RuntimeWarning:
                return -np.inf
    else:
        return -np.inf
