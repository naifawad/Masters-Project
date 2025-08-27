import numpy as np
import pandas as pd
from scipy.optimize import minimize


def get_risk_parity_weights(cov_matrix):
    inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
    weights = inv_vol / np.sum(inv_vol)
    return pd.Series(weights, index=cov_matrix.columns)
