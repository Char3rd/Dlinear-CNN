import pandas as pd
import numpy as np
from util.metrics import metric, MSE, MAE, R2, err, all_life, pre, RUL


def mse(pre, real):
    return np.mean((real - pre) ** 2)

def mae(pre, real):
    return np.mean(np.abs(real - pre))

# print(all_life.to_numpy())
# print(pre['forecast'].to_numpy())
if __name__ == '__main__':
    Err = err(all_life.to_numpy(), pre['forecast'].to_numpy(), RUL['RUL'].to_numpy())
    Mse = MSE(pre['forecast'].to_numpy(), RUL['RUL'].to_numpy())
    Mae = MAE(pre['forecast'].to_numpy(), RUL['RUL'].to_numpy())
    print(Err)
    print(Mse)
    print(Mae)