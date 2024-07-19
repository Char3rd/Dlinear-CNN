import numpy as np
import pandas as pd

RUL = pd.read_csv("dataset/RUL_FD001.csv", names=["RUL"])
df_test = pd.read_csv("dataset/test_FD001 copy.csv")

pre = pd.read_csv("results/RUL-ForecastResults.csv")

b =pd.DataFrame(df_test.groupby("单元序号")["时间"].max()).reset_index()
b.columns = ["UnitNumber", "max"]

all_life  = RUL['RUL'] + b['max']

# print(all_life)

def err(all_life, pre, real): 
    # print(all_life.shape, pre.shape, real.shape)
    err = np.mean(np.abs(real - pre) / all_life)
    # print(np.abs(real - pre))
    return err

def MSE(pred, true):
    print(pred.shape, true.shape)
    return np.mean((pred - true) ** 2)

def MAE(pred, true):
    print(pred.shape, true.shape)
    return np.mean(np.abs(pred - true))

def R2(pred, true):
    return 1 - np.mean((pred - true) ** 2) / np.var(true)

def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    r_squared = R2(pred, true)

    return mse, mae, r_squared
