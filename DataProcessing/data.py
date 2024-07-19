import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from DataProcessing import Pca
except:
    import Pca
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pywt

dependent_var = ["RUL"]  # 依赖的变量
index_columns_names = ["UnitNumber", "Cycle"]
operational_settings_columns_names = [
    "OpSet" + str(i) for i in range(1, 4)
]  # 添加操作条件1,2,3,
sensor_measure_columns_names = [
    "Sensor" + str(i) for i in range(1, 22)
]  # 添加传感器编号1，…，21
# 输入发动机编号/运行Cycle/设置/哪个传感器数据/
input_file_column_names = (
    index_columns_names
    + operational_settings_columns_names
    + sensor_measure_columns_names
)

is_PCA = True


def fun(x, max_rul:int = 125):
    if x >= max_rul:
        x = max_rul

    return x

def small_wave(input):
    coeffs = pywt.wavedec(input, 'db4', level=4)
    threshold = np.median(np.abs(coeffs[-1]))  /0.5
    coeffs = [pywt.threshold(coeff, threshold, mode='soft') for coeff in coeffs]
    reconstructed_signal = pywt.waverec(coeffs, 'db4')
    return reconstructed_signal

def get_train_data(path="dataset/train_FD001.csv", max_rul:int = 125):

    dependent_var = ["RUL"]  # 依赖的变量

    df_train = pd.read_csv(path, names=input_file_column_names)

    rul = pd.DataFrame(df_train.groupby("UnitNumber")["Cycle"].max()).reset_index()
    rul.columns = ["UnitNumber", "max"]
    # 将每一UnitNumber中最大的Cycle找到，并在原来的df_train中添加新的colum，位置为UnitNumber的左边，也即在最右端
    df_train = df_train.merge(rul, on=["UnitNumber"], how="left")
    # 计算出RUL对应的各个值
    df_train["RUL"] = df_train["max"] - df_train["Cycle"]
    # 然后在把最大的max去掉
    df_train.drop("max", axis=1, inplace=True)

    df_train["RUL"] = df_train["RUL"].apply(lambda x: fun(x, max_rul))

    return df_train


def get_test_data(path="dataset/test_FD001.csv"):

    dependent_var = ["RUL"]  # 依赖的变量

    df_test = pd.read_csv(path, names=input_file_column_names)

    rul = pd.DataFrame(df_test.groupby("UnitNumber")["Cycle"].max()).reset_index()
    rul.columns = ["UnitNumber", "max"]

    df_test = df_test.merge(rul, on=["UnitNumber"], how="left")
    # 计算出RUL对应的各个值
    df_test["RUL"] = df_test["max"] - df_test["Cycle"]
    df_test.drop("max", axis=1, inplace=True)

    y_true = pd.read_csv("dataset/RUL_FD001.csv", names=["RUL"])
    y_true["UnitNumber"] = y_true.index + 1

    actual_rul = pd.DataFrame(y_true.groupby("UnitNumber")["RUL"].max()).reset_index()
    actual_rul.columns = ["UnitNumber", "acrul"]
    df_test = df_test.merge(actual_rul, on=["UnitNumber"], how="left")
    df_test["RUL"] = df_test["RUL"] + df_test["acrul"]
    df_test.drop("acrul", axis=1, inplace=True)

    # df_test["RUL"] = df_test["RUL"].apply(lambda x: fun(x))

    return df_test


def show_info(df_train):
    plt.cla()

    temp_df = df_train[["UnitNumber", "Cycle"]].groupby("UnitNumber").max()
    sns.violinplot(temp_df.Cycle)
    plt.title("Life of Engines")
    plt.xticks(fontsize=12, fontweight="bold")

    fig, ax = plt.subplots(1, 3, figsize=(30, 8), sharex="all")
    for i in range(0, 3):
        df_u1 = df_train.query("UnitNumber==8")
        ax[i].plot(df_u1.Cycle.values, df_u1["OpSet" + str(i + 1)])
        ax[i].set_title("OpSet" + str(i + 1))
        ax[i].set_xlabel("Cycle")

    fig, ax = plt.subplots(7, 3, figsize=(30, 20), sharex=True)
    df_u1 = df_train.query("UnitNumber==50")
    c = 0
    for i in range(0, 7):
        for j in range(0, 3):
            ax[i, j].plot(df_u1.Cycle.values, df_u1["Sensor" + str(c + 1)])
            ax[i, j].set_title("Sensor" + str(c + 1), fontsize=20)
            ax[i, j].axvline(0, c="r")
            c += 1
    plt.suptitle("Sensor Traces: Unit 50", fontsize=25)
    plt.show()

def show_info2(df_train):
    plt.cla()
    temp_df = df_train[["UnitNumber", "Cycle"]].groupby("UnitNumber").max()
    sns.violinplot(temp_df.Cycle)
    plt.title("Life of Engines")
    plt.xticks(fontsize=12, fontweight="bold")
    df_u1 = df_train.query("UnitNumber==50")
    c = 0
    fig, ax = plt.subplots(4, 3, figsize=(30, 20), sharex=True)
    for i in range(0, 4):
        for j in range(0, 3):
            ax[i, j].plot(df_u1.Cycle.values, df_u1[c])
            ax[i, j].set_title(str(c + 1), fontsize=20)
            ax[i, j].axvline(0, c="r")
            c += 1
    plt.suptitle("Sensor Traces: Unit 50", fontsize=25)
    plt.show()



def corr(df_train, feats):
    corr = df_train[feats + ["RUL"]].corr()
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    pairPlot = sns.PairGrid(
        data=df_train[df_train.UnitNumber < 10],
        x_vars="RUL",
        y_vars=feats,
        hue="UnitNumber",
        aspect=2,
    )
    pairPlot = pairPlot.map(plt.scatter, alpha=0.5)
    pairPlot = pairPlot.set(xlim=(50, 0))
    pairPlot = pairPlot.add_legend()
    plt.plot(df_train.Sensor9, df_train.Sensor14)
    plt.show()


def gen_data(n: int = 12, threshold: float = 1.5, max_rul: int = 125):
    df_train = get_train_data(max_rul=max_rul)
    df_test = get_test_data()

    not_required_feats = [
        "Sensor1",
        "Sensor5",
        "Sensor6",
        "Sensor10",
        "Sensor16",
        "Sensor18",
        "Sensor19",
        "Sensor14",
    ]
    feats = [
        feat for feat in sensor_measure_columns_names if feat not in not_required_feats
    ] + ["Cycle"]

    
    pca, scaler = Pca.pca(
        data=np.array(df_train[feats]), n_components=n, is_show_info=False
    )
    train_data = pd.DataFrame(pca.transform(scaler.transform(df_train[feats])))
    test_data = pd.DataFrame(pca.transform(scaler.transform(df_test[feats])))

    df_train = pd.concat([df_train, train_data], axis=1)
    df_test = pd.concat([df_test, test_data], axis=1)
    if n is not None:
        feats = [i for i in range(n)]

    for i in feats:

        data_temp = pd.DataFrame()

        data_temp['moving_avg_train'] = df_train[i].rolling(window=5, min_periods=1).mean()
        data_temp['moving_avg_test'] = df_test[i].rolling(window=5, min_periods=1).mean()

        data_temp['z_score_train'] = (df_train[i] - df_train[i].mean()) / df_train[i].std()
        data_temp['z_score_test'] = (df_test[i] - df_test[i].mean()) / df_test[i].std()

        df_train.loc[data_temp['z_score_train'].abs() > threshold, i] = data_temp['moving_avg_train']
        df_test.loc[data_temp['z_score_test'].abs() > threshold, i] = data_temp['moving_avg_test']



    df_train[['UnitNumber']+ feats + dependent_var].to_csv('dataset/train.csv', index=False)
    df_test[['UnitNumber'] + feats +  dependent_var].to_csv('dataset/test.csv', index=False)

if __name__ == "__main__":
    df_train = get_train_data()
    df_test = get_test_data()

    # show_info(df_train)

    not_required_feats = [
        "Sensor1",
        "Sensor5",
        "Sensor6",
        "Sensor10",
        "Sensor16",
        "Sensor18",
        "Sensor19",
    ]
    feats = [
        feat for feat in sensor_measure_columns_names if feat not in not_required_feats
    ] + ["Cycle"]

    corr(df_train, feats)
    not_required_feats = [
        "Sensor1",
        "Sensor5",
        "Sensor6",
        "Sensor10",
        "Sensor16",
        "Sensor18",
        "Sensor19",
        "Sensor14",
    ]
    # feats = [
    #     feat for feat in sensor_measure_columns_names if feat not in not_required_feats
    # ]+ ["Cycle"]
    # df_train[feats] = df_train[feats].apply(lambda x: small_wave(x))
    # df_test[feats] = df_test[feats].apply(lambda x: small_wave(x))
    # show_info(df_train)
    # show_info(df_test)
    # feats = [
    #     feat for feat in sensor_measure_columns_names if feat not in not_required_feats
    # ]+ ["Cycle"]

    
    feats = [
        feat for feat in sensor_measure_columns_names if feat not in not_required_feats
    ]+ ["Cycle"]

    if is_PCA:

        n = 12
        # scaler = StandardScaler()
        pca, scaler = Pca.pca(
            data=np.array(df_train[feats]), n_components=n, is_show_info=True
        )
        train_data = pd.DataFrame(pca.transform(scaler.transform(df_train[feats])))
        test_data = pd.DataFrame(pca.transform(scaler.transform(df_test[feats])))

        df_train = pd.concat([df_train, train_data], axis=1)
        df_test = pd.concat([df_test, test_data], axis=1)
        if n is not None:
            feats = [i for i in range(n)]
    else:
        scaler = StandardScaler()
        df_train[feats] = scaler.fit_transform(df_train[feats])
        df_test[feats] = scaler.transform(df_test[feats])
        pass

    # show_info2(df_train)
    # show_info2(df_test)

    for i in feats:

        data_temp = pd.DataFrame()

        data_temp['moving_avg_train'] = df_train[i].rolling(window=5, min_periods=1).mean()
        data_temp['moving_avg_test'] = df_test[i].rolling(window=5, min_periods=1).mean()

        data_temp['z_score_train'] = (df_train[i] - df_train[i].mean()) / df_train[i].std()
        data_temp['z_score_test'] = (df_test[i] - df_test[i].mean()) / df_test[i].std()

        df_train.loc[data_temp['z_score_train'].abs() > 1.5, i] = data_temp['moving_avg_train']
        df_test.loc[data_temp['z_score_test'].abs() > 1.5, i] = data_temp['moving_avg_test']


    # show_info2(df_train)
    # show_info2(df_test)

    # df_train[feats] = df_train[feats].rolling(window=7, center=True, min_periods=0).mean()
    # df_test[feats] = df_test[feats].rolling(window=3, center=True, min_periods=0).mean()


    df_train[['UnitNumber']+ feats + dependent_var].to_csv('dataset/train.csv', index=False)
    df_test[['UnitNumber'] + feats +  dependent_var].to_csv('dataset/test.csv', index=False)
