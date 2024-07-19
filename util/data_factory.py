import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    for data in input_data:
        L = len(data)
        for i in range(L - tw):
            train_seq = data[i : i + tw]
            if (i + tw + pre_len) > len(data):
                break
            if config.features == "MS" or config.features == "S":
                train_label = data[:, -1:][i + tw : i + tw + pre_len]
            else:
                train_label = data[i + tw : i + tw + pre_len]
            inout_seq.append(
                (train_seq[:, :-1], train_label)
            )  # train_seq[:, :-1] 除去最后一列的数据,即去除标签列

    return inout_seq


def create_test_seq(test_data, train_window):
    inout_seq = []
    for data in test_data:
        if data.shape[0] < train_window:
            num_to_pad = train_window - data.shape[0]
            to_pad = data[0]
            to_pad = to_pad.repeat(num_to_pad, 1)
            data = torch.cat([to_pad, data], dim=0)


        inout_seq.append((data[-train_window:, :-1], data[-1, -1]))

    return inout_seq


def get_data(path, target):
    df = pd.read_csv(
        path
    )  # 填你自己的数据地址,自动选取你最后一列数据为特征列 # 添加你想要预测的特征列
    # 观测窗口

    # 将特征列移到末尾
    target_data = df[[target]]
    df = df.drop(target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    grouped = df.groupby("UnitNumber")

    true_data = []

    for name, group in grouped:
        group.drop("UnitNumber", axis=1, inplace=True)

        # cols_data = group.columns[1:]
        cols_data = group.columns
        df_data = group[cols_data]
        true_data.append(df_data.values)

    return true_data


def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    pre_len = config.pred_len  # 预测未来数据的长度
    train_window = config.seq_len

    true_data = get_data(config.train_path, config.target)
    test_data = get_data(config.test_path, config.target)

    # 定义标准化优化器

    train_data = true_data[int(0 * len(true_data)) :]
    valid_data = true_data[: int(0 * len(true_data))]
    print(
        "训练集尺寸:",
        len(train_data),
        "测试集尺寸:",
        len(test_data),
        "验证集尺寸:",
        len(valid_data),
    )

    # 转化为深度学习模型需要的类型Tensor
    train_data_ = [torch.FloatTensor(data).to(device) for data in train_data]
    test_data_ = [torch.FloatTensor(data).to(device) for data in test_data]
    valid_data_ = [torch.FloatTensor(data).to(device) for data in valid_data]

    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_, train_window, pre_len, config)
    test_inout_seq = create_test_seq(test_data_, train_window)
    valid_inout_seq = create_inout_sequences(valid_data_, train_window, pre_len, config)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True
    )

    print(
        "通过滑动窗口共有训练集数据：",
        len(train_inout_seq),
        "转化为批次数据:",
        len(train_loader),
    )
    print(
        "通过滑动窗口共有测试集数据：",
        len(test_inout_seq),
        "转化为批次数据:",
        len(test_loader),
    )
    print(
        "通过滑动窗口共有验证集数据：",
        len(valid_inout_seq),
        "转化为批次数据:",
        len(valid_loader),
    )
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader, valid_loader
