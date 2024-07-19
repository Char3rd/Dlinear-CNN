"""
erditor: Snu77
"""


import os
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# params
from layers import DLinear
from util import metrics
from util.data_factory import create_dataloader
from util.tools import adjust_learning_rate, add_attributes



class SCINetinitialization:
    def __init__(self, args):
        super(SCINetinitialization, self).__init__()
        self.args = args
        self.model, self.device = self.build_model(args)

        self.train_dataloader, self.test_dataloader, _ = create_dataloader(
            self.args, "cuda"
        )

    def build_model(self, args):
        model = DLinear.Model(self.args).float()
        # 将模型定义在GPU上

        if args.use_gpu:
            device = torch.device("cuda:{}".format(self.args.device))
            print("Use GPU: cuda:{}".format(args.device))
        else:
            print("Use CPU")
            device = torch.device("cpu")
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameters: %.2fM" % (total / 1e6))  # 打印模型参数量

        if self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=[device])

        return model, device

    def train(self, setting, is_save_model: bool = True):
        train_loader = self.train_dataloader

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.loss == "mse":
            criterion = nn.MSELoss()
        if self.args.loss == "mae":
            criterion = nn.L1Loss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss
                )
            )
            # if epoch % 5 == 0:
            #     self.predict(None, False)


            adjust_learning_rate(model_optim, epoch + 1, self.args)

        if is_save_model:
            self.save(path + "/model.pth")

        return self.model

    def predict(self, model_path, load=True):
        # predict
        preds = []
        RUL = []

        # 加载模型
        if load:
            self.model.load_state_dict(torch.load(model_path))

        # 评估模式
        self.model.eval()

        pred_loader = self.test_dataloader

        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                if self.args.features == "MS":
                    for i in range(self.args.pred_len):
                        preds.append(
                            outputs[0][i][outputs.shape[2] - 1].cpu()
                        )  # 取最后一个预测值即对应target列
                        RUL.append(batch_y.cpu()[0])

                else:
                    for i in range(self.args.pred_len):
                        preds.append(outputs[0][i][outputs.shape[2] - 1].cpu())
                # print(outputs)
        preds = np.array(preds)
        RUL = np.array(RUL)
        mae = metrics.MAE(preds, RUL)
        mse = metrics.MSE(preds, RUL)
        err = metrics.err(metrics.all_life, preds, RUL)
        print(f"mae: {mae}\tmse: {mse}\terr: {err}")

        # 保存结果
        if self.args.show_results:
            df = pd.DataFrame({"forecast": preds})
            df.to_csv(
                "./results/{}-ForecastResults.csv".format(self.args.target), index=False
            )

            if self.args.show_results:
                plt.cla()

                # 设置绘图风格
                plt.style.use("ggplot")

                # 创建折线图
                plt.plot(RUL.tolist(), label="real", color="blue")  # 实际值
                plt.plot(
                    preds.tolist(), label="forecast", color="red", linestyle="--"
                )  # 预测值

                # 增强视觉效果
                plt.grid(True)
                plt.title("real vs forecast")
                plt.xlabel("time")
                plt.ylabel("value")
                plt.legend()

                plt.savefig("results.png")
        return mae, mse, err

    def save(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)


if __name__ == "__main__":

    with open("./config.json", 'r', encoding='utf-8') as file:
        config = json.load(file)


    @add_attributes(config)
    class Config:
        train: bool = False
        show_results: bool = False
        train_path: str = "dataset/train.csv"
        test_path: str = "dataset/test.csv"
        features: str = "MS"
        target: str = "RUL"
        checkpoints: str = "models/"
        pred_len: int = 1
        individual: bool = False
        enc_in: int = 12
        lradj: int = "6"
        use_gpu : bool = True
        device: int = 0
        train_epochs: int = 27

    SCI = SCINetinitialization(Config)  # 实例化模型
    SCI.train("model")
    SCI.predict("", False)

    plt.show()
