import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = configs.D_kernel_size
        cnn_kernel_size: int = configs.cnn_kernel_size
        l1 = configs.l1
        l2 = l1 - cnn_kernel_size + 1
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.Relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        

        self.Linear_Seasonal = nn.Linear(self.seq_len,l1)
        self.Linear_Seasonal2 = nn.Linear(l2, self.pred_len)
        self.Linear_Seasonal3 = nn.Linear(self.channels, self.pred_len)
        self.Conv1D_Seasonal = nn.Conv1d(self.channels, self.channels, kernel_size=cnn_kernel_size, stride=1, padding=0)

        self.Linear_Trend = nn.Linear(self.seq_len, l1)
        self.Linear_Trend2 = nn.Linear(l2, self.pred_len)
        self.Linear_Trend3 = nn.Linear(self.channels, self.pred_len)
        self.Conv1D_Trend = nn.Conv1d(self.channels, self.channels, kernel_size=cnn_kernel_size, stride=1, padding=0)

            # self.Linear_output = nn.Linear(self.pred_len, self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))



    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)


        seasonal_output = self.Linear_Seasonal(seasonal_init)
        seasonal_output = self.tanh(seasonal_output)
        seasonal_output = self.Conv1D_Seasonal(seasonal_output)
        seasonal_output = self.tanh(seasonal_output)
        seasonal_output = self.Linear_Seasonal2(seasonal_output)
        seasonal_output = seasonal_output.view(seasonal_output.size(0), self.channels)
        seasonal_output = self.Linear_Seasonal3(seasonal_output)

        # seasonal_output = self.Linear_Seasonal2(seasonal_output)
        trend_output = self.Linear_Trend(trend_init)
        trend_output = self.tanh(trend_output)
        trend_output = self.Conv1D_Trend(trend_output)
        trend_output = self.tanh(trend_output)
        trend_output = self.Linear_Trend2(trend_output)
        trend_output = trend_output.view(trend_output.size(0), self.channels)
        trend_output = self.Linear_Trend3(trend_output)
        x = seasonal_output + trend_output
        x = x.view(x.size(0), 1, self.pred_len)
        return x.permute(0, 2, 1) # to [Batch, Output length, Channel] 
