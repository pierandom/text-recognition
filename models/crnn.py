import os
import json
import torch
from torch import nn


class Conv2dNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups),
            nn.GELU(),
            nn.BatchNorm2d(out_ch))
        
    def forward(self, x):
        return self.layer(x)


class SeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            Conv2dNorm(in_ch, in_ch, kernel_size, stride, padding=kernel_size//2, groups=in_ch),
            Conv2dNorm(in_ch, out_ch, kernel_size=1))
        
    def forward(self, x):
        return self.layer(x)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.residual = SeparableConv(channels, channels, kernel_size)
    
    def forward(self, x):
        return x + self.residual(x)


class Stage(nn.Module):
    def __init__(self, channels, kernel_size, repeat):
        super().__init__()
        self.stage = nn.Sequential(*[
            ResBlock(channels, kernel_size) for _ in range(repeat)])
    
    def forward(self, x):
        return self.stage(x)


class CNN(nn.Module):
    def __init__(self, input_channels, kernel_size, net_width_factor,
                 n_stages, stage_repeat):
        super().__init__()
        

        channels = [16 * net_width_factor * 2**stage_idx for stage_idx in range(n_stages)]
        self.out_features = channels[-1]
        self.reduction_factor = 2**(n_stages-1)

        self.cnn = nn.Sequential()
        self.cnn.add_module('input_conv', Conv2dNorm(input_channels, channels[0], kernel_size=1))
        self.cnn.add_module('stage_0', Stage(channels[0], kernel_size, stage_repeat))
        for stage_idx in range(1, n_stages):
            self.cnn.add_module(f'pooling_conv_{stage_idx-1}',
                SeparableConv(channels[stage_idx-1], channels[stage_idx], kernel_size, stride=2))
            self.cnn.add_module(f'stage_{stage_idx}',
                Stage(channels[stage_idx], kernel_size, stage_repeat))


    def forward(self, x):
        return self.cnn(x)   



class CRNN(nn.Module):
    def __init__(self, input_shape, num_classes, kernel_size, cnn_width_factor,
                 n_stages, stage_repeat, rnn_hidden_size, rnn_num_layers):
        super().__init__()
        in_channels, in_height, in_width = input_shape
        self.cnn = CNN(in_channels, kernel_size, cnn_width_factor, n_stages, stage_repeat)
        self.reduction_factor = self.cnn.reduction_factor
        rnn_input_size = (in_height // self.cnn.reduction_factor) * self.cnn.out_features
        self.rnn = nn.LSTM(rnn_input_size, rnn_hidden_size, num_layers=rnn_num_layers, bidirectional=True)
        self.head = nn.Linear(2*rnn_hidden_size, num_classes) # 2*rnn_hidden_size because rnn is bidirectional

    def forward(self, x):
        vision_features = self.cnn(x)
        b, c, h, w = vision_features.size()
        vision_features = vision_features.view(b, c*h, w)
        vision_features = vision_features.permute(2, 0, 1)
        sequence_features, _ = self.rnn(vision_features)
        logits = self.head(sequence_features)
        return logits

    def weight_decay(self, device):
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        n = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                loss += m.weight.norm(2)
                n += 1
        return loss / n
    