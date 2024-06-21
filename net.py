
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layers):
        """
        初始化自定义模型。
        """
        super(Actor, self).__init__()
        # 定义隐藏层序列
        layers = []
        for i in range(hidden_layers):
            if i == 0:
                # 第一层之后使用hidden_dim作为中间层的维度
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            # 使用ReLU作为隐藏层的激活函数
            layers.append(nn.ReLU())
        
        # 最后一层线性变换到输出维度，并使用Softmax作为激活函数
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.LogSoftmax(dim=-1))
        
        self.hidden = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.hidden(x)
        dist = Categorical(out)
        action = dist.sample()
        return dist,action 

class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super(Critic, self).__init__()
        def he_init(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.hidden = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(256, 384),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
        )
        for layer in self.hidden:
            he_init(layer)
        self.out = nn.Linear(192, 1)
        he_init(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # x = torch.cat((state, action), dim=-1)
        x = self.hidden(state)
        value = self.out(x)
        value = value.sum(dim=0)
        value = value.reshape(x.size(1),)
        return value











