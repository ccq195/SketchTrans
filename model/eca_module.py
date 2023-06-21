import torch
from torch import nn
from torch.nn.parameter import Parameter


class MAM(nn.Module):
    def __init__(self, dim, r=16):
        super(MAM, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask = self.channel_attention(pooled)
        x = x * mask + self.IN(x) * (1 - mask)

        return x

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=768, k_size=3):
        super(eca_layer, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.IN = nn.InstanceNorm1d(768, track_running_stats=False)
        self.BN = nn.BatchNorm1d(768)


    def forward(self, x):
        # feature descriptor on the global spatial information

        # Two different branches of ECA module
        y = self.conv(x[:,0].unsqueeze(-1).transpose(-2,-1))

        # Multi-scale information fusion
        y = self.sigmoid(y).transpose(-2,-1)
        x = self.BN(x).transpose(-2,-1)*y +  self.IN(x.transpose(-2,-1)) * (1 - y)

        return x.transpose(-2,-1)
        
