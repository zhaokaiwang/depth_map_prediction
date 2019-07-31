import torch
import torch.nn as nn
import torch.functional as F


normalization = nn.GroupNorm
def conv3x3(input_channel, output_channel, strides=1):
    return nn.Conv2d(input_channel, output_channel, 3, strides, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, input_channel, output_channel, strides, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(input_channel, output_channel, strides)
        self.norm1 = normalization(output_channel)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_channel, output_channel)
        self.norm2 = normalization(output_channel)
        self.downsample = downsample
        self.strides = strides
    
    def forward(self, input):
        residual = input 

        output = self.conv1(input)
        output = self.norm1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.norm2(output)

        if self.downsample is not None:
            residual = self.downsample(input)
        
        output += residual
        output = self.relu(output)
        
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, strides, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.norm1 = normalization(output_channel)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(output_channel, output_channel)
        self.norm2 = normalization(output_channel)

        self.conv3 = nn.Conv2d(output_channel, self.expansion * output_channel, 1, 1, bias=False)
        self.norm3 = normalization(output_channel * self.expansion)

        self.downsample = downsample
        self.strides = strides
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class IdenityMapping(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, strides, downsample=None):
        super(IdenityMapping, self).__init__()

        self.norm1 = normalization(input_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, output_channel, 1, 1, bias=False)
        
        self.norm2 = normalization(input_channel)
        self.conv2 = conv3x3(output_channel, output_channel)

        self.norm3 = normalization(input_channel)
        self.conv3 = nn.Conv2d(output_channel, self.expansion * output_channel, 1, 1, bias=False)
        
        self.downsample = downsample
        self.strides = strides

    def forward(self, x):
        residual = x 

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual 

        return output 