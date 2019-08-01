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

        return out 

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                num_outchannels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        assert num_branches == len(num_blocks)
        assert num_branches == len(num_inchannels)
        assert num_branches == len(num_outchannels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
    
        self.mutil_sclae_output = True
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)
    
    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []

        for i in range(num_branches if self.mutil_sclae_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], 
                        num_branches[i],
                        1,
                        1,
                        0,
                        bias=False),
                        normalization(num_inchannels[i]),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outputchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j]),
                                nn.BatchNorm2d(num_outputchannels_conv3x3),
                            ))
                        else:
                            num_outputchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outputchannels_conv3x3,
                                    3, 2, 1, bias=False),
                            normalization(num_outputchannels_conv3x3),
                            nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)
                        


class HighResolutionNet(nn.Module):
    def __init__(self):
        super(HighResolutionNet, self).__init__()

        #first stget 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = normalization(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = normalization(64)
        self.relu = nn.ReLU(inplac=True)

        self.layer1 = self.makr

    def _make_layer (self, block, input_channel, output_channel, blocks, stride):
        downsample = None
        if stride != 1 or input_channel != block.expansion * output_channel:
            downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel * blocks.expansion, kernel_size = 1, stride=stride, bias=False),
                normalization(output_channel * blocks.expansion),
            )

        layers = []
        layers.append(block(input_channel, output_channel, stride, downsample))
        input_channel = input_channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(input_channel, output_channel))
        
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_branches_pre[i],
                                num_branches_cur[i],
                                3,
                                1,
                                1,
                                bias=False),
                        normalization(num_branches_cur[i]),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    input_channels = num_channels_pre_layer[-1]
                    output_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else input_channels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                        normalization(output_channels),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)
