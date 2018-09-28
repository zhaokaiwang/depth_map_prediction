import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np 
from config import get_config
import torchvision

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class _DenseBlock_last(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock_last, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        
        return torch.cat(features[1:], 1)


class _Transition_down(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate=0.2):
        super(_Transition_down, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        #self.add_module('dropout', nn.Dropout(drop_rate))
        self.add_module('max_pooling', nn.MaxPool2d(kernel_size=2, stride=2))
    

class _Transition_up(nn.Sequential):
    def __init__(self, num_input_features, padding, output_padding):
        super(_Transition_up, self).__init__()

        self.add_module('transition', 
                        nn.ConvTranspose2d(num_input_features, num_input_features, 3, stride=2,
                        padding=padding, output_padding=output_padding, bias=False))

class densenet_cat(nn.Module):
    r"""Densenet fully convolution mode, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, dataset, growth_rate=16, block_config=[4, 5, 7, 10, 12], last_layer=15,
                 num_init_features=48, bn_size=4, drop_rate=0.2, efficient=False,
                 constrain=False, source=False):
        super(densenet_cat, self).__init__()

        self.growth_rate = growth_rate
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.efficient = efficient
        self.last_layer = last_layer
        self.constrain = constrain
        self.source = source
        self.block_config = block_config

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        num_features = num_init_features
        self.features = [num_init_features]
        self.resolution = [[self.input_height, self.input_weight]]

        self.conv = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)

        #Dense block and transtion down
        for i in range(len(self.block_config)):
            self.add_module('denseBlock{}'.format(i), _DenseBlock(self.block_config[i], num_features, self.bn_size, self.growth_rate, self.drop_rate, efficient=self.efficient))
            num_features = num_features + growth_rate * self.block_config[i]
            self.features.append(num_features)
            self.add_module('transitionDown{}'.format(i), _Transition_down(num_features, num_features, self.drop_rate))
            self.resolution.append([self.resolution[-1][0] // 2, self.resolution[-1][1] // 2])

        self.add_module('last_denseBlock', _DenseBlock_last(last_layer, num_features, self.bn_size, self.growth_rate, self.drop_rate, self.efficient))
        num_features = num_features + growth_rate * last_layer
        
        self.block_config.append(last_layer)
        self.block_config = self.block_config[::-1]
        self.features = self.features[::-1]
        self.resolution = self.resolution[::-1]

        #Reverse dense block and tranistion up 
        for i in range(len(self.block_config) - 1):
            self.add_module('deDenseBlock{}'.format(i), _DenseBlock_last(self.block_config[i + 1], self.features[i] + self.block_config[i] * self.growth_rate, self.bn_size, self.growth_rate, self.drop_rate, self.efficient))
            
            in_h, in_w = self.resolution[i][0], self.resolution[i][1]
            out_h, out_w = self.resolution[i + 1][0], self.resolution[i + 1][1]

            pad, out_pad = [1, 1], [1, 1]
            if out_h % in_h == 1:
                pad[0], out_pad[0] = 0, 0

            if out_w % in_w == 1:
                pad[1], out_pad[1] = 0, 0

            pad, out_pad = tuple(pad), tuple(out_pad)
            self.add_module('transitionUp{}'.format(i), _Transition_up(self.block_config[i] * self.growth_rate, pad, out_pad))

        self.last_conv = nn.Conv2d(self.block_config[-1] * self.growth_rate, 1, 1, stride=1, bias=False)

        if self.source:
            self.upsample = nn.Upsample(size=[self.source_height, self.source_weight], mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(size=[self.input_height, self.input_weight], mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, input):
        output = self.conv(input)
        
        #Dense block and transition down 
        output = self.denseBlock0(output)
        cat0 = output
        output = self.transitionDown0(output)

        output = self.denseBlock1(output)
        cat1 = output
        output = self.transitionDown1(output)

        output = self.denseBlock2(output)
        cat2 = output
        output = self.transitionDown2(output)

        output = self.denseBlock3(output)
        cat3 = output
        output = self.transitionDown3(output)

        output = self.denseBlock4(output)
        cat4 = output
        output = self.transitionDown4(output)
        
        output = self.last_denseBlock(output)
        
        #Dense block and transition up
        output = self.transitionUp0(output)
        output = torch.cat([output, cat4], 1)
        output = self.deDenseBlock0(output)

        output = self.transitionUp1(output)
        output = torch.cat([output, cat3], 1)
        output = self.deDenseBlock1(output)

        output = self.transitionUp2(output)
        output = torch.cat([output, cat2], 1)
        output = self.deDenseBlock2(output)
        
        output = self.transitionUp3(output)
        output = torch.cat([output, cat1], 1)
        output = self.deDenseBlock3(output)
        
        output = self.transitionUp4(output)
        output = torch.cat([output, cat0], 1)
        output = self.deDenseBlock4(output) 

        output = self.last_conv(output)

        if self.source is True:
            output = self.upsample(output)

        output = torch.squeeze(output, 1)
        return output


class densenet_fcn(nn.Module):
    r"""Densenet fully convolution mode, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, dataset, growth_rate=16, block_config=[4, 5, 7, 10, 12], last_layer=15,
                 num_init_features=48, bn_size=4, drop_rate=0.2, efficient=True,
                 constrain=False, source=False):
        super(densenet_fcn, self).__init__()

        self.growth_rate = growth_rate
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.efficient = efficient
        self.last_layer = last_layer
        self.constrain = constrain
        self.source = source
        self.block_config = block_config

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        num_features = num_init_features
        self.features = [num_init_features]
        self.resolution = [[self.input_height, self.input_weight]]

        self.conv = nn.Conv2d(3, num_init_features, kernel_size=3, padding=1, bias=False)

        #Dense block and transtion down
        for i in range(len(self.block_config)):
            self.add_module('denseBlock{}'.format(i), _DenseBlock(self.block_config[i], num_features, self.bn_size, self.growth_rate, self.drop_rate, efficient=self.efficient))
            num_features = num_features + growth_rate * self.block_config[i]
            self.features.append(num_features)
            self.add_module('transitionDown{}'.format(i), _Transition_down(num_features, num_features, self.drop_rate))
            self.resolution.append([self.resolution[-1][0] // 2, self.resolution[-1][1] // 2])

        self.add_module('last_denseBlock', _DenseBlock_last(last_layer, num_features, self.bn_size, self.growth_rate, self.drop_rate, self.efficient))
        num_features = num_features + growth_rate * last_layer
        
        self.block_config.append(last_layer)
        self.block_config = self.block_config[::-1]
        self.features = self.features[::-1]
        self.resolution = self.resolution[::-1]

        #Reverse dense block and tranistion up 
        for i in range(len(self.block_config) - 1):
            self.add_module('deDenseBlock{}'.format(i), _DenseBlock_last(self.block_config[i + 1], self.features[i] + self.block_config[i] * self.growth_rate, self.bn_size, self.growth_rate, self.drop_rate, self.efficient))
            
            in_h, in_w = self.resolution[i][0], self.resolution[i][1]
            out_h, out_w = self.resolution[i + 1][0], self.resolution[i + 1][1]

            pad, out_pad = [1, 1], [1, 1]
            if out_h % in_h == 1:
                pad[0], out_pad[0] = 0, 0

            if out_w % in_w == 1:
                pad[1], out_pad[1] = 0, 0

            pad, out_pad = tuple(pad), tuple(out_pad)
            self.add_module('transitionUp{}'.format(i), _Transition_up(self.block_config[i] * self.growth_rate, pad, out_pad))
            self.add_module('predict{}'.format(i), nn.Conv2d(self.block_config[i] * self.growth_rate, 1, 3, stride=1, bias=False, padding=1))
        
        self.last_conv = nn.Conv2d(self.block_config[-1] * self.growth_rate, 1, 1, stride=1, bias=False)
        self.upsample_source = nn.Upsample(size=[self.source_height, self.source_weight], mode='bilinear', align_corners=True)
        self.upsample_input = nn.Upsample(size=[self.input_height, self.source_weight], mode='bilinear', align_corners=True)

        self.add_module('predict', nn.Conv2d(5, 1, 1, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, input):
        output = self.conv(input)
        
        #Dense block and transition down 
        output = self.denseBlock0(output)
        cat0 = output
        output = self.transitionDown0(output)

        output = self.denseBlock1(output)
        cat1 = output
        output = self.transitionDown1(output)

        output = self.denseBlock2(output)
        cat2 = output
        output = self.transitionDown2(output)

        output = self.denseBlock3(output)
        cat3 = output
        output = self.transitionDown3(output)

        output = self.denseBlock4(output)
        cat4 = output
        output = self.transitionDown4(output)
        
        output = self.last_denseBlock(output)
        
        #Dense block and transition up
        output = self.transitionUp0(output)
        predict0 = self.predict0(output)
        output = torch.cat([output, cat4], 1)
        output = self.deDenseBlock0(output)

        output = self.transitionUp1(output)
        predict1 = self.predict1(output)
        output = torch.cat([output, cat3], 1)
        output = self.deDenseBlock1(output)

        output = self.transitionUp2(output)
        predict2 = self.predict2(output)
        output = torch.cat([output, cat2], 1)
        output = self.deDenseBlock2(output)
        
        output = self.transitionUp3(output)
        predict3 = self.predict3(output)
        output = torch.cat([output, cat1], 1)
        output = self.deDenseBlock3(output)
        
        output = self.transitionUp4(output)
        predict4 = self.predict4(output)
        output = torch.cat([output, cat0], 1)
        output = self.deDenseBlock4(output) 

        output = self.last_conv(output)
        
        #temp = output

        predict0 = self.upsample_input(predict0)
        predict1 = self.upsample_input(predict1)
        predict2 = self.upsample_input(predict2)
        predict3 = self.upsample_input(predict3)
        predict4 = self.upsample_input(predict4)

        output = torch.cat([predict0, predict1, predict2, predict3, output], 1)
        output = self.predict(output)
          
        if self.source is True:
            output = self.upsample_source(output)

        #output = temp
        output = torch.squeeze(output, 1)
        return output


class densenet_fcn_final(nn.Module):
    r"""Densenet fully convolution mode, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, dataset, growth_rate=16, block_config=[4, 5, 7, 10, 12], last_layer=15,
                 num_init_features=48, bn_size=4, drop_rate=0.2, efficient=True,
                 constrain=False, source=False):
        super(densenet_fcn_final, self).__init__()

        self.growth_rate = growth_rate
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.efficient = efficient
        self.last_layer = last_layer
        self.constrain = constrain
        self.source = source
        self.block_config = block_config

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        num_features = num_init_features
        self.features = [num_init_features]
        self.resolution = [[self.input_height, self.input_weight]]

        self.conv = nn.Conv2d(3, num_init_features, kernel_size=3, padding=1, bias=False)

        #Dense block and transtion down
        for i in range(len(self.block_config)):
            self.add_module('denseBlock{}'.format(i), _DenseBlock(self.block_config[i], num_features, self.bn_size, self.growth_rate, self.drop_rate, efficient=self.efficient))
            num_features = num_features + growth_rate * self.block_config[i]
            self.features.append(num_features)
            self.add_module('transitionDown{}'.format(i), _Transition_down(num_features, num_features, self.drop_rate))
            self.resolution.append([self.resolution[-1][0] // 2, self.resolution[-1][1] // 2])

        self.add_module('last_denseBlock', _DenseBlock_last(last_layer, num_features, self.bn_size, self.growth_rate, self.drop_rate, self.efficient))
        num_features = num_features + growth_rate * last_layer
        
        self.block_config.append(last_layer)
        self.block_config = self.block_config[::-1]
        self.features = self.features[::-1]
        self.resolution = self.resolution[::-1]

        #Reverse dense block and tranistion up 
        for i in range(len(self.block_config) - 1):
            self.add_module('deDenseBlock{}'.format(i), _DenseBlock_last(self.block_config[i + 1], self.features[i] + self.block_config[i] * self.growth_rate, self.bn_size, self.growth_rate, self.drop_rate, self.efficient))
            
            in_h, in_w = self.resolution[i][0], self.resolution[i][1]
            out_h, out_w = self.resolution[i + 1][0], self.resolution[i + 1][1]

            pad, out_pad = [1, 1], [1, 1]
            if out_h % in_h == 1:
                pad[0], out_pad[0] = 0, 0

            if out_w % in_w == 1:
                pad[1], out_pad[1] = 0, 0

            pad, out_pad = tuple(pad), tuple(out_pad)
            self.add_module('transitionUp{}'.format(i), _Transition_up(self.block_config[i] * self.growth_rate, pad, out_pad))
            self.add_module('predict{}'.format(i), nn.Conv2d(self.block_config[i] * self.growth_rate + self.features[i] , 1, 3, stride=1, bias=False, padding=1))
        
        self.last_conv = nn.Conv2d(self.block_config[-1] * self.growth_rate, 1, 1, stride=1, bias=False)
        self.upsample_source = nn.Upsample(size=[self.source_height, self.source_weight], mode='bilinear', align_corners=True)
        self.upsample_input = nn.Upsample(size=[self.input_height, self.source_weight], mode='bilinear', align_corners=True)

        self.add_module('predict', nn.Conv2d(5, 1, 1, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, input):
        output = self.conv(input)
        
        #Dense block and transition down 
        output = self.denseBlock0(output)
        cat0 = output
        output = self.transitionDown0(output)

        output = self.denseBlock1(output)
        cat1 = output
        output = self.transitionDown1(output)

        output = self.denseBlock2(output)
        cat2 = output
        output = self.transitionDown2(output)

        output = self.denseBlock3(output)
        cat3 = output
        output = self.transitionDown3(output)

        output = self.denseBlock4(output)
        cat4 = output
        output = self.transitionDown4(output)
        
        output = self.last_denseBlock(output)
        
        #Dense block and transition up
        output = self.transitionUp0(output)
        output = torch.cat([output, cat4], 1)
        predict0 = self.predict0(output)
        output = self.deDenseBlock0(output)

        output = self.transitionUp1(output)
        output = torch.cat([output, cat3], 1)
        predict1 = self.predict1(output)
        output = self.deDenseBlock1(output)

        output = self.transitionUp2(output)
        output = torch.cat([output, cat2], 1)
        predict2 = self.predict2(output)
        output = self.deDenseBlock2(output)
        
        output = self.transitionUp3(output)
        output = torch.cat([output, cat1], 1)
        predict3 = self.predict3(output)
        output = self.deDenseBlock3(output)
        
        output = self.transitionUp4(output)
        output = torch.cat([output, cat0], 1)
        predict4 = self.predict4(output)
        output = self.deDenseBlock4(output) 

        output = self.last_conv(output)

        predict0 = self.upsample_input(predict0)
        predict1 = self.upsample_input(predict1)
        predict2 = self.upsample_input(predict2)
        predict3 = self.upsample_input(predict3)
        predict4 = self.upsample_input(predict4)
        output = torch.cat([predict0, predict1, predict2, predict3, predict4, output], 1)
        output = self.predict(output)
          
        if self.source is True:
            output = self.upsample_source(output)

        output = torch.squeeze(output, 1)
        return output

class densenet_fc(nn.Module):
    def __init__(self, dataset, source=False):
        super(densenet_fc, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=False)

        self.source = source
        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        self.output_height = int(get_config(dataset, 'output_height'))
        self.output_weight = int(get_config(dataset, 'output_weight'))

        self.fc = nn.Linear(49152, self.output_height * self.output_weight)

        if self.source is False:
            self.upsample = nn.Upsample((self.input_height, self.input_weight), mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample((self.source_height, self.source_weight), mode='bilinear', align_corners=True)

    def forward(self, input):
        output = self.model.features(input)
        output = output.view(output.size(0), - 1)
        output = self.fc(output)

        output = output.view(output.size(0), 1, self.output_height, self.output_weight)

        output= self.upsample(output)
        output = torch.squeeze(output, dim=1)
        return output


def main():
    device = torch.device('cuda:0')
    model = densenet_fcn_final('Make3D')
    model.to(device)
    input = np.random.rand(8, 3, 256, 192).astype(np.float32)
    input = torch.from_numpy(input)
    input = input.cuda()
    output = model(input)
    print(output.shape)
    pass

if __name__ == '__main__':
    main()