import torch 
import torch.nn as nn 
import torchvision
from config import get_config
import numpy as np 
import math

class resnet_fc(nn.Module):
    def __init__(self, dataset, source=False):
        super(resnet_fc, self).__init__()
        self.source = source

        self.model = torchvision.models.resnet50(pretrained=False)

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        self.output_height = int(get_config(dataset, 'output_height'))
        self.output_weight = int(get_config(dataset, 'output_weight'))

        if dataset == 'NyuV2':
            self.model.fc = nn.Linear(114688, self.output_height * self.output_weight)
        else:
            self.model.fc = nn.Linear(98304, self.output_height * self.output_weight)
        if self.source is False:
            self.upsample = nn.Upsample((self.input_height, self.input_weight), mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample((self.source_height, self.source_weight), mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.model.conv1(input)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = output.view(output.size(0), -1)
        output = self.model.fc(output)
              
        output = output.view(output.size(0), 1, self.output_height, self.output_weight)

        output= self.upsample(output)
        output = torch.squeeze(output, dim=1)
        return output

class _basic_block(nn.Sequential):
    def __init__(self, input_feature):
        super(_basic_block, self).__init__()

        self.add_module('bn', nn.BatchNorm2d(input_feature))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(input_feature, input_feature, kernel_size=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class _conv_bn_relu(nn.Sequential):
    def __init__(self, input_feature):
        super(_conv_bn_relu, self).__init__()

        self.add_module("conv", nn.Conv2d(input_feature, input_feature // 4, kernel_size=1, stride=1, bias=False))
        self.add_module("bn", nn.BatchNorm2d(input_feature // 4))
        self.add_module("relu", nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class resnet_deconv_cat(nn.Module):
    def __init__(self, dataset, source=False):
        super(resnet_deconv_cat, self).__init__()
        self.source = source

        self.model = torchvision.models.resnet50(pretrained=True)

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        self.output_height = int(get_config(dataset, 'output_height'))
        self.output_weight = int(get_config(dataset, 'output_weight'))

        self.deconv1 = nn.ConvTranspose2d(4096, 1024, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(2048, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)

        for i in range(1, 5):
            self.add_module('conv_{}'.format(i), _basic_block(int(math.pow(2, i - 1)) * 256))

        self.add_module('last_conv', nn.Conv2d(128, 1, kernel_size=1, padding=False, bias=False))
        if self.source is False:
            self.upsample = nn.Upsample((self.input_height, self.input_weight), mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample((self.source_height, self.source_weight), mode='bilinear', align_corners=True)

        nn.init.kaiming_normal_(self.last_conv.weight)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        output = self.model.conv1(input)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        cat1 = output
        output = self.model.layer2(output)
        cat2 = output
        output = self.model.layer3(output)
        cat3 = output
        output = self.model.layer4(output)
        cat4 = output

        output = self.conv_4(output)
        output = torch.cat([output, cat4], 1)
        output = self.deconv1(output)

        output = self.conv_3(output)
        output = torch.cat([output, cat3], 1)
        output = self.deconv2(output)

        output = self.conv_2(output)
        output = torch.cat([output, cat2], 1)
        output = self.deconv3(output)

        output = self.conv_1(output)
        output = torch.cat([output, cat1], 1)
        output = self.deconv4(output)

        output = self.last_conv(output)
        output= self.upsample(output)
        output = torch.squeeze(output, dim=1)
        return output

class resnet_upsample(nn.Module):
    def __init__(self, dataset, source=False):
        super(resnet_upsample, self).__init__()
        self.source = source

        self.model = torchvision.models.resnet50(pretrained=True)

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        self.output_height = int(get_config(dataset, 'output_height'))
        self.output_weight = int(get_config(dataset, 'output_weight'))

        #default is the nearst upsample
        self.upsample1 = nn.Upsample((self.input_height // 16, self.input_weight // 16), mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample((self.input_height // 8, self.input_weight // 8), mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample((self.input_height // 4, self.input_weight // 4), mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample((self.input_height // 2, self.input_weight // 2), mode='bilinear', align_corners=True)

        for i in range(1, 5):
            self.add_module('conv_{}'.format(i), _conv_bn_relu(int(2 * int(math.pow(2, i - 1)) * 256)))
        
        self.add_module("first_conv", nn.Conv2d(2048, 2048, kernel_size=1, padding=False, bias=False))
        self.add_module('last_conv', nn.Conv2d(128, 1, kernel_size=1, padding=False, bias=False))

        if self.source is False:
            self.upsample = nn.Upsample((self.input_height, self.input_weight), mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample((self.source_height, self.source_weight), mode='bilinear', align_corners=True)

        nn.init.kaiming_normal_(self.last_conv.weight)

    def forward(self, input):
        output = self.model.conv1(input)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        cat1 = output
        output = self.model.layer2(output)
        cat2 = output
        output = self.model.layer3(output)
        cat3 = output
        output = self.model.layer4(output)
        cat4 = output

        #middle layer
        output = self.first_conv(output)

        output = torch.cat([output, cat4], 1)
        output = self.upsample1(output)
        output = self.conv_4(output)

        output = torch.cat([output, cat3], 1)
        output = self.upsample2(output)
        output = self.conv_3(output)

        output = torch.cat([output, cat2], 1)
        output = self.upsample3(output)
        output = self.conv_2(output)

        output = torch.cat([output, cat1], 1)
        output = self.upsample4(output)
        output = self.conv_1(output)

        output = self.last_conv(output)
        output = self.upsample(output)

        output = torch.squeeze(output, dim=1)
        return output

class resnet_final(nn.Module):
    def __init__(self, dataset, source=False):
        super(resnet_final, self).__init__()
        self.source = source

        self.model = torchvision.models.resnet50(pretrained=True)

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        self.output_height = int(get_config(dataset, 'output_height'))
        self.output_weight = int(get_config(dataset, 'output_weight'))

        self.deconv1 = nn.ConvTranspose2d(4096, 1024, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(2048, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)

        for i in range(1, 5):
            self.add_module('conv_{}'.format(i), _basic_block(int(math.pow(2, i - 1)) * 256))

        self.add_module('last_conv', nn.Conv2d(128, 1, kernel_size=1, padding=False, bias=False))
        if self.source is False:
            self.upsample = nn.Upsample((self.input_height, self.input_weight), mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample((self.source_height, self.source_weight), mode='bilinear', align_corners=True)

        nn.init.kaiming_normal_(self.last_conv.weight)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        output = self.model.conv1(input)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        cat1 = output
        output = self.model.layer2(output)
        cat2 = output
        output = self.model.layer3(output)
        cat3 = output
        output = self.model.layer4(output)
        cat4 = output

        output = self.conv_4(output)
        output = torch.cat([output, cat4], 1)
        output = self.deconv1(output)

        output = self.conv_3(output)
        output = torch.cat([output, cat3], 1)
        output = self.deconv2(output)

        output = self.conv_2(output)
        output = torch.cat([output, cat2], 1)
        output = self.deconv3(output)

        output = self.conv_1(output)
        output = torch.cat([output, cat1], 1)
        output = self.deconv4(output)

        output = self.last_conv(output)
        output= self.upsample(output)
        output = torch.squeeze(output, dim=1)
        return output

class resnet_deconv(nn.Module):
    def __init__(self, dataset, source=False):
        super(resnet_deconv, self).__init__()
        self.source = source

        self.model = torchvision.models.resnet50(pretrained=True)

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        self.output_height = int(get_config(dataset, 'output_height'))
        self.output_weight = int(get_config(dataset, 'output_weight'))

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 3, stride=2)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 3, stride=2)

        self.conv = nn.Conv2d(128, 1, kernel_size=1, padding=False, stride=1)
        if self.source is False:
            self.upsample = nn.Upsample((self.input_height, self.input_weight), mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample((self.source_height, self.source_weight), mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.model.conv1(input)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.deconv1(output)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.conv(output)

        output= self.upsample(output)
        output = torch.squeeze(output, dim=1)
        return output


class resnet_deconv_sum(nn.Module):
    def __init__(self, dataset, source=False):
        super(resnet_deconv_sum, self).__init__()
        self.source = source

        self.model = torchvision.models.resnet50(pretrained=True)

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        self.output_height = int(get_config(dataset, 'output_height'))
        self.output_weight = int(get_config(dataset, 'output_weight'))

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)

        for i in range(1, 5):
            self.add_module('conv_{}'.format(i), _basic_block(int(math.pow(2, i - 1)) * 256))

        self.add_module('last_conv', nn.Conv2d(128, 1, kernel_size=1, padding=False, bias=False))
        if self.source is False:
            self.upsample = nn.Upsample((self.input_height, self.input_weight), mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample((self.source_height, self.source_weight), mode='bilinear', align_corners=True)

        nn.init.kaiming_normal_(self.last_conv.weight)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        output = self.model.conv1(input)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        cat1 = output
        output = self.model.layer2(output)
        cat2 = output
        output = self.model.layer3(output)
        cat3 = output
        output = self.model.layer4(output)
        cat4 = output

        output = self.conv_4(output)
        output = output + cat4
        output = self.deconv1(output)

        output = self.conv_3(output)
        output = output + cat3
        output = self.deconv2(output)

        output = self.conv_2(output)
        output = output + cat2
        output = self.deconv3(output)

        output = self.conv_1(output)
        output = output + cat1
        output = self.deconv4(output)

        output = self.last_conv(output)
        output= self.upsample(output)
        output = torch.squeeze(output, dim=1)
        return output

class unpool_as_conv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super (unpool_as_conv, self).__init__()
        self.conv3x3 = nn.Conv2d(input_channel, output_channel, 3, stride=1, padding=1, bias=False)
        self.conv2x3 = nn.Conv2d(input_channel, output_channel, 3, stride=1, padding=1,bias=False)
        self.conv3x2 = nn.Conv2d(input_channel, output_channel, 3, stride=1, padding=1, bias=False)
        self.conv2x2 = nn.Conv2d(input_channel, output_channel, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)

        
    def forward(self, input):
        conv3x3 = self.conv3x3(input)
        conv2x2 = self.conv2x2(input)
        conv3x2 = self.conv3x2(input)
        conv2x3 = self.conv2x3(input)

        left = torch.cat([conv3x3, conv3x2], 2)
        right = torch.cat([conv2x3, conv2x2], 2)

        final = torch.cat((left, right), 3)
        output = self.bn(final)
        return output

class up_project(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(up_project, self).__init__()
        self.branch1 = unpool_as_conv(input_channel, output_channel)
        self.conv = nn.Conv2d(output_channel, output_channel, 3, paddind=1, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.branch2 = unpool_as_conv(input_channel, output_channel)
        self.relu = nn.ReLU(True)
    
    def forward(self, input):
        branch1 = self.branch1(input)
        branch1 = self.relu(branch1)
        branch1 = self.bn(self.conv(branch1))
        branch2 = self.branch2(input)
        output =  self.relu(branch1 + branch2)
        return output
    
class resnet_up_projection(nn.Module):
    def __init__(self, dataset, source=False):
        super(resnet_up_projection, self).__init__()

        self.source = source

        self.model = torchvision.models.resnet50(pretrained=True)

        #Read the config file and the input and source resolution
        self.input_height = int(get_config(dataset, 'input_height'))
        self.input_weight = int(get_config(dataset, 'input_weight'))

        self.source_height = int(get_config(dataset, 'source_height'))
        self.source_weight = int(get_config(dataset, 'source_weight'))

        self.output_height = int(get_config(dataset, 'output_height'))
        self.output_weight = int(get_config(dataset, 'output_weight'))
        
        input_channel = 2048
        self.conv = nn.Conv2d(input_channel, input_channel // 2 , 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(input_channel)

        input_channel =  1024
        self.final = nn.Conv2d(64, 1, 3, padding=1, bias=False)
        for i in range(4):
            self.add_module("up{}".format(i), up_project(input_channel, input_channel // 2))
            input_channel = input_channel // 2

        if self.source is False:
            self.upsample = nn.Upsample((self.input_height, self.input_weight), mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample((self.source_height, self.source_weight), mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        output = self.model.conv1(input)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.bn(self.conv(output))
        output = self.up0(output)
        output = self.up1(output)
        output = self.up2(output)
        output = self.up3(output)
        output = self.final(output)
        output = self.upsample(output)
        output = torch.squeeze(output)
        return output



def main():
    model = resnet_up_projection('Make3D')
    input = np.random.randn(4, 3, 256 , 192).astype(np.float32)
    input = torch.from_numpy(input)
    output = model(input)
    print (output.shape)

    pass

if __name__ == '__main__':
    main()