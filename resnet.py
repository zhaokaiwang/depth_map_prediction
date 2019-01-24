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

        self.model = torchvision.models.resnet50(pretrained=True)

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

def main():
    device = torch.device('cuda:0')
    model = resnet_deconv_sum('NyuV2')

    model.to(device)
    input = np.random.randn(4, 3, 256 , 192).astype(np.float32)
    input = torch.from_numpy(input).cuda()
    output = model(input)
    print (output)

    pass

if __name__ == '__main__':
    main()