import torch 
import torch.nn as nn 
import torchvision
from config import get_config
import numpy as np 
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

def main():
    device = torch.device('cuda:0')
    model = resnet_fc('Make3D')

    model.to(device)
    input = np.random.randn(4, 3, 256 , 192).astype(np.float32)
    input = torch.from_numpy(input).cuda()
    output = model(input)
    print (output.shape)

    pass

if __name__ == '__main__':
    main()