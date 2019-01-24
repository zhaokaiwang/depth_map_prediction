import torch
import torch.nn as nn 
import torch.functional as F 
import os 
import numpy as np 
from config import get_config

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 48, 3, 2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),

            nn.Conv2d(48, 96, 3, 2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),

            nn.Conv2d(96, 192, 3, 2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),

            nn.Conv2d(192, 384, 3, 2, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            nn.Conv2d(384, 768, 3, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential( 
            nn.Linear(26880, 1, bias=False),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
        
    def forward(self, input):
        output = self.main(input)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)

        return output


class WganDiscriminator(nn.Module):
    def __init__(self):
        super(WganDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 48, 3, 2, bias=False),
            nn.ReLU(True),

            nn.Conv2d(48, 96, 3, 2, bias=False),
            nn.ReLU(True),

            nn.Conv2d(96, 192, 3, 2, bias=False),
            nn.ReLU(True),

            nn.Conv2d(192, 384, 3, 2, bias=False),
            nn.ReLU(True),

            nn.Conv2d(384, 768, 3, 2, bias=False),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential( 
            nn.Linear(26880, 1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
        
    def forward(self, input):
        output = self.main(input)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)

        return output 

def calcGradientPenalty(netD, real_data, fake_data):
    batch_size = real_data.shape[0]
    temp = torch.rand(batch_size, 1)
    temp = temp.expand(batch_size, 3 * 64 * 64).contiguous().view(batch_size, 3, 64, 64).cuda()

    input = real_data * temp + fake_data * (1. - temp)
    input = input.requires_grad_(True)
    output = netD(input)

    grad = torch.autograd.grad(outputs=output, inputs=input,
        grad_outputs=torch.ones(output.size()).cuda(), create_graph=True,
        retain_graph=True, only_inputs=True)[0]

    gradientPenalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * LAMDBDA

    return gradientPenalty 
def main():
    device = torch.device('cuda:0')
    netD = Discriminator()
    netD.train()
    netD.to(device)
    
    input = torch.rand(64, 1, 64, 64).to(device)
    output = netD(input)
    print (output.shape)
    pass

if __name__ == "__main__":
    main()